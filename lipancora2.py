"""
LIdar Pre-ANalysis CORrection Algorithm - LIPANCORA
This script reads standardized Level 0 SCC NetCDF files, applies physical 
corrections, and outputs Level 1 data (Corrected Lidar Data and Range Corrected Signal).

Corrections applied:
    - Deadtime correction
    - Bin-shift correction 
    - Background calculation and subtraction
    - Range corrected signal

@author: Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo, Luisa Mello
"""

import os
import numpy as np
import xarray as xr
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# ==========================================
# SETTINGS
# ==========================================
INCREMENTAL_PROCESSING = False  # True skips processed files, False rewrites processed files
MAX_WORKERS = 1  # Multiprocessing: 1= none, 2-8 threads

INPUT_DIR = "03-netcdf_data"
OUTPUT_DIR = "05-data_level1"

# Channels deadtime correction (µs), bin-shift and bg-cheating (MHz / mV)
CHANNELS = {
    "00355.o_ph": [0.0020, -2, 0.0005],
    "00355.o_an": [0.0000,  8, 0.0000],
    "00387.o_ph": [0.0000, -2, 0.0000],
    "00387.o_an": [0.0000,  8, 0.0000],
    "00408.o_ph": [0.0000, -2, 0.0000],
    "00408.o_an": [0.0000,  8, 0.0000],
    "00530.o_ph": [0.0000, -2, 0.0015],
    "00530.o_an": [0.0000,  7, 0.0000],
    "00532.o_ph": [0.0035, -3, 0.0015],
    "00532.o_an": [0.0000,  6, 0.0000],
    "01064.o_ph": [0.0000,  0, 0.0000],
    "01064.o_an": [0.0000,  1, 0.0000],
}

ID_TO_NAME = {
    934: "01064.o_an", 935: "01064.o_ph",
    722: "00532.o_an", 1593: "00532.o_an", 716: "00532.o_ph",
    1558: "00530.o_an", 1595: "00530.o_an", 1557: "00530.o_ph",
    737: "00355.o_an", 1594: "00355.o_an", 736: "00355.o_ph",
    749: "00387.o_an", 1596: "00387.o_an", 748: "00387.o_ph",
    1446: "00408.o_an", 1447: "00408.o_ph",
}

def apply_deadtime(signal, deadtime_us):
    """Applies deadtime correction"""
    if deadtime_us <= 0:
        return signal
    
    denom = 1.0 - (signal * deadtime_us)
    denom = np.where(denom <= 1e-6, np.nan, denom) 
    return signal / denom

def apply_binshift(signal, shift_bins):
    """ Shifts signal due to trigger delay"""
    if shift_bins == 0:
        return signal
    out = np.full_like(signal, np.nan)
    if shift_bins > 0:
        out[:, shift_bins:] = signal[:, :-shift_bins]
    else:
        k = abs(shift_bins)
        out[:, :-k] = signal[:, k:]
    return out

def compute_background(signal, z, bg_low_m, bg_high_m, offset=0.0):
    """Subtracts computed mean background noise in certain altitude range (m)"""
    i1 = np.searchsorted(z, bg_low_m)
    i2 = np.searchsorted(z, bg_high_m)
    if i1 > i2:
        i1, i2 = i2, i1
    if i1 == i2:
        i2 += 1 
        
    try:
        bg = np.nanmean(signal[..., i1:i2], axis=-1, keepdims=True) - offset
        return np.where(np.isnan(bg), 0.0, bg)
    except Exception as e:
        print('  -> [ERROR]: Background subtraction failed: ', e)


def process_file(nc_path):
    try:
        stem = Path(nc_path).stem
        year = stem[:4]
        base_dir = Path(OUTPUT_DIR) / year / stem
        
        corrected_path = base_dir / f"{stem}_level1_corrected.nc"
        rcs_path = base_dir / f"{stem}_level1_rcs.nc"

        # ==========================================================
        # INCREMENTAL CHECK
        # ==========================================================
        if INCREMENTAL_PROCESSING and corrected_path.exists() and rcs_path.exists():
            return f"  -> [SKIPPED] Correction exists for: {stem}"

        # Se não pulou, abre o arquivo para fazer a mágica da física
        ds = xr.open_dataset(nc_path)
        ds.load() 

        rename_dict = {}
        if 'channels' in ds.dims: rename_dict['channels'] = 'channel'
        if 'points' in ds.dims: rename_dict['points'] = 'range'
        if rename_dict: ds = ds.rename(rename_dict)

        raw = ds["Raw_Lidar_Data"].astype(np.float32)
        if list(raw.dims) != ["time", "channel", "range"]:
            raw = raw.transpose("time", "channel", "range")

        dz = float(ds.get("Raw_Data_Range_Resolution", [[7.5]])[0][0]) if "Raw_Data_Range_Resolution" in ds else 7.5
        z = np.arange(raw.sizes["range"], dtype=np.float32) * dz
        dr = np.mean(np.diff(z))

        corrected = np.full_like(raw.values, np.nan)
        rcs = np.full_like(raw.values, np.nan)

        channel_ids = ds["channel_ID"].values if "channel_ID" in ds else np.arange(raw.sizes["channel"])
        channel_names_scc = [ID_TO_NAME.get(int(cid), f"unknown_{cid}") for cid in channel_ids]

        bg_low_arr = ds["Background_Low"].values if "Background_Low" in ds else None
        bg_high_arr = ds["Background_High"].values if "Background_High" in ds else None

        c = 2.99792458e8
        bin_time_us = (2 * dr / c) * 1e6 
        shots = float(ds.attrs.get("Accumulated_Shots", 600.0))

        # ==========================================
        # CORRECTIONS
        # ==========================================
        for ch_i, ch_name in enumerate(channel_names_scc):
            sig = raw[:, ch_i, :].values.copy()
            deadtime, shift, bg_offset = CHANNELS.get(ch_name, (0.0, 0, 0.0))

            if "ph" in ch_name.lower() and np.nanmax(sig) > 1000:
                sig = sig / (shots * bin_time_us)

            sig = apply_deadtime(sig, deadtime)
            sig = apply_binshift(sig, shift)

            bg_low = float(bg_low_arr[ch_i]) if bg_low_arr is not None else 29000.0
            bg_high = float(bg_high_arr[ch_i]) if bg_high_arr is not None else 29999.0
            
            bg_val = compute_background(sig, z, bg_low, bg_high, bg_offset)
            sig -= bg_val

            corrected[:, ch_i, :] = sig
            rcs[:, ch_i, :] = sig * (z[np.newaxis, :] ** 2)

        # ==========================================
        # NETCDF CONFIGURATION
        # ==========================================
        attrs_common = dict(ds.attrs)
        attrs_common["processing_level"] = "Level 1: PC->MHz, Deadtime, Bin-Shift, Background"
        attrs_common["history"] = f"{ds.attrs.get('history', '')}\nProcessed with LIPANCORA on {datetime.now(timezone.utc).isoformat()} UTC"

        coords = {"time": ds["time"], "channel": ("channel", np.array(channel_names_scc)), "range": ("range", np.float32(z))}

        corrected_ds = xr.Dataset({"Corrected_Lidar_Data": (("time", "channel", "range"), corrected)}, coords=coords, attrs=attrs_common)
        rcs_ds = xr.Dataset({"Range_Corrected_Signal": (("time", "channel", "range"), rcs)}, coords=coords, attrs=attrs_common)

        corrected_ds["Corrected_Lidar_Data"].encoding.update(dtype="float32", _FillValue=np.nan)
        rcs_ds["Range_Corrected_Signal"].encoding.update(dtype="float32", _FillValue=np.nan)

        os.makedirs(base_dir, exist_ok=True)
        corrected_ds.to_netcdf(corrected_path)
        rcs_ds.to_netcdf(rcs_path)
        
        ds.close()
        return f"  -> [OK] Processed and saved: {stem}"

    except Exception as e:
        return f"  -> [ERROR] Error in file {Path(nc_path).name}: {str(e)}"

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    files = sorted(Path(INPUT_DIR).glob("*.nc"))
    if not files:
        print(f"[INFO] No raw data NetCDF found in '{INPUT_DIR}'")
    else:
        modo = "Incremental" if INCREMENTAL_PROCESSING else "Rewriting"
        print(f"[INFO] LIPANCORA: Iniciating process ({modo})...")
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = executor.map(process_file, files)
            for res in results:
                print(res)
                
        print("[INFO] LIPANCORA processing finalized successfully!")