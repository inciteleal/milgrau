"""
LIdar RAnge COrrection Signal - LIRACOS
This script provides tools to handle with range corrected signal graphics and RCS maps, 
so-called quicklooks graphics. This script uses range corrected data with all corrections
applyied by LIPANCORA scripts
@author: Fábio J. S. Lopes, Alexandre C. Yoshida Alexandre Cacheffo, Luisa Mello
"""

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor

# ==========================================
# SETTINGS
# ==========================================
INCREMENTAL_PROCESSING = False  
MAX_WORKERS = 1  

rootdir_name = os.getcwd() 
files_dir_level1 = "05-data_level1"
base_data_folder = os.path.join(rootdir_name, files_dir_level1)

ALTITUDE_RANGES = [5, 15, 30] # km
VERTICAL_RESOLUTION_M = 7.5

channels_to_plot = [
    "01064.o_an", 
    "00532.o_an", "00532.o_ph",
    "00355.o_an", "00355.o_ph",
]

LOGO_LEAL = os.path.join(rootdir_name, "img", "logo_leal.png")
LOGO_INCITE = os.path.join(rootdir_name, "img", "Logo_InCite_blue_site.png")

# ==========================================
# FORMATTING
# ==========================================
def extract_datetime_strings(ds):
    try:
        dt_in_str = f"{ds.attrs['RawData_Start_Date']}{str(ds.attrs['RawData_Start_Time_UT']).zfill(6)}"
        dt_end_str = f"{ds.attrs['RawData_Start_Date']}{str(ds.attrs['RawData_Stop_Time_UT']).zfill(6)}"
        
        dt_in = datetime.strptime(dt_in_str, "%Y%m%d%H%M%S")
        dt_end = datetime.strptime(dt_end_str, "%Y%m%d%H%M%S")
        
        date_title = f"{dt_in.strftime('%d %b %Y - %H:%M')} to {dt_end.strftime('%d %b %Y - %H:%M')} UTC"
        date_footer = dt_in.strftime("%d %b %Y")
        return date_title, date_footer
    except Exception:
        return "Unknown date", "Unknown date"

def format_channel_name(raw_name):
    try:
        parts = raw_name.split('.')
        wavelength = int(parts[0])             
        mode = parts[1].split('_')[1].upper()  
        if mode == 'PH': 
            mode = 'PC'
            return f"{wavelength}nm {mode}"
        return f"{wavelength}nm {mode}"
    except Exception:
        return raw_name 

def add_footer_and_logos(fig, date_footer):
    fig.text(0.10, 0.03, date_footer, fontsize=13, fontweight="bold", va="center")
    fig.text(0.70, 0.03, "LEAL-IPEN-LALINET", fontweight="bold", fontsize=12, color="black", ha="right", va="center")
    
    if os.path.exists(LOGO_LEAL):
        newax_logo = fig.add_axes([0.72, 0.01, 0.09, 0.06], zorder=12)
        newax_logo.imshow(mpimg.imread(LOGO_LEAL), alpha=1, aspect="equal")
        newax_logo.axis("off")
        
    if os.path.exists(LOGO_INCITE):
        newax_incite = fig.add_axes([0.82, 0.01, 0.08, 0.06], zorder=12)
        newax_incite.imshow(mpimg.imread(LOGO_INCITE), alpha=0.9, aspect="equal")
        newax_incite.axis("off")

# ==========================================
#  QUICKLOOK (COLORMAP + MEAN RCS)
# ==========================================
def plot_quicklook(data_slice, max_altitude, channel_name, ds, output_folder, file_name_prefix):
    date_title, date_footer = extract_datetime_strings(ds)
    pretty_channel = format_channel_name(channel_name)
    meas_title = f"RCS at {pretty_channel} (0 - {max_altitude} km)\n{date_title}\nSPU Lidar Station - São Paulo"
    
    fig = plt.figure(figsize=[15, 7.5])
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.03)
    
    # colormap axis
    ax0 = plt.subplot(gs[0])
    
    plot = data_slice.plot(
        x='time', y='altitude',
        cmap='jet', robust=True, vmin=0, add_colorbar=False, ax=ax0, add_labels=False
    )
    
    ax0.set_title(meas_title, fontsize=15, fontweight="bold", loc='Center')
    ax0.set_xlabel('Time (UTC)', fontsize=13, fontweight="bold")
    ax0.set_ylabel('Altitude (km a.g.l.)', fontsize=13, fontweight="bold")
    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # mean rcs axis
    ax1 = plt.subplot(gs[1], sharey=ax0)
    
    mean_profile = data_slice.mean(dim='time')
    smooth_profile = mean_profile.rolling(altitude=20, min_periods=1).mean()
    
    cor_linha = "black"
    if "532" in channel_name: cor_linha = "forestgreen"
    elif "355" in channel_name: cor_linha = "rebeccapurple"
    elif "1064" in channel_name: cor_linha = "crimson"
    elif "387" in channel_name: cor_linha = "darkblue"
    elif "408" in channel_name: cor_linha = "darkcyan"
    elif "530" in channel_name: cor_linha = "orange"
    
    ax1.plot(smooth_profile, smooth_profile.altitude, color=cor_linha, linewidth=2)
    ax1.fill_betweenx(smooth_profile.altitude, 0, smooth_profile, color=cor_linha, alpha=0.2)
    
    ax1.set_xlabel('Mean RCS', fontsize=12, fontweight="bold")
    # ax1.set_xscale('log') 
    plt.setp(ax1.get_yticklabels(), visible=False) 
    ax1.grid(True, linestyle='--', alpha=0.6, which='both')
    
    # margin for colorbar
    plt.subplots_adjust(left=0.14, bottom=0.15, right=0.95, top=0.88)
    
    # colorbar
    cb_ax = fig.add_axes([0.06, 0.15, 0.015, 0.73]) # X, Y, W, H
    cb = fig.colorbar(plot, cax=cb_ax, orientation='vertical')
    cb.set_label("Intensity [a.u.]", fontsize=12, fontweight="bold")
    cb_ax.yaxis.set_ticks_position('left')
    cb_ax.yaxis.set_label_position('left')
    
    add_footer_and_logos(fig, date_footer)
    
    safe_channel_name = pretty_channel.replace(" ", "_")
    file_name = f'Quicklook_{file_name_prefix}_{safe_channel_name}_{max_altitude}km.webp'
    
    plt.savefig(os.path.join(output_folder, file_name), dpi=120)
    plt.close(fig)

# ==========================================
# MEAN RCS 
# ==========================================
def plot_global_mean_rcs(ds, output_folder, file_name_prefix):
    max_altitude = max(ALTITUDE_RANGES) 
    date_title, date_footer = extract_datetime_strings(ds)
    meas_title = f"Mean RCS (0 - {max_altitude} km)\n{date_title}\nSPU Lidar Station - São Paulo"
    
    fig, ax = plt.subplots(figsize=(8, 9.6))
    fig.subplots_adjust(top=0.90, bottom=0.15)
    
    cores_base = { 355: "rebeccapurple", 387: "darkblue", 408: "darkcyan", 530: "orange", 532: "forestgreen", 1064: "crimson" }
    plotou_algo = False
    
    for ch in channels_to_plot:
        if ch in ds.channel.values:
            label = format_channel_name(ch)
            try:
                wavelength = int(ch.split('.')[0])
                cor = cores_base.get(wavelength, "black")
            except Exception:
                cor = "black"
            
            estilo = "-" if "an" in ch.lower() else "--"
            rc_signal = ds['Range_Corrected_Signal'].sel(channel=ch)
            rc_slice = rc_signal.where(rc_signal['altitude'] <= max_altitude, drop=True)
            
            mean_profile = rc_slice.mean(dim='time')
            smooth_profile = mean_profile.rolling(altitude=50, min_periods=1).mean()
            
            ax.plot(smooth_profile, smooth_profile.altitude, color=cor, linestyle=estilo, label=label, linewidth=2)
            plotou_algo = True
            
    if not plotou_algo:
        plt.close(fig)
        return

    ax.set_title(meas_title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean RCS [a.u.]", fontsize=14, fontweight="bold")
    ax.set_ylabel("Altitude (km a.g.l.)", fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.set_ylim(0, max_altitude)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, which='both', alpha=0.5)
    
    add_footer_and_logos(fig, date_footer)
    
    file_name = f'MeanRCS_{file_name_prefix}.webp'
    plt.savefig(os.path.join(output_folder, file_name), dpi=120)
    plt.close(fig)

# ==========================================
# MAIN PROCESSING
# ==========================================

def process_single_nc(nc_file):
    try:
        file_name_prefix = os.path.basename(nc_file).replace('_level1_rcs.nc', '')
        
        base_folder = os.path.dirname(nc_file) 
        output_folder = os.path.join(base_folder, "quicklooks")
        os.makedirs(output_folder, exist_ok=True)
        
        check_file = os.path.join(output_folder, f'GlobalMeanRCS_{file_name_prefix}.webp')
        if INCREMENTAL_PROCESSING and os.path.exists(check_file):
            return f"  -> [SKIPPED] File exists: {file_name_prefix}"

        ds = xr.open_dataset(nc_file)
        ds.load()

        num_bins = ds.sizes['range']
        new_altitude_km = np.arange(0, num_bins * VERTICAL_RESOLUTION_M, VERTICAL_RESOLUTION_M) / 1000.0
        ds = ds.assign_coords(altitude=("range", new_altitude_km))
        ds = ds.swap_dims({'range': 'altitude'})
        
        try:
            dt_in_str = f"{ds.attrs['RawData_Start_Date']}{str(ds.attrs['RawData_Start_Time_UT']).zfill(6)}"
            dt_end_str = f"{ds.attrs['RawData_Start_Date']}{str(ds.attrs['RawData_Stop_Time_UT']).zfill(6)}"
            dt_in = datetime.strptime(dt_in_str, "%Y%m%d%H%M%S")
            dt_end = datetime.strptime(dt_end_str, "%Y%m%d%H%M%S")
            if dt_end < dt_in:
                dt_end += timedelta(days=1)
            time_array = pd.date_range(start=dt_in, end=dt_end, periods=ds.sizes['time'])
            ds = ds.assign_coords(time=time_array)
        except Exception:
            pass 
        
        for channel_name in channels_to_plot:
            if channel_name in ds.channel.values:
                rc_signal = ds['Range_Corrected_Signal'].sel(channel=channel_name)
                for max_altitude in ALTITUDE_RANGES:
                    data_slice = rc_signal.where(rc_signal['altitude'] <= max_altitude, drop=True)
                    plot_quicklook(data_slice, max_altitude, channel_name, ds, output_folder, file_name_prefix)

        plot_global_mean_rcs(ds, output_folder, file_name_prefix)
        
        ds.close()
        return f"  -> [OK] Plots saved: {file_name_prefix}"

    except Exception as e:
        return f"  -> [ERROR]: {os.path.basename(nc_file)}: {e}"


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    file_pattern = os.path.join(base_data_folder, '**', '*rcs.nc')
    nc_files = glob.glob(file_pattern, recursive=True)

    if not nc_files:
        print(f"[INFO] No level-1 data found in {base_data_folder}")
    else:
        modo = "Incremental" if INCREMENTAL_PROCESSING else "Rewriting"
        print(f"[INFO] LIRACOS: Iniciating process ({modo})...")
        
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = executor.map(process_single_nc, nc_files)
            for res in results:
                print(res)
                
        print("[INFO] processing finalized successfully!")