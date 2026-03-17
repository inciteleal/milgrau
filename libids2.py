"""
LIdar BInary Data Standardized - LIBIDS
Script to read raw lidar binary data, clean up spurious data (temp.dat, AutoSave.dpp, 
invalid laser shots), and directly convert the valid data into standardized NETCDF format 
for the Single Calculus Chain (SCC) algorithm.

@author: Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo, Luisa Mello
"""

import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from statistics import mode
import pandas as pd
from atmospheric_lidar.licel import LicelLidarMeasurement
from atmospheric_lidar_parameters import (
    msp_netcdf_parameters_system484,
    msp_netcdf_parameters_system565,
)

# ==========================================
# SETTINGS
# ==========================================
INCREMENTAL_PROCESSING = True  # True skips processed files, False rewrites processed files
MAX_WORKERS = 1  # Multiprocessing: 1= none, 2-8 threads

ROOT_DIR = os.getcwd()
FILES_DIR_STAND = "01-data"
NETCDF_DIR = "03-netcdf_data"

DATADIR = os.path.join(ROOT_DIR, FILES_DIR_STAND)

def readfiles(datadir_name):
    """Return a list with all filenames from all subdirectories for the original raw lidar database."""
    filepath = []
    flag_period_files = []
    meas_type = []

    for dirpath, dirnames, files in os.walk(datadir_name):
        dirnames.sort()
        files.sort()
        for file in files:
            if file.endswith(".dat"):
                os.remove(os.path.join(dirpath, file))
                print("temp.dat file deleted")
            elif file.endswith(".dpp"):
                os.remove(os.path.join(dirpath, file))
                print(".dpp file deleted")
            else:
                filepath.append(os.path.join(dirpath, file))

                if Path(os.path.relpath(dirpath, datadir_name)).parts[3][12:] == "day":
                    if (
                        Path(os.path.relpath(dirpath, datadir_name)).parts[4][-2:]
                        == "01"
                    ):
                        flag_period_files.append("am")
                    elif (
                        Path(os.path.relpath(dirpath, datadir_name)).parts[4][-2:]
                        == "02"
                    ):
                        flag_period_files.append("pm")
                else:
                    flag_period_files.append("nt")
                if (
                    Path(os.path.relpath(dirpath, datadir_name)).parts[-1]
                    == "dark_current"
                ):
                    meas_type.append("dark_current")
                else:
                    meas_type.append("measurements")
    return filepath, flag_period_files, meas_type

fileinfo = readfiles(DATADIR)

def make_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            pass 

def get_night_date(dt):
    if dt.hour < 6:
        dt = dt - pd.Timedelta(days=1)
    return dt

def read_header(filepath):
    with open(filepath, "rb") as f:
        _ = f.readline().decode("utf-8")
        lines = [f.readline().decode("utf-8") for _ in range(3)]
    start_time_str = lines[0][10:29].strip()
    stop_time_str = lines[0][30:49].strip()
    n_shots = int(lines[1][16:21])
    laser_freq = int(lines[1][22:27])
    start_time = datetime.strptime(start_time_str, "%d/%m/%Y %H:%M:%S")
    stop_time = datetime.strptime(stop_time_str, "%d/%m/%Y %H:%M:%S")
    duration = (stop_time - start_time).total_seconds()
    return start_time, stop_time, duration, n_shots, laser_freq

def process_single_netcdf(meas_id, group):
    date_str = meas_id[:8]
    period = meas_id[8:]
    save_id = f"{date_str}sa{period}"
    netcdf_path = os.path.join(ROOT_DIR, NETCDF_DIR, f"{save_id}.nc")

    if INCREMENTAL_PROCESSING and os.path.exists(netcdf_path):
        return f"  -> [Skipped] File previously processed: {save_id}.nc"

    files_meas = group[group["meas_type"] == "measurements"]["filepath"].tolist()
    files_meas_dc = group[group["meas_type"] != "measurements"]["filepath"].tolist()

    if not files_meas:
        return f"  -> [ERROR] No measurement files for: {save_id}"

    if period in ["am", "pm"]:
        class mspLidarMeasurement(LicelLidarMeasurement):
            extra_netcdf_parameters = msp_netcdf_parameters_system565
    else:
        class mspLidarMeasurement(LicelLidarMeasurement):
            extra_netcdf_parameters = msp_netcdf_parameters_system484

    my_measurement = mspLidarMeasurement(files_meas)
    if files_meas_dc:
        my_dark_measurement = mspLidarMeasurement(files_meas_dc)
        my_measurement.dark_measurement = my_dark_measurement
        
    my_measurement.info["Measurement_ID"] = save_id
    my_measurement.info["Temperature"] = "25"
    my_measurement.info["Pressure"] = "940"
    
    duration = mode(group["duration"])
    freq = mode(group["laser_freq"])
    expected_shots = int(duration * freq)
    
    my_measurement.info["Accumulated_Shots"] = str(expected_shots)
    my_measurement.info["Laser_Frequency"] = str(freq)
    my_measurement.info["Measurement_Duration"] = str(duration)

    make_dir(os.path.join(ROOT_DIR, NETCDF_DIR))
    my_measurement.save_as_SCC_netcdf(netcdf_path)
    
    del my_measurement 
    
    return f"  -> [OK] File saved: {save_id}.nc"

# ==========================================
# MAIN
# ==========================================
print(f"[INFO] Reading files to process...")
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    results = list(executor.map(read_header, fileinfo[0]))

start_times, stop_times, durations, nshots_list, laser_freqs = zip(*results)

df = pd.DataFrame(
    {
        "filepath": fileinfo[0],
        "flag_period": fileinfo[1],
        "meas_type": fileinfo[2],
        "start_time": start_times,
        "stop_time": stop_times,
        "nshots": nshots_list,
        "duration": durations,
        "laser_freq": laser_freqs,
    }
)

df["meas_id"] = df["start_time"].apply(get_night_date).dt.strftime("%Y%m%d") + df["flag_period"]

df_good_list = []
df_bad_list = []


for (meas_id, meas_type), group in df.groupby(["meas_id"]):
    try:
        duration = mode(group["duration"])
        freq = mode(group["laser_freq"])
        expected_shots = duration*freq

        bad_condition = (
                (group["nshots"] == 0)
                | (group["nshots"] <= expected_shots - 2e-3 * expected_shots)
                | (group["nshots"] >= expected_shots + 2e-3 * expected_shots)
            )
            
        df_bad_list.append(group.loc[bad_condition])
        df_good_list.append(group.loc[~bad_condition])
        
    except Exception as e:
        print('  -> [WARNING]: Error verifying file condition: ', e)
        df_bad_list.append(group)

df_bad = pd.concat(df_bad_list).reset_index(drop=True) if df_bad_list else pd.DataFrame()
df_good = pd.concat(df_good_list).reset_index(drop=True) if df_good_list else pd.DataFrame()


# ==========================================
# REPORT ON BAD FILES
# ==========================================
total_files = len(df)
bad_files = len(df_bad)
good_files = len(df_good)

if total_files > 0:
    loss_percent = (bad_files / total_files) * 100
    print("\n" + "="*45)
    print(" REPORT ON BAD FILES")
    print("="*45)
    print(f" Processed files : {total_files}")
    print(f" Bad files    : {bad_files} ({loss_percent:.2f}%)")
    print("="*45 + "\n")

else:
    print("[WARNING] No file found.")

# ==========================================
# NETCDF CONVERSION
# ==========================================
if not df_good.empty:
    modo = "Incremental" if INCREMENTAL_PROCESSING else "Rewriting"
    print(f"[INFO] Converting to NetCDF SCC ({modo})...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for meas_id, group in df_good.groupby("meas_id"):
            futures.append(executor.submit(process_single_netcdf, meas_id, group))
            
        for future in futures:
            print(future.result())
            
    print("[INFO] LIBIDS processing finalized successfully!")