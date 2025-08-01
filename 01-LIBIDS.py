"""
LIdar BInary Data Standardized - LIBIDS
Script to organize raw lidar binary data into standardized folder and cleaning up
spurious data, such as, temp.dat, AutoSave.dpp, binary data with not accepted number of
laser shots. Also offers the option to convert the organized data to NETCDF files format
to be processed by Single Calculus Chain algorithm from EARLINET.

@author: Fábio J. S. Lopes, Alexandre C. Yoshida and Alexandre Cacheffo
Adapted on Apr2025 by Luisa Mello
"""

import glob
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from statistics import mode

import pandas as pd

# netcdf
from atmospheric_lidar.licel import LicelLidarMeasurement
from atmospheric_lidar_parameters import (
    msp_netcdf_parameters_system484,
    msp_netcdf_parameters_system565,
)
from functions.milgrau_function import readfiles_libids

RUN_NETCDF_CONVERSION = True

ROOT_DIR = os.getcwd()
FILES_DIR_STAND = "01-data"
BAD_FILES_DIR = "00-bad_files_dir"
FILES_DIR_ORGANIZED = "02-data_raw_organized"
NETCDF_DIR = "03-netcdf_data"

DATADIR = os.path.join(ROOT_DIR, FILES_DIR_STAND)
fileinfo = readfiles_libids(DATADIR)


def make_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            print(f"Creation of the file directory {path} failed")
        else:
            print(f"Successfully created the file directory {path}")


def move_file_to_target(row, base_dir):
    night_date = get_night_date(row.start_time)
    target = os.path.join(
        ROOT_DIR,
        base_dir,
        night_date.strftime("%Y"),
        f"{night_date.strftime('%Y%m%d')}{row.flag_period}",
        row.meas_type,
    )
    make_dir(target)
    shutil.copy(row.filepath, target)


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


def convert_to_netcdf():
    for path in glob.glob(os.path.join(ROOT_DIR, FILES_DIR_ORGANIZED, "*", "*")):
        meas_name = Path(
            os.path.relpath(path, os.sep.join([ROOT_DIR, FILES_DIR_ORGANIZED]))
        ).parts[1][:8]
        meas_period = Path(
            os.path.relpath(path, os.sep.join([ROOT_DIR, FILES_DIR_ORGANIZED]))
        ).parts[1][8:]
        save_id = f"{meas_name}sa{meas_period}"
        files_meas = []
        files_meas_dc = []

        if meas_period in ["am", "pm"]:
            print(
                "Day time",
                meas_period.upper(),
                "--> Using msp_netcdf_parameters_system565",
            )

            class mspLidarMeasurement(LicelLidarMeasurement):
                extra_netcdf_parameters = msp_netcdf_parameters_system565

        else:
            print("Night time period --> Using msp_netcdf_parameters_system484")

            class mspLidarMeasurement(LicelLidarMeasurement):
                extra_netcdf_parameters = msp_netcdf_parameters_system484

        for dir_meas in os.listdir(path):
            full_path = os.path.join(path, dir_meas)
            for file in os.listdir(full_path):
                if dir_meas == "measurements":
                    files_meas.append(os.path.join(full_path, file))
                else:
                    files_meas_dc.append(os.path.join(full_path, file))

        if files_meas:
            my_measurement = mspLidarMeasurement(files_meas)
            my_dark_measurement = mspLidarMeasurement(files_meas_dc)
            my_measurement.dark_measurement = my_dark_measurement
            my_measurement.info["Measurement_ID"] = save_id
            my_measurement.info["Temperature"] = "25"
            my_measurement.info["Pressure"] = "940"
            make_dir(os.path.join(ROOT_DIR, NETCDF_DIR))
            netcdf_path = os.path.join(ROOT_DIR, NETCDF_DIR, f"{save_id}.nc")
            my_measurement.save_as_SCC_netcdf(netcdf_path)
            print(f"NetCDF saved: {netcdf_path}")


with ThreadPoolExecutor() as executor:
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

df["meas_id"] = (
    df["start_time"].apply(get_night_date).dt.strftime("%Y%m%d") + df["flag_period"]
)
df_good_list = []
df_bad_list = []

for meas_id, group in df.groupby("meas_id"):
    duration = mode(group["duration"])
    freq = mode(group["laser_freq"])

    expected_shots = duration * freq
    bad_condition = (
        (group["nshots"] == 0)
        | (group["nshots"] < expected_shots - 2e-3 * expected_shots)
        | (group["nshots"] > expected_shots + 2e-3 * expected_shots)
    )
    df_bad_list.append(group.loc[bad_condition])
    df_good_list.append(group.loc[~bad_condition])

df_bad = pd.concat(df_bad_list).reset_index(drop=True)
df_good = pd.concat(df_good_list).reset_index(drop=True)

with ThreadPoolExecutor() as executor:
    executor.map(
        lambda row: move_file_to_target(row, BAD_FILES_DIR), df_bad.itertuples()
    )
    executor.map(
        lambda row: move_file_to_target(row, FILES_DIR_ORGANIZED), df_good.itertuples()
    )

print(f"[INFO] {len(df_bad)} files moved to '{BAD_FILES_DIR}'")
print(f"[INFO] {len(df_good)} files moved to '{FILES_DIR_ORGANIZED}'")

if RUN_NETCDF_CONVERSION:
    convert_to_netcdf()
