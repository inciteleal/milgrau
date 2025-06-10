"""
LIdar Pre-ANalysis CORrection Algorithm - LIPANCORA
This script provides tools to handle raw binary lidar input converting Licel Binary to csv file as level 0 data with no corrections
applied.
This script uses the REad BINary Data - REBIND function to organize raw lidar binary data into standardized folder and save it as
level 0 data in .csv format.
LIPANCORA also applies all the pre-processed corrections to lidar raw data:
    -Deadtime correction
    -Dark current subtraction
    -First signal range-bin correction (zero-bin)
    -Trigger-delay correction (bin-shift)
    -Background calculation and subtraction
Created on Sun Jun 27 22:12:49 2021
@author: Fábio J. S. Lopes, Alexandre C. Yoshida and Alexandre Cacheffo
Adapted on 2025 by Luisa Mello
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from functions import milgrau_function as mf
from pathos.multiprocessing import ProcessingPool as Pool

# corrections, channels \/
# 1064AN,1064PC,532AN,532PC,530AN,530PC,355AN,355PC,387AN,387PC,408AN,408PC
deadtime = [0, 0, 0, 0.0035, 0, 0, 0, 0.002, 0, 0, 0, 0]
binshiftcorr = [1, -1, 6, -3, 7, -2, 8, -2, 8, -2, 8, -2]
bglow = 29000
bghigh = 29999
bgcheating = [0, 0, 0, 0.0015, 0, 0, 0, 0.0005, 0, 0, 0, 0]

rootdir_name = os.getcwd()
files_dir_stand = "02-data_raw_organized"
files_dir_level0 = "04-data_level0"
files_dir_level1 = "05-data_level1"
preprocessed_level1_dir = "02-preprocessed_corrected"
rcsignal_dir = "03-rcsignal"
datadir_name = os.path.join(rootdir_name, files_dir_stand)


def datetime2matlabdn(dt):
    mdn = dt + timedelta(days=366)
    frac_seconds = (dt - datetime(dt.year, dt.month, dt.day)).seconds / 86400.0
    frac_microseconds = dt.microsecond / (86400.0 * 1e6)
    return mdn.toordinal() + frac_seconds + frac_microseconds


def write_header(folder, filename, dictsetup, k):
    lines = [
        f"station {dictsetup['site']}\n",
        f"altitude {dictsetup['altitude']}\n",
        f"lat {dictsetup['lat']}\n",
        f"long {dictsetup['long']}\n",
        f"starttime {dictsetup['start_time'][k]}\n",
        f"stoptime {dictsetup['stop_time'][k]}\n",
        f"bins {dictsetup['nbins'][0]}\n",
        f"vert_res {dictsetup['vert_res'][0]}\n",
        f"shotnumber {dictsetup['nshots'][0]}\n",
        f"laser_freq {dictsetup['laser_freq']}\n",
    ]
    mf.writedown_header(os.path.join(folder, filename), *lines)


def process_measurement(args):
    i, fileinfo, subfolderinfo = args
    rawdata, dtrawdata, bsrawdata, rawdatabgcorrected, rcsignal = [], [], [], [], []
    meandcfiles, rawdatafiles, dictsetup, filenameaux, csv_files_path = (
        None,
        None,
        None,
        None,
        None,
    )

    for j in range(len(subfolderinfo)):
        data = mf.readfiles_generic(os.path.join(fileinfo[i], subfolderinfo[j]))
        if subfolderinfo[j] == "dark_current":
            meandcfiles = mf.rebind(
                data,
                deadtime,
                rootdir_name,
                datadir_name,
                files_dir_level0,
                files_dir_level1,
            )
        else:
            rawdatafiles, dictsetup, filenameaux, csv_files_path = mf.rebind(
                data,
                deadtime,
                rootdir_name,
                datadir_name,
                files_dir_level0,
                files_dir_level1,
            )

    if rawdatafiles is None or meandcfiles is None:
        print(f"[WARN] data incomplete for {fileinfo[i]}")
        return

    alt = (
        pd.DataFrame(list(range(len(rawdatafiles[0].index)))).mul(
            dictsetup["vert_res"][0]
        )
        + dictsetup["vert_res"][0]
    )
    alt.columns = ["altitude"]
    alt["rangesqrt"] = alt["altitude"].pow(2)

    yeardir = Path(os.path.relpath(csv_files_path[0], datadir_name)).parts[-4]
    datedir = Path(os.path.relpath(csv_files_path[0], datadir_name)).parts[-3]
    filenameaux_corr = [
        item.replace("level0", "level1_preprocessed") for item in filenameaux
    ]
    filenameaux_rcsignal = [
        item.replace("level0", "level1_rcsignal") for item in filenameaux
    ]
    csv_files_dir_corrected = os.path.join(
        rootdir_name, files_dir_level1, yeardir, datedir, preprocessed_level1_dir
    )
    csv_files_dir_rcsignal = csv_files_dir_corrected.replace(
        preprocessed_level1_dir, rcsignal_dir
    )
    mf.folder_creation(csv_files_dir_corrected)
    mf.folder_creation(csv_files_dir_rcsignal)

    for k in range(len(rawdatafiles)):
        dtraw = rawdatafiles[k].sub(meandcfiles, axis=0)
        bsraw = mf.binshift_function(binshiftcorr, dtraw)
        background = (
            bsraw.loc[
                int(bglow / dictsetup["vert_res"][0]) : int(
                    bghigh / dictsetup["vert_res"][0]
                )
            ].mean(axis=0)
            - bgcheating
        )
        corrected = bsraw.sub(background)
        rcs = corrected.mul(alt["rangesqrt"], axis=0)

        corrected.to_csv(
            os.path.join(csv_files_dir_corrected, filenameaux_corr[k]),
            index=False,
            float_format="%.4f",
        )
        rcs.to_csv(
            os.path.join(csv_files_dir_rcsignal, filenameaux_rcsignal[k]),
            index=False,
            float_format="%.4f",
        )

        write_header(csv_files_dir_corrected, filenameaux_corr[k], dictsetup, k)
        write_header(csv_files_dir_rcsignal, filenameaux_rcsignal[k], dictsetup, k)


if __name__ == "__main__":
    fileinfo, subfolderinfo = mf.readfiles_meastype(datadir_name)
    indices = list(range(len(fileinfo)))
    tasks = [(i, fileinfo, subfolderinfo) for i in indices]
    with Pool() as pool:
        pool.map(process_measurement, tasks)
