"""
readfiles_func is a function to read organized files used on LIdar Pre-ANalysis CORrection Algorithm - LIPANCORA
Created on Fri Jun 18 07:02:05 2021
@author: Fábio J. S. Lopes
"""

import locale
import os
from pathlib import Path

import numpy as np
import pandas as pd

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")


def readfiles_libids(datadir_name):
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


def readfiles_generic(datadir_name):
    """Return a list with all filenames from all subdirectories."""
    filepath = []

    for dirpath, dirnames, files in os.walk(datadir_name):
        dirnames.sort()
        files.sort()
        for file in files:
            filepath.append(os.path.join(dirpath, file))

    return filepath


def readfiles_meastype(datadir_name):
    """Return a list of directories for measurement days and types, i.e., dark current and atmospheric measurements."""
    filepath = []
    meas_date = []
    meas_type = []
    pathdirname = []

    for dirpath, dirnames, files in os.walk(datadir_name):
        dirnames.sort()
        files.sort()
        for file in files:
            filepath.append(os.path.join(dirpath, file))
            meas_date.append(Path(os.path.relpath(dirpath, datadir_name)).parts[-2])
            meas_type.append(Path(os.path.relpath(dirpath, datadir_name)).parts[-1])

    datefolder, counts = np.unique(meas_date, return_counts=True)
    datefoldertype, counts = np.unique(meas_type, return_counts=True)
    subfolders = [f.path for f in os.scandir(datadir_name) if f.is_dir()]
    subfolders.reverse()

    for i in range(len(subfolders)):
        for j in range(len(datefolder)):
            if datefolder[j][0:4] == os.path.basename(subfolders[i]):
                pathdirname.append(os.path.join(subfolders[i], datefolder[j]))

    return pathdirname, datefoldertype


def folder_creation(csvfiledir):
    if not os.path.exists(csvfiledir):
        try:
            os.makedirs(csvfiledir)
        except OSError:
            print(
                "Creation of the CSV file directory % s failed"
                % Path(os.path.relpath(csvfiledir)).parts[-1]
            )
        else:
            print(
                "Successfully created the CSV file directory % s"
                % Path(os.path.relpath(csvfiledir)).parts[-1]
            )


def writedown_header(
    filepath_towrite,
    line1,
    line2,
    line3,
    line4,
    line5,
    line6,
    line7,
    line8,
    line9,
    line10,
):
    ff = open(filepath_towrite, "r")
    lines = ff.readlines()
    ff = open(filepath_towrite, "w")
    ff.write(line1)
    ff.write(line2)
    ff.write(line3)
    ff.write(line4)
    ff.write(line5)
    ff.write(line6)
    ff.write(line7)
    ff.write(line8)
    ff.write(line9)
    ff.write(line10)
    ff = open(filepath_towrite, "a")
    ff.writelines(lines)
    ff.close()


def readdown_header(filepath):
    dictsetup = {}
    rows_list = []
    with open(filepath, mode="r") as read_obj:
        for line in read_obj:
            data = line.split()
            if len(data) == 2:
                key, value = data[0], data[1]
                dictsetup[key] = value
        rows_list = dictsetup.copy()
    #        rows_list.append(dictsetup.copy())
    #        dfdict = pd.DataFrame(rows_list)
    return rows_list


def rebind(
    rawdata, deadtime, rootdir_name, datadir_name, files_dir_level0, files_dir_level1
):
    """Save Level 0 database - raw signal as csv format with no pre-corrections.
    Return the mean dark-current file and the measurements files with dead-time correction applied.
    """
    starttimeaux = []
    stoptimeaux = []
    filenameaux = []
    rawdatafiles = []
    csv_files_pathaux = []
    meandc_df = pd.DataFrame()

    """Reading binary data and its header"""
    for j in range(len(rawdata)):
        f = open(rawdata[j], "rb")
        line = f.readline().decode("utf-8")  # noqa: F841
        aux = []
        """14 is the line number in the header of licel binary data"""
        """ If measurement is from January 23 2022 the number should be 12 """
        for ix in range(14):
            aux.append(f.readline().decode("utf-8"))

        """Reading binary data heads (2nd line) to select number of shots, start and stop time of measurements"""
        start_time = aux[0][10:20] + "-" + aux[0][21:29]
        stop_time = aux[0][30:40] + "-" + aux[0][41:49]
        alt_station = aux[0][51:54]
        site = aux[0][1:4]
        if site == "Sao":
            site = site + "_Paulo"
        lat = aux[0][62:63] + aux[0][64:68]
        long = aux[0][55:56] + aux[0][57:61]
        laser_freq = aux[1][10:13]
        n_channel = int(aux[1][27:29])
        n_bit = []
        norm = []
        channel = []
        n_bins = []
        n_shots = []
        n_pcfactor = []
        n_resolution = []
        flag_channel = []

        for k in range(2, n_channel + 2):
            if aux[k][3] == "0":
                flag_channel.append(int(aux[k][3]))
                n_bit.append(int(aux[k][44:46]))
                norm.append(int(float(aux[k][54:59]) * 1e3))
            else:
                flag_channel.append("1")
                n_bit.append(16)
                norm.append(20)

            channel.append(str(int(aux[k][25:30])))
            n_bins.append(int(aux[k][7:12]))
            n_shots.append(int(aux[k][47:53]))
            n_resolution.append(float(aux[k][20:23]))
            n_pcfactor.append(2 * (int(aux[k][7:12])) + 1)

        df = pd.DataFrame()
        dfdeadtime = pd.DataFrame()
        f.readline().decode("utf-8")

        for ii in range(int(n_channel / 2)):
            binarydata = np.fromfile(f, np.int32, n_pcfactor[ii])

            an = []
            for m in range(n_bins[ii]):
                an.append(
                    (binarydata[m] / 2 ** n_bit[2 * ii])
                    * (norm[2 * ii] / n_shots[2 * ii])
                )

            pc = []
            for m in range(n_bins[ii], 2 * n_bins[ii]):
                pc.append(
                    round(binarydata[m] / 2 ** n_bit[2 * ii + 1])
                    * (norm[2 * ii + 1] / n_shots[2 * ii + 1])
                )

            df[channel[2 * ii] + "AN"] = an
            df[channel[2 * ii + 1] + "PC"] = pc
            dfdeadtime[channel[2 * ii] + "AN"] = df[channel[2 * ii] + "AN"].apply(
                lambda x: x / (1 - x * deadtime[2 * ii])
            )
            dfdeadtime[channel[2 * ii + 1] + "PC"] = df[
                channel[2 * ii + 1] + "PC"
            ].apply(lambda x: x / (1 - x * deadtime[2 * ii + 1]))

        rawdatafiles.append(dfdeadtime)

        yeardir = Path(os.path.relpath(rawdata[j], datadir_name)).parts[-4]
        datedir = Path(os.path.relpath(rawdata[j], datadir_name)).parts[-3]
        meastypedir = Path(os.path.relpath(rawdata[j], datadir_name)).parts[-2]
        filename = Path(os.path.relpath(rawdata[j], datadir_name)).parts[-1] + "_level0"
        csv_dir = os.path.join(
            rootdir_name, files_dir_level0, yeardir, datedir, meastypedir
        )
        csv_files_path = os.path.join(csv_dir, filename)

        filenameaux.append(filename)
        starttimeaux.append(start_time)
        stoptimeaux.append(stop_time)
        csv_files_pathaux.append(csv_files_path)

        dictsetup = {
            "start_time": starttimeaux,
            "stop_time": stoptimeaux,
            "nshots": n_shots,
            "vert_res": n_resolution,
            "nbins": n_bins,
            "laser_freq": laser_freq,
            "site": site,
            "altitude": alt_station,
            "lat": lat,
            "long": long,
        }

        folder_creation(csv_dir)

        df.to_csv(csv_files_path, index=False, float_format="%.4f")

        """writing down the header for Raw data files """
        line1 = " ".join(["station " + str(site) + "\n"])
        line2 = " ".join(["altitude " + str(alt_station) + "\n"])
        line3 = " ".join(["lat " + str(lat) + "\n"])
        line4 = " ".join(["long " + str(long) + "\n"])
        line5 = " ".join(["starttime " + str(start_time) + "\n"])
        line6 = " ".join(["stoptime " + str(stop_time) + "\n"])
        line7 = " ".join(["bins " + str(n_bins[0]) + "\n"])
        line8 = " ".join(["vert_res " + str(n_resolution[0]) + "\n"])
        line9 = " ".join(["shotnumber " + str(n_shots[0]) + "\n"])
        line10 = " ".join(["laser_freq " + laser_freq + "\n"])
        writedown_header(
            csv_files_path,
            line1,
            line2,
            line3,
            line4,
            line5,
            line6,
            line7,
            line8,
            line9,
            line10,
        )

    """RETURN ALSO dfdeadtime"""
    if meastypedir == "measurements":
        return rawdatafiles, dictsetup, filenameaux, csv_files_pathaux
    else:
        for ch in list(rawdatafiles[0].columns):
            meandc_df[ch] = pd.concat(
                [
                    pd.DataFrame(rawdatafiles[ik], columns=[ch])
                    for ik in range(len(rawdatafiles))
                ],
                axis=1,
                join="inner",
            ).mean(axis=1)

        meandc_csv_dir = os.path.join(
            rootdir_name, files_dir_level1, yeardir, datedir, "01-mean_" + meastypedir
        )
        meandc_csv_files = os.path.join(
            meandc_csv_dir,
            "".join(
                [
                    "meandc_",
                    filenameaux[0][1:8],
                    filenameaux[0][9:11],
                    "_",
                    filenameaux[-1][1:8],
                    filenameaux[-1][9:11],
                ]
            ),
        )

        folder_creation(meandc_csv_dir)
        meandc_df.to_csv(meandc_csv_files, index=False, float_format="%.4f")

        line11 = " ".join(["station " + str(site) + "\n"])
        line12 = " ".join(["altitude " + str(alt_station) + "\n"])
        line13 = " ".join(["lat " + str(lat) + "\n"])
        line14 = " ".join(["long " + str(long) + "\n"])
        line15 = " ".join(
            [
                "starttime1 " + str(starttimeaux[0]) + ",",
                "stoptime1 " + str(stoptimeaux[0]) + "\n",
            ]
        )
        line16 = " ".join(
            [
                "starttime2 " + str(starttimeaux[-1]) + ",",
                "stoptime2 " + str(stoptimeaux[-1]) + "\n",
            ]
        )
        line17 = " ".join(["bins " + str(n_bins[0]) + "\n"])
        line18 = " ".join(["vert_res " + str(n_resolution[0]) + "\n"])
        line19 = " ".join(["shotnumber " + str(n_shots[0]) + "\n"])
        line20 = " ".join(["laser_freq " + laser_freq + "\n"])
        writedown_header(
            meandc_csv_files,
            line11,
            line12,
            line13,
            line14,
            line15,
            line16,
            line17,
            line18,
            line19,
            line20,
        )

        return meandc_df


def binshift_function(binshiftcorr, dtrawdata):
    for j in range(len(dtrawdata.columns)):
        if binshiftcorr[j] > 0:
            value = dtrawdata.iloc[-1][dtrawdata.columns[j]]
        else:
            value = dtrawdata.iloc[0][dtrawdata.columns[j]]
        dtrawdata.iloc[:, [j]] = dtrawdata.iloc[:, [j]].shift(
            -binshiftcorr[j], fill_value=value
        )

    return dtrawdata
