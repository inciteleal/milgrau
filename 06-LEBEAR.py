"""
Lidar Elastic Backscatter and Extinction Analysis Routine - LEBEAR
This script provides tools to processes the pre-analyzed data and retrieval the backscatter and extinction profiles of the atmosphere, tunning the 
aerosol optical depth values measured with the lidar and the sun-photometer
Created on Sat Feb  5 07:51:19 2022
@author: Fábio J. S. Lopes
"""

import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from functions import milgrau_function as mf
from lidar_retrievals import glue, kfs, retrieval_plots
from molecular import lidarmolfit as lmfit

"""initial setup"""
rootdir_name = os.getcwd()
files_dir_level1 = "05-data_level1"
files_dir_to_read = "02-preprocessed_corrected"
rawinsonde_folder = "07-rawinsonde"
datadir_name = os.path.join(rootdir_name, files_dir_level1)


"""flag to calculate molecular profile """

atmospheric_flag = "us_std"  # 'radiosounding' for rawinsonde data or 'us_std' for US-standard atmosphere

"""Input data from user"""
lamb = 532  # elastic wavelength to be analyzed (1064, 532 and 355 nm)
glueflag = "yes"  # glueing flag --> 'yes' for glueing process, otherwise, 'no'
channelmode = "AN"  # channel mode --> analogic: 'AN' or photocounting: 'PC'
ini_molref_alt = 5000  # initial altitude range for molecular calibration
fin_molref_alt = 25000  # final altitude range for molecular calibration
optical_prop_scale = 1e6  # optical properties graphics unit in Mega-meter
altitude_scale = 1000  # altitude scale in km (a.g.l.)
altitude_min = 0.0  # minimum altitude range for bacckscatter and extinction graphics
altitude_max = 30  # minimum altitude range for bacckscatter and extinction graphics


lraerosol_month = {
    "01": 51,
    "02": 50,
    "03": 59,
    "04": 57,
    "05": 55,
    "06": 50,
    "07": 48,
    "08": 49,
    "09": 51,
    "10": 54,
    "11": 51,
    "12": 50,
}

fileinfo, subfolderinfo = mf.readfiles_meastype(datadir_name)
lraerosol = lraerosol_month[fileinfo[1][-6:-4]]


altitude_min_01 = 0  # minimum altitude range for scattering ratio graphic 01
altitude_max_01 = 10  # maximum altitude range for scattering ratio graphic 01
altitude_min_02 = 10  # minimum altitude range for scattering ratio graphic 02
altitude_max_02 = 30  # maximum altitude range for scattering ratio graphic 02
base_altitude = 13.8  # volcanic base plume altitude (visual)
top_altitude = 20  # volcanic top plume altitude (visual)


"""Reading all measurements directory"""
fileinfo, subfolderinfo = mf.readfiles_meastype(datadir_name)

for i in range(len(fileinfo)):
    preprocessedsignal = []
    datafiles = []
    filenameheader = []
    preprocessedtime = []

    for j in range(len(subfolderinfo)):
        datafiles.append(
            mf.readfiles_generic(os.path.join(fileinfo[i], subfolderinfo[j]))
        )
        if subfolderinfo[j] == files_dir_to_read:
            for filename in datafiles[j]:
                preprocessedsignal.append(
                    pd.read_csv(filename, sep=",", skiprows=range(0, 10))
                )
                filenameheader.append(mf.readdown_header(filename))
                dfdict = pd.DataFrame(filenameheader)

    for k in range(len(preprocessedsignal)):
        alt = pd.DataFrame(list(range(len(preprocessedsignal[k].index)))).mul(
            float(dfdict["vert_res"][k])
        ) + float(dfdict["vert_res"][k])
        alt.columns = ["altitude"]
        preprocessedsignalmean = pd.concat(preprocessedsignal).groupby(level=0).mean()


""" Calling the glue function to gluing Analogic and Photocounting channels - glue.py file in lidar_retrievals folder"""

if glueflag == "yes":
    window_length = 50
    correlation_threshold = 0.95
    intercept_threshold = 0.5
    gaussian_threshold = 0.1
    minmax_threshold = 0.5
    min_idx = 200  # 200 * 7.5 = 1500 m ,
    max_idx = 2000  # 2000 * 7.5 = 15000 m

    glued_signal, gluing_central_idx, score, c_lower, c_upper = glue.glue_signals_1d(
        preprocessedsignalmean[str(lamb) + "AN"].to_numpy(),
        preprocessedsignalmean[str(lamb) + "PC"].to_numpy(),
        window_length,
        correlation_threshold,
        intercept_threshold,
        gaussian_threshold,
        minmax_threshold,
        min_idx,
        max_idx,
    )

    """ Calling the Glueing Graphics Plot from LIDAR - GGPLIDAR function - retrieval_plots.py file in lidar_retrievals folder"""

    retrieval_plots.ggplidar(
        preprocessedsignalmean[str(lamb) + "AN"].to_numpy(),
        preprocessedsignalmean[str(lamb) + "PC"].to_numpy(),
        glued_signal,
        alt["altitude"],
        gluing_central_idx,
        window_length,
    )


""" Calling the lmfit function to calculate atmospheric molecular extinction and backscatter using radiosounding data"""

if glueflag == "yes":
    channelmode = "GL"
    dfglueing = pd.DataFrame(glued_signal[0], columns=["glued"])
    beta_molecular, simulated_signal = lmfit.lidarmolfit(
        dfdict["station"][0],
        atmospheric_flag,
        filenameheader,
        dfglueing["glued"],
        ini_molref_alt,
        fin_molref_alt,
        lamb,
        channelmode,
        rawinsonde_folder,
    )
    rcs = np.multiply(dfglueing["glued"].values.tolist(), np.power(alt["altitude"], 2))

else:
    beta_molecular, simulated_signal = lmfit.lidarmolfit(
        dfdict["station"][0],
        atmospheric_flag,
        filenameheader,
        preprocessedsignalmean[str(lamb) + channelmode],
        ini_molref_alt,
        fin_molref_alt,
        lamb,
        channelmode,
        rawinsonde_folder,
    )
    rcs = np.multiply(
        preprocessedsignalmean[str(lamb) + channelmode].values.tolist(),
        np.power(alt["altitude"], 2),
    )

"""KLETT-FERNALD-SASANO INVERSION"""

reference_range = int(
    (
        fin_molref_alt / float(dfdict["vert_res"][0])
        - ini_molref_alt / float(dfdict["vert_res"][0])
    )
    / 2
)
index_reference = int(ini_molref_alt / float(dfdict["vert_res"][0])) + reference_range
beta_aerosol_reference = 0
bin_length = float(dfdict["vert_res"][0])
lidar_ratio_molecular = 8.37758041
# rcs = np.multiply(preprocessedsignalmean[str(lamb)+channelmode].values.tolist(),np.power(alt['altitude'],2))

aerosol_backscatter = kfs.klett_backscatter_aerosol(
    rcs,
    lraerosol,
    beta_molecular,
    index_reference,
    reference_range,
    beta_aerosol_reference,
    bin_length,
    lidar_ratio_molecular,
)
aerosol_backscatter_smooth = savgol_filter(aerosol_backscatter.values.tolist(), 15, 3)
aerosol_extinction_smooth = savgol_filter(
    np.multiply(aerosol_backscatter.values.tolist(), lraerosol), 15, 3
)

retrieval_plots.kfs_plot(
    lamb,
    dfdict,
    lraerosol,
    alt["altitude"],
    aerosol_backscatter_smooth,
    aerosol_extinction_smooth,
    altitude_min,
    altitude_max,
    optical_prop_scale,
    altitude_scale,
    channelmode,
)

retrieval_plots.sr_plot(
    lamb,
    dfdict,
    alt,
    altitude_scale,
    channelmode,
    dfglueing["glued"],
    simulated_signal,
    altitude_min_01,
    altitude_max_01,
    altitude_min_02,
    altitude_max_02,
    base_altitude,
    top_altitude,
)

