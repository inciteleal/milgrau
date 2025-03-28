
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
from molecular import lidarmolfit as lmfit
from lidar_retrievals import kfs
from lidar_retrievals import glue
from lidar_retrievals import raman_retrievals
from lidar_retrievals import retrieval_plots

'''initial setup'''
rootdir_name = os.getcwd()
files_dir_level1 = '05-data_level1'
files_dir_to_read = '02-preprocessed_corrected'
rawinsonde_folder = '06-rawinsonde'
datadir_name = os.path.join(rootdir_name,files_dir_level1)


'''flag to calculate molecular profile '''

atmospheric_flag = 'radiosounding'    # 'radiosounding' for rawinsonde data or 'us_std' for US-standard atmosphere

'''Input data from user'''
lamb = [532, 530]                   # elastic wavelength to be analyzed (1064, 532 and 355 nm)    
glueflag = 'yes'             # glueing flag --> 'yes' for glueing process, otherwise, 'no'
channelmode = 'AN'           # channel mode --> analogic: 'AN' or photocounting: 'PC'
ini_molref_alt = 6500        # initial altitude range for molecular calibration
fin_molref_alt = 7000        # final altitude range for molecular calibration
optical_prop_scale = 1e0     # optical properties graphics unit in Mega-meter
altitude_scale = 1000        # altitude scale in km (a.g.l.)
altitude_min = 0.0          # minimum altitude range for bacckscatter and extinction graphics  
altitude_max = 30            # minimum altitude range for bacckscatter and extinction graphics

lraerosol = 60               # initial lidar ratio values  


'''Reading all measurements directory'''
fileinfo, subfolderinfo = mf.readfiles_meastype(datadir_name)

for i in range(len(fileinfo)):
    preprocessedsignal=[]
    datafiles=[]
    filenameheader=[]
    preprocessedtime=[]

    for j in range(len(subfolderinfo)):
        datafiles.append(mf.readfiles_generic(os.path.join(fileinfo[i], subfolderinfo[j])))
        if subfolderinfo[j] == files_dir_to_read:
            for filename in datafiles[j]:
                preprocessedsignal.append(pd.read_csv(filename, sep = ',', skiprows=range(0, 10)))
                filenameheader.append(mf.readdown_header(filename))
                dfdict = pd.DataFrame(filenameheader)
                
    for k in range(len(preprocessedsignal)):
        alt = pd.DataFrame(list(range(len(preprocessedsignal[k].index)))).mul(float(dfdict['vert_res'][k]))+float(dfdict['vert_res'][k])
        alt.columns=['altitude']
        preprocessedsignalmean=pd.concat(preprocessedsignal).groupby(level=0).mean()


''' Calling the glue function to gluing Analogic and Photocounting channels - glue.py file in lidar_retrievals folder'''

glued_signal_final = []
gluing_central_final = []
    
for i in range(len(lamb)):
    
    if glueflag == 'yes':
        window_length = 150
        correlation_threshold = 0.95
        intercept_threshold = 0.5
        gaussian_threshold = 0.1
        minmax_threshold = 0.5
        min_idx = 200  # 200 * 7.5 = 1500 m ,
        max_idx = 2000  # 2000 * 7.5 = 15000 m
        
        glued_signal, gluing_central_idx, score, c_lower, c_upper = glue.glue_signals_1d(preprocessedsignalmean[str(lamb[i])+'AN'].to_numpy(), 
                                                                                preprocessedsignalmean[str(lamb[i])+'PC'].to_numpy(), 
                                                                                window_length,
                                                                                correlation_threshold, 
                                                                                intercept_threshold, 
                                                                                gaussian_threshold,
                                                                                minmax_threshold, 
                                                                                min_idx, max_idx)
        
    glued_signal_final.append(glued_signal[0])
    gluing_central_final.append(gluing_central_idx)
        
    ''' Calling the Glueing Graphics Plot from LIDAR - GGPLIDAR function - retrieval_plots.py file in lidar_retrievals folder'''
    
    retrieval_plots.ggplidar(preprocessedsignalmean[str(lamb[i])+'AN'].to_numpy(),
                              preprocessedsignalmean[str(lamb[i])+'PC'].to_numpy(),
                              glued_signal, alt['altitude'], gluing_central_idx, window_length)

dz = 7.5
AE = 1
window_size = 41
order = 3

raman_rcs = np.multiply(glued_signal_final[1],np.power(alt['altitude'],2))  

press_interpolated, temp_interpolated = lmfit.mol_parameters_raman(dfdict['station'][0], atmospheric_flag,filenameheader, preprocessedsignalmean[str(lamb[1])+channelmode],rawinsonde_folder)

alpha_aer = raman_retrievals.raman_extinction(raman_rcs.values.tolist(), dz, lamb[0], lamb[1], AE, temp_interpolated, press_interpolated,
                     window_size, order)

alpha_aer_smooth = savgol_filter(alpha_aer, 151, 3)

import matplotlib.pyplot as plt

plt.style.use('seaborn')
fig = plt.figure()
gs = fig.add_gridspec(1,1)
ax1 = fig.add_subplot(gs[0, 0])

if lamb[0] == 355:
    colorgraph = 'rebeccapurple'
elif lamb[0] == 532:
    colorgraph = 'green'
elif lamb[0] == 1020:
    colorgraph = 'crimson'  

#dateinstr = datetime.strptime(dfdict['starttime'][0], '%d/%m/%Y-%H:%M:%S').strftime('%d %b %Y %H:%M')
#dateendstr = datetime.strptime(dfdict['starttime'].iloc[-1], '%d/%m/%Y-%H:%M:%S').strftime('%H:%M')
#kfs_title = 'SPU LALINET Station - ' +  dateinstr + ' to ' + dateendstr + ' UTC' + '\n Aerosol optical profiles retrieved using LR = ' + str(lraerosol) + ' sr'
   
plt.suptitle('kfs_title', fontsize = 20, fontweight='bold')
plt.subplots_adjust(top = 0.9)

plt.plot(np.multiply(alpha_aer_smooth,optical_prop_scale), np.divide(alt['altitude'].values.tolist(),altitude_scale), color= colorgraph,linestyle='-', linewidth=1.5, label= str(lamb[1]) + ' nm ' + channelmode, zorder=1)
 
plt.xlabel("Extinction Coefficient [Mm$^{-1}$]", fontsize = 18, fontweight='bold')
plt.ylabel("Height a.g.l. [km]", fontsize = 18, fontweight='bold')
for label in ax1.get_xticklabels():
    label.set_fontweight(500)
for label in ax1.get_yticklabels():
    label.set_fontweight(500)
ax1.legend(fontsize = 18, loc = 'best', markerscale = 1.5, handletextpad = 0.2)
ax1.grid(which = 'minor', alpha = 0.5)
ax1.grid(which = 'major', alpha = 1.0)
ax1.tick_params(axis='both', which='major', labelsize=22)
ax1.tick_params(axis='both', which='minor', labelsize=22)
#ax1.axis('auto')
ax1.axis(xmin = -5e-2, xmax = 5)
ax1.axis(ymin = altitude_min, ymax = 15)
plt.legend(fontsize = 20)

#''' Calling the lmfit function to calculate atmospheric molecular extinction and backscatter using radiosounding data'''
#
#if glueflag == 'yes':
#    channelmode = 'GL'
#    dfglueing = pd.DataFrame(glued_signal[0], columns= ['glued'])
#    beta_molecular = lmfit.lidarmolfit(dfdict['station'][0], atmospheric_flag,filenameheader,dfglueing['glued'],ini_molref_alt,fin_molref_alt,lamb,channelmode,rawinsonde_folder)
#    rcs = np.multiply(dfglueing['glued'].values.tolist(),np.power(alt['altitude'],2))
#    
#else:
#    beta_molecular = lmfit.lidarmolfit(dfdict['station'][0], atmospheric_flag,filenameheader,preprocessedsignalmean[str(lamb)+channelmode],ini_molref_alt,fin_molref_alt,lamb,channelmode,rawinsonde_folder)
#    rcs = np.multiply(preprocessedsignalmean[str(lamb)+channelmode].values.tolist(),np.power(alt['altitude'],2))        
#
#'''KLETT-FERNALD-SASANO INVERSION'''
#
#reference_range = int( (fin_molref_alt/float(dfdict['vert_res'][0]) - ini_molref_alt/float(dfdict['vert_res'][0]) )/2 )
#index_reference = int(ini_molref_alt/float(dfdict['vert_res'][0]))+reference_range
#beta_aerosol_reference = 0
#bin_length = float(dfdict['vert_res'][0])
#lidar_ratio_molecular = 8.37758041
##rcs = np.multiply(preprocessedsignalmean[str(lamb)+channelmode].values.tolist(),np.power(alt['altitude'],2))
#
#aerosol_backscatter = kfs.klett_backscatter_aerosol(rcs, lraerosol, beta_molecular, index_reference, 
#                                                    reference_range, beta_aerosol_reference, bin_length, lidar_ratio_molecular)
#aerosol_backscatter_smooth = savgol_filter(aerosol_backscatter.values.tolist(), 31, 3)
#aerosol_extinction_smooth = savgol_filter(np.multiply(aerosol_backscatter.values.tolist(),lraerosol), 31, 3)
#
#retrieval_plots.kfs_plot(lamb, dfdict, lraerosol, alt['altitude'], aerosol_backscatter_smooth, aerosol_extinction_smooth, altitude_min, altitude_max, optical_prop_scale, altitude_scale, channelmode)
#
#


