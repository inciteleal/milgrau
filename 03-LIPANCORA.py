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
"""

import os
import numpy as np  # noqa: F401
import pandas as pd
from pathlib import Path
from datetime import datetime
from datetime import timedelta
from functions import milgrau_function as mf

rootdir_name = os.getcwd()
files_dir_stand = '02-data_raw_organized'
files_dir_level0 = '04-data_level0'
files_dir_level1 = '05-data_level1'
preprocessed_level1_dir = '02-preprocessed_corrected' 
rcsignal_dir = '03-rcsignal'
datadir_name = os.path.join(rootdir_name, files_dir_stand)
        
'''Reading all measurements directory'''

fileinfo, subfolderinfo = mf.readfiles_meastype(datadir_name)

'''Reading dark-current and atmospheric measurements files'''
binshiftcorr = [1, -1, 6, -3, 7, -2, 8, -2, 8, -2, 8, -2]
deadtime = [0, 0, 0, 0.0035, 0, 0, 0, 0.002, 0, 0, 0, 0]
bglow = 29000
bghigh = 29999

bgcheating = [0, 0, 0, 0.0020, 0, 0, 0, 0.0020, 0, 0, 0, 0]
# bgcheating = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def datetime2matlabdn(dt):
   mdn = dt + timedelta(days = 366)
   frac_seconds = (dt-datetime(dt.year,dt.month,dt.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
   frac_microseconds = dt.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
   return mdn.toordinal() + frac_seconds + frac_microseconds

for i in range(len(fileinfo)):
    rawdata=[]
    rcsignal=[]
    dtrawdata=[]
    bsrawdata=[]
    background=[]
    meandcfiles=[]
    rawdatafiles=[]
    rawdatabgcorrected=[]
    signal532AN=[]
    signal532ANarray = []
    bgarray = []
    timearray=[]
    altarray=[]
    rcsignal1=[]

    for j in range(len(subfolderinfo)):
        rawdata.append(mf.readfiles_generic(os.path.join(fileinfo[i], subfolderinfo[j])))
        if subfolderinfo[j] == 'dark_current': 
            meandcfiles = mf.rebind(rawdata[j], deadtime, rootdir_name, datadir_name, files_dir_level0, files_dir_level1)
        else:
            rawdatafiles, dictsetup, filenameaux, csv_files_path = mf.rebind(rawdata[j], deadtime, rootdir_name, datadir_name, files_dir_level0, files_dir_level1)
            
    alt = pd.DataFrame(list(range(len(rawdatafiles[i].index)))).mul(dictsetup['vert_res'][0])+dictsetup['vert_res'][0]
    alt.columns=['altitude']
    alt['rangesqrt']=alt['altitude'].pow(2)
    
    yeardir = Path(os.path.relpath(csv_files_path[0],datadir_name)).parts[-4]
    datedir = Path(os.path.relpath(csv_files_path[0],datadir_name)).parts[-3]
    filenameaux_corr = [item.replace('level0','level1_preprocessed') for item in filenameaux]
    filenameaux_rcsignal = [item.replace('level0','level1_rcsignal') for item in filenameaux]
    csv_files_dir_corrected = os.path.join(rootdir_name,files_dir_level1,yeardir,datedir,preprocessed_level1_dir)
    csv_files_dir_rcsignal = csv_files_dir_corrected.replace(preprocessed_level1_dir,rcsignal_dir)
    mf.folder_creation(csv_files_dir_corrected)
    mf.folder_creation(csv_files_dir_rcsignal)
    
    for k in range(len(rawdatafiles)):
        dtrawdata.append(rawdatafiles[k].sub(meandcfiles,axis=0)) #mean dark-current subtraction
        bsrawdata.append(mf.binshift_function(binshiftcorr, dtrawdata[k])) #binshift and zerobin correction
#        background = bsrawdata[k].loc[int(bglow/dictsetup['vert_res'][0]):int(bghigh/dictsetup['vert_res'][0])].mean(axis=0)-0.0022
        background = bsrawdata[k].loc[int(bglow/dictsetup['vert_res'][0]):int(bghigh/dictsetup['vert_res'][0])].mean(axis=0)
        background=background-bgcheating
        rawdatabgcorrected.append(bsrawdata[k].sub(background))
        
        rcsignal.append(rawdatabgcorrected[k].mul(alt['rangesqrt'],axis = 0))
        rawdatabgcorrected[k].to_csv(os.path.join(csv_files_dir_corrected,filenameaux_corr[k]), index=False, float_format="%.4f")
        rcsignal[k].to_csv(os.path.join(csv_files_dir_rcsignal,filenameaux_rcsignal[k]), index=False, float_format="%.4f")  
        
        '''writing down the header for Corrected and Pre-processed and Range Corrected files '''
        line1 = ' '.join(['station ' + dictsetup['site'] + '\n'])
        line2 = ' '.join(['altitude ' + dictsetup['altitude'] + '\n'])
        line3 = ' '.join(['lat ' + dictsetup['lat'] + '\n'])
        line4 = ' '.join(['long ' + dictsetup['long'] + '\n'])
        line5 = ' '.join(['starttime ' + dictsetup['start_time'][k] + '\n'])
        line6 = ' '.join(['stoptime ' + dictsetup['stop_time'][k] + '\n'])
        line7 = ' '.join(['bins ' + str(dictsetup['nbins'][0]) + '\n'])
        line8 = ' '.join(['vert_res ' + str(dictsetup['vert_res'][0]) + '\n'])
        line9 = ' '.join(['shotnumber ' + str(dictsetup['nshots'][0]) + '\n'])
        line10 = ' '.join(['laser_freq ' + dictsetup['laser_freq'] + '\n'])
        mf.writedown_header(os.path.join(csv_files_dir_corrected,filenameaux_corr[k]), line1, line2, line3, line4, line5, line6, line7, line8, line9, line10)
        mf.writedown_header(os.path.join(csv_files_dir_rcsignal,filenameaux_rcsignal[k]), line1, line2, line3, line4, line5, line6, line7, line8, line9, line10)

#np.savetxt("signal532.dat", signal532ANarray)
#np.savetxt("bg.dat", bgarray)
#np.savetxt("time.dat", timearray)
#np.savetxt("alt.dat", alt['altitude'])

#del i, j, k, background, bghigh, bglow, binshiftcorr, deadtime, line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, subfolderinfo
