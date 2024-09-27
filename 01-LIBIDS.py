"""
LIdar BInary Data Standardized - LIBIDS
Script to organize raw lidar binary data into standardized folder and cleaning up spurious data, such as, temp.dat, AutoSave.dpp, 
binary data with not accepted number of laser shots.
Created on Wed Dec 17 06:38:50 2020
@author: Fábio J. S. Lopes, Alexandre C. Yoshida and Alexandre Cacheffo
"""
import os
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from functions.milgrau_function import readfiles_libids

rootdir_name = os.getcwd()
files_dir_stand = '01-data'
bad_files_dir = '00-bad_files_dir'
files_dir_organized = '02-data_raw_organized'
datadir_name = os.path.join(rootdir_name, files_dir_stand)
fileinfo = readfiles_libids(datadir_name)

start_time_obj_list = []
stop_time_obj_list = []
diff_time_list = []
nshots_list = []
          
'''Reading binary data and its head'''            
for i in range(len(fileinfo[0])):
    f = open(fileinfo[0][i],"rb")
    dic = f.readline().decode('utf-8')
    aux = []
    '''14 is the line number in the header of licel binary data - here it is needed only the 1st and 2nd lines'''
    for j in range(2):
        aux.append(f.readline().decode('utf-8'))    
    
    '''Reading binary data heads (2nd line) to select number of shots, start and stop time of measurements'''
    n_shots = int(aux[1][16:21])
    start_time = aux[0][10:29]
    stop_time = aux[0][30:49]
    start_time_obj = datetime.strptime(start_time, '%d/%m/%Y %H:%M:%S')
    stop_time_obj = datetime.strptime(stop_time, '%d/%m/%Y %H:%M:%S')
    diff_time = (stop_time_obj - start_time_obj).total_seconds()
    diff_time_list.append(diff_time)
    start_time_obj_list.append(start_time_obj) 
    stop_time_obj_list.append(stop_time_obj)
    nshots_list.append(n_shots)
   
'''Setting up Dataframe with key variables to select data'''
df_head = pd.DataFrame()
df_head['filepath'] = fileinfo[0]
df_head['flag_period'] = fileinfo[1]
df_head['meas_type'] = fileinfo[2]
df_head['start_time'] = start_time_obj_list
df_head['stop_time'] = stop_time_obj_list
df_head['nshots'] = nshots_list
df_head['gap_nshots'] = diff_time_list

'''Condition to select bad files from original dataframe'''
bad_file_cond = ((df_head['nshots'] == 0) | (df_head['nshots'] < 2998) | (df_head['nshots'] > 3008) )
df_bad_files = df_head.loc[np.where(bad_file_cond)].reset_index(drop=True)

'''Removing bad files from original dataframe'''
index_cond = df_head.loc[bad_file_cond].index
df_head.drop(index_cond, inplace = True)
df_head.reset_index(drop=True, inplace=True)

'''Moving bad files from its original directory to 00-bad_files_dir folder'''
for i in range(len(df_bad_files['filepath'])):
    new_bad_files_path = os.path.join(rootdir_name,bad_files_dir,df_bad_files['start_time'][i].strftime('%Y'),''.join([df_bad_files['start_time'][i].strftime('%Y%m%d'), df_bad_files['flag_period'][i]]), df_bad_files['meas_type'][i])
    if not os.path.exists(new_bad_files_path):
        try:
            os.makedirs(new_bad_files_path)
        except OSError:
            print ('Creation of the bad files directory % s failed' % new_bad_files_path)
        else:
            print ('Successfully created the bad files directory % s' % new_bad_files_path)
    shutil.copy(df_bad_files['filepath'][i],new_bad_files_path)

'''Copy binary files from its original directory to 02-data_raw_organized folder to run with LIBIDS-SSC-2netcdf and LIPANCORA scripts'''
for i in range(len(df_head['filepath'])):
    scc_files_path = os.path.join(rootdir_name,files_dir_organized,df_head['start_time'][i].strftime('%Y'),''.join([df_head['start_time'][i].strftime('%Y%m%d'), df_head['flag_period'][i]]), df_head['meas_type'][i])
    if not os.path.exists(scc_files_path):
        try:
            os.makedirs(scc_files_path)
        except OSError:
#            print ('Creation of the raw organized file directory % s failed' % scc_files_path)
            print ('Creation of the raw organized file directories % s failed' % Path(os.path.relpath(scc_files_path,datadir_name)).parts[-2])
        else:
#            print ('Successfully created the raw organized file directory % s' % scc_files_path)
            print ('Successfully created the raw organized file directories % s' % Path(os.path.relpath(scc_files_path,datadir_name)).parts[-2])
    shutil.copy(df_head['filepath'][i],scc_files_path)        
