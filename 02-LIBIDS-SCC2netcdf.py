"""
LIdar BInary Data Standardized - LIBIDS-SCC2netcdf
Script to read Licel raw lidar binary data from raw organized folder and convert it to NETCDF files format to be processed by 
Single Calculus Chain algorithm from EARLINET.
This script was addapted from the Atmospheric lidar packages created by Ioannis Binietoglou 
More informations can be found in
https://gitlab.com/ioannis_binietoglou/atmospheric-lidar
https://pypi.org/project/atmospheric-lidar/
Created on Wed Dec 17 06:38:50 2020
@author: Fábio J. S. Lopes, Alexandre C. Yoshida and Alexandre Cacheffo, Marcia Marques
"""

#from atmospheric_lidar.licel import LicelLidarMeasurement
from atmospheric_lidar.licel import LicelLidarMeasurement
from atmospheric_lidar_parameters import msp_netcdf_parameters_system565
from atmospheric_lidar_parameters import msp_netcdf_parameters_system484

import os
import glob
from pathlib import Path

rootdir_name = os.getcwd()
stand_files_dir = '02-data_raw_organized'
scc_files_dir = '03-netcdf_data'

'''Reading folder with binary data'''   
for path in glob.glob(f'{os.path.join(rootdir_name,stand_files_dir)}/*/*/'):    
    meas_name = Path(os.path.relpath(path,os.sep.join([rootdir_name,stand_files_dir]))).parts[1][:8]
    meas_period = Path(os.path.relpath(path,os.sep.join([rootdir_name,stand_files_dir]))).parts[1][8:]
    save_id = (''.join([meas_name,'sa',meas_period]))
    files_meas = []
    files_meas_dc = []

    '''Set up of netcdf data parameters for day and nigth measurements - important for SCC internal configuration'''
    '''Use msp_netcdf_parameters_system565.py for daytime measurements'''
    '''Use msp_netcdf_parameters.py for nighttime measurements'''              
    if (meas_period == 'am') or (meas_period == 'pm'):
        print('Day time ' + meas_period.upper() + ' period --> Using msp_netcdf_parameters_system565')
        class mspLidarMeasurement(LicelLidarMeasurement):
            extra_netcdf_parameters = msp_netcdf_parameters_system565
    else:
        print('Nigth time period --> Using msp_netcdf_parameters_system484')
        class mspLidarMeasurement(LicelLidarMeasurement):
            extra_netcdf_parameters = msp_netcdf_parameters_system484
    
    '''Selection of measurement and dar current measurements'''
    for dir_meas in os.listdir(path):
        
        for files in os.listdir(os.path.join(path,dir_meas)):
            
            if dir_meas == 'measurements':
                files_meas.append(os.path.join(path,dir_meas,files))
            else:
                files_meas_dc.append(os.path.join(path,dir_meas,files))
           
    '''Reading file measurements'''
    my_measurement = mspLidarMeasurement(files_meas)
    
    '''Reading dark current measurements'''
    my_dark_measurement = mspLidarMeasurement(files_meas_dc)
    
    '''Link between measurement and dark measurement'''
    my_measurement.dark_measurement = my_dark_measurement
    
    '''File ID name for SCC intenrl usage. Temperature and Pressure from Lidar site at surface level'''
    my_measurement.info["Measurement_ID"] = save_id
    my_measurement.info["Temperature"] = "25"
    my_measurement.info["Pressure"] = "940"
    
    '''saving as netcdf data'''
    save_files_path = os.path.join(rootdir_name,scc_files_dir)
    if not os.path.exists(save_files_path):
        try:
            os.makedirs(save_files_path)
        except OSError:
            print ('Creation of the bad files directory % s failed' % save_files_path)
        else:
            print ('Successfully created the bad files directory % s' % save_files_path)
    my_measurement.save_as_SCC_netcdf(os.path.join(save_files_path,''.join([save_id,'.nc'])))
    print('Successfully created the SCC netcdf data % s' % save_id)
