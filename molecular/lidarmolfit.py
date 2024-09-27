#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIDAR MOLecular FIT - LIDARMOLFIT

Module to calculate atmospheric molecular backscattering and extinction parameters to lidar inversion
Creation started on Sun Jan 30 08:19:50 2022
@author: Fábio J. S. Lopes
"""

import os
import os.path
import numpy as np
import pandas as pd
from scipy import integrate
from scipy import interpolate
from datetime import datetime
from datetime import timedelta
from molecular import us_std
from molecular import lidarmolfit_plots
from molecular import rayleigh_scattering_bucholtz as rsbucholtz

    
def lidarmolfit(station, atmospheric_flag, filenameheader,preprocessedsignalmean, ini_molref_alt, fin_molref_alt,lamb,channelmode,rawinsonde_folder):
    
    if atmospheric_flag == 'radiosounding':
    
        rawinsonde_files = []
        datestr = []
                
        if filenameheader[0]['station'] == 'Sao_Paulo' :
            rawinsonde_station = '83779_SBMT'
            rawinsonde_folder0 = rawinsonde_folder
            # rawinsonde_folder01 = os.path.join(os.getcwd(),rawinsonde_folder0,rawinsonde_station)
      
        if datetime.strptime(filenameheader[0]['starttime'],'%d/%m/%Y-%H:%M:%S').hour >= 0 and datetime.strptime(filenameheader[0]['starttime'],'%d/%m/%Y-%H:%M:%S').hour <= 8:
            datestr = datetime.strftime(datetime.strptime(filenameheader[0]['starttime'],'%d/%m/%Y-%H:%M:%S'),('%Y_%m_%d_'))
            rawinsonde_files = os.path.join(os.getcwd(),rawinsonde_folder0,rawinsonde_station,''.join([rawinsonde_station,'_',datestr,'00Z.csv']))
       
        elif datetime.strptime(filenameheader[0]['starttime'],'%d/%m/%Y-%H:%M:%S').hour >= 9 and datetime.strptime(filenameheader[0]['starttime'],'%d/%m/%Y-%H:%M:%S').hour <= 20:
            datestr = datetime.strftime(datetime.strptime(filenameheader[0]['starttime'],'%d/%m/%Y-%H:%M:%S'),('%Y_%m_%d_'))
            rawinsonde_files = os.path.join(os.getcwd(),rawinsonde_folder0,rawinsonde_station,''.join([rawinsonde_station,'_',datestr,'12Z.csv']))

        elif datetime.strptime(filenameheader[0]['starttime'],'%d/%m/%Y-%H:%M:%S').hour >= 21 and datetime.strptime(filenameheader[0]['starttime'],'%d/%m/%Y-%H:%M:%S').hour <= 23:
            datestr = datetime.strftime(datetime.strptime(filenameheader[0]['starttime'],'%d/%m/%Y-%H:%M:%S')+timedelta(days=1),('%Y_%m_%d_'))
            rawinsonde_files = os.path.join(os.getcwd(),rawinsonde_folder0,rawinsonde_station,''.join([rawinsonde_station,'_',datestr,'00Z.csv']))
        
        if os.path.isfile(rawinsonde_files) == True:
            rawinsonde = pd.read_table(rawinsonde_files, skiprows = [0, 1, 3, 4, 5], skipfooter = 0, sep = '\s+', engine = 'python')
        else:
            return print('There are no radiosounding data - try to use Atmospheric Standard Model \n change atmospheric_flag to us_std')
            
        # getting data products from Radisounding - altitude (a.s.l.), pressure (hPa) and Temperature (in C to K)
        rawinsonde['press'] = rawinsonde['PRES'] # Pressão atmosférica local.
        rawinsonde['alt'] = (rawinsonde['HGHT'])  # Altitude acima do solo em m.
        rawinsonde['temp'] = rawinsonde['TEMP'] + 273.15  # Temperatura local em K.
        
    # Organizing dataframe and remove negative values
        rawinsonde = pd.DataFrame(rawinsonde, columns = ['alt', 'press', 'temp'])
        rawinsonde.where(rawinsonde < 0, np.nan)
        rawinsonde = rawinsonde.dropna()
        
        rawbetamol = []
        rawalphamol = []
        
        spustation_alt=766 # altitude station as the sea level
        altitude=[]
        for i in range(0,len(preprocessedsignalmean)):
            altitude.append(i*float(filenameheader[0]['vert_res'])+float(filenameheader[0]['vert_res']))
            
        for i in range(0,len(rawinsonde)):
            rawbetamol.append(rsbucholtz.angular_volume_scattering_coefficient(lamb, rawinsonde['press'][i], rawinsonde['temp'][i], np.pi)*1e-3) 
            rawalphamol.append(rsbucholtz.volume_scattering_coefficient(lamb, rawinsonde['press'][i], rawinsonde['temp'][i])*1e-3)
        
        rawbetamol_func = interpolate.interp1d(rawinsonde['alt'].values.tolist(),np.log(rawbetamol), kind ='linear' ,fill_value="extrapolate")
        betamol_interp = np.exp(rawbetamol_func(np.add(altitude,spustation_alt)))
        
        rawalphamol_func = interpolate.interp1d(rawinsonde['alt'].values.tolist(),np.log(rawalphamol), kind ='linear' ,fill_value="extrapolate")
        alphamol_interp = np.exp(rawalphamol_func(np.add(altitude,spustation_alt)))
                
        integralmol = []
        aodmol = []
        smolsimulated = []
           
        for j in range(0,len(preprocessedsignalmean)):
            integralmol = integrate.cumtrapz(alphamol_interp[0:j+1],altitude[0:j+1],initial=0)   
        
        for i in range(0,len(integralmol)):
            aodmol.append(np.exp(-2*integralmol[i]))
            smolsimulated.append(betamol_interp[i]*aodmol[i]/altitude[i]**(2))
        
        x = smolsimulated[int(ini_molref_alt/float(filenameheader[0]['vert_res'])):int(fin_molref_alt/float(filenameheader[0]['vert_res']))]
        y = preprocessedsignalmean.values.tolist()[int(ini_molref_alt/float(filenameheader[0]['vert_res'])):int(fin_molref_alt/float(filenameheader[0]['vert_res']))]
        model = np.polyfit(x,y,1)
        predict = np.poly1d(model)
        predict(smolsimulated)
        simulatedsignalscaled = np.multiply(smolsimulated,predict[1])
        
        lidarmolfit_plots.molfit_graphs(x,y,predict(x),altitude,preprocessedsignalmean,simulatedsignalscaled,lamb,channelmode,atmospheric_flag,filenameheader)
        
        return betamol_interp, simulatedsignalscaled
    
    elif atmospheric_flag == 'us_std':
        my_atmosphere_std = us_std.Atmosphere() #Here one can set the standard values of t = 288.15, p = 1013.25, alt = 00 to provide tha atmospheric profile
    
        altitude = []
        temp = []
        press = []
        rawbetamol = []
        rawalphamol = []
        spustation_alt=766 # altitude station as the sea level
        
        for i in range(0,len(preprocessedsignalmean)):
            altitude.append(i*float(filenameheader[0]['vert_res'])+float(filenameheader[0]['vert_res']))
            temp.append(my_atmosphere_std.temperature(altitude[i]))
            press.append(my_atmosphere_std.pressure(altitude[i]))
            rawbetamol.append(rsbucholtz.angular_volume_scattering_coefficient(lamb,press[i],temp[i], np.pi)*1e-3) 
            rawalphamol.append(rsbucholtz.volume_scattering_coefficient(lamb,press[i],temp[i])*1e-3)

        rawbetamol_func = interpolate.interp1d(altitude,np.log(rawbetamol), kind ='linear' ,fill_value="extrapolate")
        betamol_interp = np.exp(rawbetamol_func(np.add(altitude,spustation_alt)))
        
        rawalphamol_func = interpolate.interp1d(altitude,np.log(rawalphamol), kind ='linear' ,fill_value="extrapolate")
        alphamol_interp = np.exp(rawalphamol_func(np.add(altitude,spustation_alt)))
                
        integralmol = []
        aodmol = []
        smolsimulated = []
           
        for j in range(0,len(preprocessedsignalmean)):
            integralmol = integrate.cumtrapz(alphamol_interp[0:j+1],altitude[0:j+1],initial=0)   
        
        for i in range(0,len(integralmol)):
            aodmol.append(np.exp(-2*integralmol[i]))
            smolsimulated.append(betamol_interp[i]*aodmol[i]/altitude[i]**(2))
        
        x = smolsimulated[int(ini_molref_alt/float(filenameheader[0]['vert_res'])):int(fin_molref_alt/float(filenameheader[0]['vert_res']))]
        y = preprocessedsignalmean.values.tolist()[int(ini_molref_alt/float(filenameheader[0]['vert_res'])):int(fin_molref_alt/float(filenameheader[0]['vert_res']))]
        model = np.polyfit(x,y,1)
        predict = np.poly1d(model)
        predict(smolsimulated)
        simulatedsignalscaled = np.multiply(smolsimulated,predict[1])
        
        atmospheric_flag = 'U.S. Standard Atmosphere'
        lidarmolfit_plots.molfit_graphs(x,y,predict(x),altitude,preprocessedsignalmean,simulatedsignalscaled,lamb,channelmode,atmospheric_flag,filenameheader)
        
        return betamol_interp, simulatedsignalscaled
    
def mol_parameters_raman(station, atmospheric_flag, filenameheader, preprocessedsignalmean, rawinsonde_folder):
    
    if atmospheric_flag == 'radiosounding':
    
        rawinsonde_files = []
        datestr = []
                
        if filenameheader[0]['station'] == 'Sao_Paulo' :
            rawinsonde_station = '83779_SBMT'
            rawinsonde_folder0 = rawinsonde_folder
            # rawinsonde_folder01 = os.path.join(os.getcwd(),rawinsonde_folder0,rawinsonde_station)
      
        if datetime.strptime(filenameheader[0]['starttime'],'%d/%m/%Y-%H:%M:%S').hour >= 0 and datetime.strptime(filenameheader[0]['starttime'],'%d/%m/%Y-%H:%M:%S').hour <= 8:
            datestr = datetime.strftime(datetime.strptime(filenameheader[0]['starttime'],'%d/%m/%Y-%H:%M:%S'),('%Y_%m_%d_'))
            rawinsonde_files = os.path.join(os.getcwd(),rawinsonde_folder0,rawinsonde_station,''.join([rawinsonde_station,'_',datestr,'00Z.csv']))
       
        elif datetime.strptime(filenameheader[0]['starttime'],'%d/%m/%Y-%H:%M:%S').hour >= 9 and datetime.strptime(filenameheader[0]['starttime'],'%d/%m/%Y-%H:%M:%S').hour <= 20:
            datestr = datetime.strftime(datetime.strptime(filenameheader[0]['starttime'],'%d/%m/%Y-%H:%M:%S'),('%Y_%m_%d_'))
            rawinsonde_files = os.path.join(os.getcwd(),rawinsonde_folder0,rawinsonde_station,''.join([rawinsonde_station,'_',datestr,'12Z.csv']))

        elif datetime.strptime(filenameheader[0]['starttime'],'%d/%m/%Y-%H:%M:%S').hour >= 21 and datetime.strptime(filenameheader[0]['starttime'],'%d/%m/%Y-%H:%M:%S').hour <= 23:
            datestr = datetime.strftime(datetime.strptime(filenameheader[0]['starttime'],'%d/%m/%Y-%H:%M:%S')+timedelta(days=1),('%Y_%m_%d_'))
            rawinsonde_files = os.path.join(os.getcwd(),rawinsonde_folder0,rawinsonde_station,''.join([rawinsonde_station,'_',datestr,'00Z.csv']))
        
        if os.path.isfile(rawinsonde_files) == True:
            rawinsonde = pd.read_table(rawinsonde_files, skiprows = [0, 1, 3, 4, 5], skipfooter = 0, sep = '\s+', engine = 'python')
        else:
            return print('There are no radiosounding data - try to use Atmospheric Standard Model \n change atmospheric_flag to us_std')
            
        # getting data products from Radisounding - altitude (a.s.l.), pressure (hPa) and Temperature (in C to K)
        rawinsonde['press'] = rawinsonde['PRES'] # Pressão atmosférica local.
        rawinsonde['alt'] = (rawinsonde['HGHT'])  # Altitude acima do solo em m.
        rawinsonde['temp'] = rawinsonde['TEMP'] + 273.15  # Temperatura local em K.
        
    # Organizing dataframe and remove negative values
        rawinsonde = pd.DataFrame(rawinsonde, columns = ['alt', 'press', 'temp'])
        rawinsonde[(rawinsonde < 0).any(1)] = np.nan
        rawinsonde = rawinsonde.dropna()
        
        press_interpolated = []
        temp_interpolated = []
        
        spustation_alt=766 # altitude station as the sea level
        altitude=[]
        
        for i in range(0,len(preprocessedsignalmean)):
            altitude.append(i*float(filenameheader[0]['vert_res'])+float(filenameheader[0]['vert_res']))

        press_func = interpolate.interp1d(rawinsonde['alt'].values.tolist(),rawinsonde['press'].values.tolist(), kind ='linear' ,fill_value="extrapolate")
        press_interpolated = press_func(np.add(altitude,spustation_alt))
        
        temp_func = interpolate.interp1d(rawinsonde['alt'].values.tolist(),rawinsonde['temp'].values.tolist(), kind ='linear' ,fill_value="extrapolate")
        temp_interpolated = temp_func(np.add(altitude,spustation_alt))
                
        return press_interpolated, temp_interpolated
    
    elif atmospheric_flag == 'us_std':
        my_atmosphere_std = us_std.Atmosphere() #Here one can set the standard values of t = 288.15, p = 1013.25, alt = 00 to provide tha atmospheric profile
    
        altitude = []
        temp = []
        press = []
        spustation_alt=766 # altitude station as the sea level
        
        for i in range(0,len(preprocessedsignalmean)):
            altitude.append(i*float(filenameheader[0]['vert_res'])+float(filenameheader[0]['vert_res']))
            temp.append(my_atmosphere_std.temperature(altitude[i]))
            press.append(my_atmosphere_std.pressure(altitude[i]))

        press_func = interpolate.interp1d(altitude,press, kind ='linear' ,fill_value="extrapolate")
        press_interpolated = press_func(np.add(altitude,spustation_alt))
        
        temp_func = interpolate.interp1d(altitude,temp, kind ='linear' ,fill_value="extrapolate")
        temp_interpolated = temp_func(np.add(altitude,spustation_alt))
                
        return press_interpolated, temp_interpolated
    
        
