#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KFS Graphics
Module to plot Backscatter and extinction profile from KFS inversion
Created on Fri Feb 18 16:01:47 2022
@author: Fábio J. S. Lopes
"""

import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def kfs_plot(lamb, dfdict, lraerosol, alt, aerosol_backscatter_smooth, aerosol_extinction_smooth, altitude_min, altitude_max, optical_prop_scale, altitude_scale, channelmode):
    # plt.style.use('seaborn')
    sns.set_theme(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
    sns.set_style('darkgrid', {"grid.color": "0.6", "grid.linestyle": ":", 'axes.facecolor': 'gainsboro'})
    fig = plt.figure()
    gs = fig.add_gridspec(1,2)
    ax1 = fig.add_subplot(gs[0, 0])
    
    if lamb == 355:
        colorgraph = 'rebeccapurple'
    elif lamb == 532:
        colorgraph = 'green'
    elif lamb == 1064:
        colorgraph = 'crimson'  
    
    dateinstr = datetime.strptime(dfdict['starttime'][0], '%d/%m/%Y-%H:%M:%S').strftime('%d %b %Y %H:%M')
    dateendstr = datetime.strptime(dfdict['starttime'].iloc[-1], '%d/%m/%Y-%H:%M:%S').strftime('%H:%M')
    kfs_title = 'SPU LALINET Station - ' +  dateinstr + ' to ' + dateendstr + ' UTC' + '\n Aerosol optical profiles retrieved using LR = ' + str(lraerosol) + ' sr'
       
    plt.suptitle(kfs_title, fontsize = 20, fontweight='bold')
    plt.subplots_adjust(top = 0.9)
    
    plt.plot(np.multiply(aerosol_backscatter_smooth,optical_prop_scale), np.divide(alt.values.tolist(),altitude_scale), color= colorgraph,linestyle='-', linewidth=1.5, label= str(lamb) + ' nm ' + channelmode, zorder=1)
     
    plt.xlabel("Backscatter Coefficient [Mm$^{-1}$sr$^{-1}$]", fontsize = 18, fontweight='bold')
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
    ax1.axis(ymin = altitude_min, ymax = altitude_max)
    plt.legend(fontsize = 20)
    
    ax2 = fig.add_subplot(gs[0, 1])
    
    plt.plot(np.multiply(aerosol_extinction_smooth,optical_prop_scale), np.divide(alt.values.tolist(),altitude_scale), color= colorgraph,linestyle='-', linewidth=1.5, label= str(lamb) + ' nm ' + channelmode, zorder=1)
     
    plt.xlabel("Extinction Coefficient [Mm$^{-1}$]", fontsize = 18, fontweight='bold')
    plt.ylabel("Height a.g.l. [km]", fontsize = 18, fontweight='bold')
    for label in ax2.get_xticklabels():
        label.set_fontweight(500)
    for label in ax2.get_yticklabels():
        label.set_fontweight(500)
    ax2.legend(fontsize = 18, loc = 'best', markerscale = 1.5, handletextpad = 0.2)
    ax2.grid(which = 'minor', alpha = 0.5)
    ax2.grid(which = 'major', alpha = 1.0)
    ax2.tick_params(axis='both', which='major', labelsize=22)
    ax2.tick_params(axis='both', which='minor', labelsize=22)
    #ax2.axis('auto')
    ax2.axis(xmin = -5e-1, xmax = 100)
    ax2.axis(ymin = altitude_min, ymax = altitude_max)
    
    plt.show() 


def sr_plot(lamb, dfdict, alt, altitude_scale, channelmode, lidar_signal, simulated_signal, altitude_min_01, altitude_max_01, altitude_min_02, altitude_max_02, base_altitude, top_altitude):
    sns.set_theme(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
    sns.set_style('darkgrid', {"grid.color": "0.6", "grid.linestyle": ":", 'axes.facecolor': 'gainsboro'})
    fig = plt.figure()
    gs = fig.add_gridspec(1,2)
    ax1 = fig.add_subplot(gs[0, 0])
    
    if lamb == 355:
        colorgraph = 'rebeccapurple'
    elif lamb == 532:
        colorgraph = 'green'
    elif lamb == 1020:
        colorgraph = 'crimson'  
    
    dateinstr = datetime.strptime(dfdict['starttime'][0], '%d/%m/%Y-%H:%M:%S').strftime('%d %b %Y %H:%M')
    dateendstr = datetime.strptime(dfdict['starttime'].iloc[-1], '%d/%m/%Y-%H:%M:%S').strftime('%H:%M')
    kfs_title = 'SPU LALINET Station - ' +  dateinstr + ' to ' + dateendstr + ' UTC' + '\n Aerosol optical profiles retrieved' 

    plt.suptitle(kfs_title, fontsize = 20, fontweight='bold')
    plt.subplots_adjust(top = 0.9)
    
    scattering_ratio = np.divide(lidar_signal,simulated_signal)
    mean_SR =np.mean(scattering_ratio[int(altitude_scale*base_altitude/float(dfdict['vert_res'][0])):int(altitude_scale*top_altitude/float(dfdict['vert_res'][0]))])
    
    smoothfactor=25
    SR = savgol_filter(scattering_ratio.values.tolist(), smoothfactor, 3) #Applying Savitzky–Golay filter, a digital filter applied to data points for smoothing the graph
    
    plt.plot(SR, np.divide(alt.values.tolist(),altitude_scale), color= colorgraph,linestyle='-', linewidth=1.5, label= str(lamb) + ' nm ' + channelmode, zorder=1)
    
    plt.xlabel("Scattering ratio", fontsize = 18, fontweight='bold')
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
    ax1.axis(xmin = 0, xmax = 6)
    ax1.axis(ymin = 0, ymax = 30)
    ax1.vlines(1,ymin=0,ymax=30, colors='black', linestyles='dashed')
    plt.legend(fontsize = 20)
        

    ax3 = fig.add_subplot(gs[0, 1])
    
    plt.plot(SR, np.divide(alt.values.tolist(),altitude_scale), color= colorgraph,linestyle='-', linewidth=1.5, label= str(lamb) + ' nm ' + channelmode, zorder=1)
     
    plt.xlabel("Scattering ratio", fontsize = 18, fontweight='bold')
    plt.ylabel("Height a.g.l. [km]", fontsize = 18, fontweight='bold')
    for label in ax1.get_xticklabels():
        label.set_fontweight(500)
    for label in ax1.get_yticklabels():
        label.set_fontweight(500)
    ax3.legend(fontsize = 18, loc = 'best', markerscale = 1.5, handletextpad = 0.2)
    ax3.grid(which = 'minor', alpha = 0.5)
    ax3.grid(which = 'major', alpha = 1.0)
    ax3.tick_params(axis='both', which='major', labelsize=22)
    ax3.tick_params(axis='both', which='minor', labelsize=22)
    #ax2.axis('auto')
    ax3.vlines(1,ymin=0,ymax=30, colors='black', linestyles='dashed')
    ax3.hlines(base_altitude, xmin=0,xmax=6, colors='crimson', linestyles='dashed')
    ax3.hlines(top_altitude, xmin=0,xmax=6, colors='crimson', linestyles='dashed')
    ax3.axis(xmin = 0, xmax = 4)
    ax3.axis(ymin = altitude_min_02, ymax = altitude_max_02)
    ax3.text(1.2, 14, r'Mean SR = '+ str(round(mean_SR, 2)), fontsize=13)
    ax3.text(1.2, 12.6, r'Savitzky–Golay smooth: '+ str(int(smoothfactor*float(dfdict['vert_res'][0])))+' m', fontsize=13)

    plt.legend(fontsize = 20)
    
    
    plt.show() 

"""
Glueing Graphics Plot from LIDAR - GGPLIDAR
Module to plot glueing analogic and photocounting signal from lidar retrieval`
Created on Thu Mar  3 16:08:34 2022
@author: Fábio J. S. Lopes
"""

def ggplidar(ppsignal_an,ppsignal_pc, glued_signal, altitude,  gluing_central_idx, window_length):
    
    idx_max = len(ppsignal_an)
    
    scale_factor = np.sum(ppsignal_pc[1000:1500]) / np.sum(ppsignal_an[1000:1500])
    
    rcsan = np.multiply((ppsignal_an[:idx_max] * scale_factor),np.power(altitude[:idx_max],2))
    rcspc = np.multiply(ppsignal_pc[:idx_max],np.power(altitude[:idx_max],2))
    rcsglued = np.multiply(glued_signal[0][:idx_max],np.power(altitude[:idx_max],2))
    
    
    # plt.style.use('seaborn')
    sns.set_theme(context='notebook', style='whitegrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
    sns.set_style('darkgrid', {"grid.color": "0.6", "grid.linestyle": ":", 'axes.facecolor': 'gainsboro'})
    fig, ax = plt.subplots()
    
    plt.plot(altitude[:idx_max]/1000, rcsan, label='analog scaled', color = 'darkcyan')
    plt.plot(altitude[:idx_max]/1000, rcspc, label='photo-counting',  color = 'orangered')
    plt.plot(altitude[:idx_max]/1000, rcsglued, label='glued signal',  color = 'darkmagenta')
    ylims = plt.ylim()
    plt.vlines([altitude[int(gluing_central_idx-window_length/2)]/1000,altitude[int(gluing_central_idx+window_length/2)]/1000],ylims[0]-5e10,ylims[1]+5e10, colors='k', linestyles='dashed',label='glueing region')
    plt.ylim(ylims)
   
    plt.xlabel("Glued signal - AN + PC", fontsize = 22, fontweight='bold')
    plt.ylabel("Height a.g.l. [km]", fontsize = 22, fontweight='bold')
    for label in ax.get_xticklabels():
        label.set_fontweight(500)
    for label in ax.get_yticklabels():
        label.set_fontweight(500)
    ax.legend(fontsize = 18, loc = 'best', markerscale = 1.5, handletextpad = 0.2)
    # ax.grid(which = 'minor', alpha = 0)
    # ax.grid(which = 'major', alpha = 0)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.tick_params(axis='both', which='minor', labelsize=22)
    # #ax1.axis('auto')
    plt.ylim(-1e8, 4e8)
    plt.xlim(0, 20)
        
    plt.show() 
