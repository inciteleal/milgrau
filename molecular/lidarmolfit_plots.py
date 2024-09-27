#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIDAR MOLecular FIT Graphics - LIDARMOLFIT-plots
Module to plot molecular fit and the fit between raw lidar data and molecular scaled signal from radiosounding information
Creation started on Tue Feb 1 22:10:49 2022
@author: Fábio J. S. Lopes
"""

import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def molfit_graphs(x,y,predict_x,altitude,dfsignal,predict_smolsimulated,lamb,channelmode,atmospheric_flag,filenameheader):
    sns.set_style('darkgrid', {"grid.color": "0.6", "grid.linestyle": ":", 'axes.facecolor': 'gainsboro'})
    fig = plt.figure()
    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0, 0])
    
    if lamb == 355:
        colorgraph = 'rebeccapurple'
    elif lamb == 532:
        colorgraph = 'green'
    elif lamb == 1020:
        colorgraph = 'crimson'  
    
    plt.plot(x, y, color= 'blue',linestyle='--', linewidth=1.5, label='raw data', zorder=1)
    plt.plot(x, predict_x, color= 'red',linestyle='-', linewidth=1.5, label='fitted curve', zorder=2)
     
    plt.title('Lidar raw signal vs. Synthetic molecular signal - fit region' , fontsize = 13, fontweight='bold' )
    plt.xlabel('Synthetic molecular signal', fontsize = 13, fontweight='bold')
    plt.ylabel(r'Lidar raw signal', fontsize = 13, fontweight='bold')
    for label in ax1.get_xticklabels():
        label.set_fontweight(500)
    for label in ax1.get_yticklabels():
        label.set_fontweight(500)
    ax1.legend(fontsize = 14, loc = 'best', markerscale = 1.5, handletextpad = 0.2)
    ax1.grid(which = 'minor', alpha = 0.5)
    ax1.grid(which = 'major', alpha = 1.0)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.tick_params(axis='both', which='minor', labelsize=16)
    ax1.axis('auto')
    
    
    ax2 = fig.add_subplot(gs[0, 1])
    
    plt.plot(np.divide(altitude,1000), dfsignal.values.tolist(), color= colorgraph, linewidth=1.3, linestyle='-', label='raw lidar signal at '+ str(lamb) +' '+ channelmode, zorder=1)
    plt.plot(np.divide(altitude,1000), predict_smolsimulated, color= 'red', linewidth=1.6, linestyle='-', label='scaled molecular signal - '+atmospheric_flag , zorder=2)
    
    plt.title('Lidar raw signal vs. Synthetic molecular signal' , fontsize = 13, fontweight='bold' )
    plt.xlabel('Altitude (a.g.l.) [km]', fontsize = 15, fontweight='bold')
    plt.ylabel(r'offset corrected signal [a.u.]', fontsize = 15, fontweight='bold')
    plt.yscale('log')
    for label in ax1.get_xticklabels():
        label.set_fontweight(500)
    for label in ax1.get_yticklabels():
        label.set_fontweight(500)
    ax2.legend(fontsize = 14, loc = 'best', markerscale = 1.5, handletextpad = 0.2)
    ax2.grid(which = 'minor', alpha = 0.5)
    ax2.grid(which = 'major', alpha = 1.0)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.tick_params(axis='both', which='minor', labelsize=14)
    #ax2.axis('auto')
    ax2.axis(xmin = 0, xmax = 30)
    ax2.axis(ymin = 1e-4, ymax = 1e5)
    
    ax3 = fig.add_subplot(gs[1, :])
    
    smoothfactor=15
    ysignal = savgol_filter(np.multiply(dfsignal.values.tolist(),np.power(altitude,2)), smoothfactor, 3) #Applying Savitzky–Golay filter, a digital filter applied to data points for smoothing the graph
            
#    plt.plot(np.divide(altitude,1000), np.multiply(dfsignal.values.tolist(),np.power(altitude,2)), color= colorgraph, linewidth=1.3, linestyle='-', label='Range corrected signal at ' + str(lamb) +' '+ channelmode, zorder=1)
    plt.plot(np.divide(altitude,1000), ysignal, color= colorgraph, linewidth=1.3, linestyle='-', label='Range corrected signal at ' + str(lamb) +' '+ channelmode, zorder=1)
    plt.plot(np.divide(altitude,1000), np.multiply(predict_smolsimulated,np.power(altitude,2)), color= 'red', linewidth=1.3, linestyle='-', label='Scaled molecular fit - '+atmospheric_flag, zorder=1)
    
    dateinstr = datetime.strptime(filenameheader[0]['starttime'], '%d/%m/%Y-%H:%M:%S').strftime('%d %b %Y-%H:%M')
    dateendstr = datetime.strptime(filenameheader[-1]['stoptime'], '%d/%m/%Y-%H:%M:%S').strftime('%H:%M')
    measurement_title = 'Mean RCS at ' + str(lamb)+' nm '+ channelmode + ' - '  + dateinstr + ' to ' + dateendstr + ' UTC'
    ax3.text(22.6, 1e8, r'Savitzky–Golay smooth: '+ str(int(smoothfactor*float(filenameheader[0]['vert_res'])))+' m', fontsize=13)
    
    plt.title(measurement_title, fontsize = 12, fontweight='bold' )
    plt.xlabel('Altitude (a.g.l.) [km]', fontsize = 13, fontweight='bold')
    plt.ylabel(r'Signal Intensity [a.u.]', fontsize = 13, fontweight='bold')
    plt.yscale('log')
    for label in ax1.get_xticklabels():
        label.set_fontweight(500)
    for label in ax1.get_yticklabels():
        label.set_fontweight(500)
    ax3.legend(fontsize = 14, loc = 'best', markerscale = 1.5, handletextpad = 0.2)
    ax3.grid(which = 'minor', alpha = 0.5)
    ax3.grid(which = 'major', alpha = 1.0)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.tick_params(axis='both', which='minor', labelsize=14)
    ax3.axis(xmin = 0, xmax = 30)
    ax3.axis(ymin = 1e4, ymax = 1e10)
    plt.show()  
    
