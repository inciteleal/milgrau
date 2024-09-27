"""
Created on Tue Jul 27 09:14:05 2021

@author: Izabel Andrade
"""

import os
import pandas as pd
from functions import milgrau_function as mf

def qlhtml(dfdict, fileinfo, rootdir_name, html_dir, version, meanrcsfigurename, measperiod):
    
    year_dir = pd.to_datetime(dfdict['starttime'][0],format = '%d/%m/%Y-%H:%M:%S').strftime('%Y')
    date = pd.to_datetime(dfdict['starttime'][0],format = '%d/%m/%Y-%H:%M:%S').strftime('%Y_%m_%d') ##Escrever a data no formato YYYY_MM_DD
    datemonth = pd.to_datetime(dfdict['starttime'][0],format = '%d/%m/%Y-%H:%M:%S').strftime('%m%b') ## Inserir o mês no padrão (01jan, 02feb, 03mar,04april, 05may, 06jun, 07jul, 08aug, 09sept, 10oct, 11nov, 12dec)
    file_dir = os.path.join(rootdir_name, html_dir)  ##main directory for HTML-Quicklookfiles files (QLfiles) 
    file_name = ''.join([date,'_QL_SPULidarStation.html'])
    mf.folder_creation(file_dir)
    
    file = open(os.path.join(file_dir,file_name), 'w')
    file.write('<!Adapted from the original created by Tim Wells>\n''<!https://timnwells.medium.com/create-a-simple-responsive-image-gallery-with-html-and-css-fcb973f595ea>\n''<!doctype html>\n\n\n'
    '<html lang="en">\n'
    ' <head>\n'
    ' <meta charset="utf-8">\n\n\n'
  
    '  <title>Quicklook SPU-Lidar</title>\n'
    '  <meta name="description" content="Responsive Image Gallery">\n'
    '  <meta name="author" content="Tim Wells">\n\n\n'
          

    '  <style type="text/css">\n'
    '   html, body {\n'
    '   background: #ffffff;\n'
    '   font-family: \'PT Sans\', sans-serif;\n'
    '   font-size: 95.0%;\n'
    '}\n\n\n'
    
    '#gallery img:hover {\n'
       'filter:none;\n'
    '}\n\n'

    '#gallery {\n'
    '   line-height:0;\n'
    '   -webkit-column-count:5; /* split it into 5 columns */\n'
    '   -webkit-column-gap:5px; /* give it a 5px gap between columns */\n'
    '   -moz-column-count:5;\n'
    '   -moz-column-gap:5px;\n'
    '   column-count:3;\n'
    '   column-gap:2px;\n'
    '}\n\n\n'
    '   #gallery img {\n'
    '   width: 100% !important;\n'
    '   height: auto !important;\n'
    '   margin-bottom:2px; /* to match column gap */\n'
    '}\n\n\n'

    '@media (max-width: 2000px) {\n'
    '   #gallery {\n'
    '   -moz-column-count:    2;\n'
    '   -webkit-column-count: 2;\n'
    '   -column-count:         2;\n'
    '   }\n'
    '}\n\n\n'

    '@media (max-width: 1000px) {\n'
    '   #gallery {\n'
    '   -moz-column-count:    2;\n'
    '   -webkit-column-count: 2;\n'
    '   column-count:         2;\n'
    '   }\n'
    '}\n\n\n'

    '@media (max-width: 800px) {\n'
    '   #gallery {\n'
    '   -moz-column-count:    1;\n'
    '   -webkit-column-count: 1;\n'
    '   column-count:         1;\n'
    '   }\n'
    '}\n\n\n'

    '@media (max-width: 400px) {\n'
    '   #gallery {\n'
    '   -moz-column-count:    1;\n'
    '   -webkit-column-count: 1;\n'
    '   column-count:         1;\n'
    '    }\n'
    '}\n\n\n\n'
    '  </style> \n'
    '</head> \n'
    '<body>\n'
    '<div id="gallery">\n\n'
    
       
    '<img src='+ ' "../measurements/' + str(year_dir) + '/' + datemonth + '/' + date +'_'+ measperiod + '_05km_355nm_Sao_Paulo_QL_'+ version +'.png'+' ">\n'
    '<img src='+ ' "../measurements/' + str(year_dir) + '/' + datemonth + '/' + date +'_'+ measperiod + '_05km_532nm_Sao_Paulo_QL_'+ version +'.png'+' ">\n'
    '<img src='+ ' "../measurements/' + str(year_dir) + '/' + datemonth + '/' + date +'_'+ measperiod + '_05km_1064nm_Sao_Paulo_QL_'+ version +'.png'+' ">\n'
    '<img src='+ ' "../measurements/' + str(year_dir) + '/' + datemonth + '/' + date +'_'+ measperiod + '_22km_355nm_Sao_Paulo_QL_'+ version +'.png'+' ">\n'
    '<img src='+ ' "../measurements/' + str(year_dir) + '/' + datemonth + '/' + date +'_'+ measperiod + '_22km_532nm_Sao_Paulo_QL_'+ version +'.png'+' ">\n'
    '<img src='+ ' "../measurements/' + str(year_dir) + '/' + datemonth + '/' + date +'_'+ measperiod + '_22km_1064nm_Sao_Paulo_QL_'+ version +'.png'+' ">\n'
    '<img src='+ ' "../measurements/' + str(year_dir) + '/' + datemonth + '/' + meanrcsfigurename +' ">\n'
    '  </div>\n' 

    ' </body>\n'
    '</html>'
          )
    file.close()