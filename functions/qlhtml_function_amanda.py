"""
Created on Tue Jul 27 09:14:05 2021

@author: Izabel Andrade, Amanda Vieira dos Santos, Fabio Lopes
"""

import os
import pandas as pd
from functions import milgrau_function as mf

def qlhtml(dfdict, fileinfo, rootdir_name, html_dir, version, meanrcsfigurename, measperiod, channelmode, lamb):
    
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
        
    '* {\n'
    '  box-sizing: border-box;\n'
    '}\n\n'
    
    
    '.gallery {\n'
    '  display: flex;\n'
    '  flex-wrap: wrap;\n'
    '  align-items: flex-end;\n'
    '  flex-direction: row;\n'
    '}\n\n\n'
    '.gallery img {\n'
    '  padding: 5px;\n'
    '  flex: 1 1 50%;\n'
    '  max-width: 50%;\n'
    '}\n\n\n'
    '@media (max-width: 800px) {\n'
    '  .gallery img {\n'
    '  max-width: 100%;\n'
    '  }\n'
    '}\n\n\n'
    '</style>\n'
    '</head>\n'
    '<body>\n\n\n'
        
    '<!-- as figuras precisam estar na ordem 15 30 15 30 15 30 RCS -->\n\n\n'
    '<div class="gallery">\n\n\n'
    

    '<img src='+ ' "../measurements/' + str(year_dir) + '/' + datemonth + '/' + date +'_'+ measperiod + '_05km_' + str(lamb[0]) + 'nm' + channelmode + '_Sao_Paulo_QL_'+ version +'.png'+' ">\n'       
    '<img src='+ ' "../measurements/' + str(year_dir) + '/' + datemonth + '/' + date +'_'+ measperiod + '_15km_' + str(lamb[0]) + 'nm' + channelmode + '_Sao_Paulo_QL_'+ version +'.png'+' ">\n'
    '<img src='+ ' "../measurements/' + str(year_dir) + '/' + datemonth + '/' + date +'_'+ measperiod + '_30km_' + str(lamb[0]) + 'nm' + channelmode + '_Sao_Paulo_QL_'+ version +'.png'+' ">\n'
    '<img src='+ ' "../measurements/' + str(year_dir) + '/' + datemonth + '/' + date +'_'+ measperiod + '_05km_' + str(lamb[1]) + 'nm' + channelmode + '_Sao_Paulo_QL_'+ version +'.png'+' ">\n'
    '<img src='+ ' "../measurements/' + str(year_dir) + '/' + datemonth + '/' + date +'_'+ measperiod + '_15km_' + str(lamb[1]) + 'nm' + channelmode + '_Sao_Paulo_QL_'+ version +'.png'+' ">\n'
    '<img src='+ ' "../measurements/' + str(year_dir) + '/' + datemonth + '/' + date +'_'+ measperiod + '_30km_' + str(lamb[1]) + 'nm' + channelmode + '_Sao_Paulo_QL_'+ version +'.png'+' ">\n'
    '<img src='+ ' "../measurements/' + str(year_dir) + '/' + datemonth + '/' + date +'_'+ measperiod + '_05km_' + str(lamb[2]) + 'nm' + channelmode + '_Sao_Paulo_QL_'+ version +'.png'+' ">\n'
    '<img src='+ ' "../measurements/' + str(year_dir) + '/' + datemonth + '/' + date +'_'+ measperiod + '_15km_' + str(lamb[2]) + 'nm' + 'AN' + '_Sao_Paulo_QL_'+ version +'.png'+' ">\n'
    '<img src='+ ' "../measurements/' + str(year_dir) + '/' + datemonth + '/' + date +'_'+ measperiod + '_30km_' + str(lamb[2]) + 'nm' + 'AN' + '_Sao_Paulo_QL_'+ version +'.png'+' ">\n'
    '<img src='+ ' "../measurements/' + str(year_dir) + '/' + datemonth + '/' + meanrcsfigurename +' ">\n'
    '  </div>\n' 

    ' </body>\n'
    '</html>'
          )
    file.close()
