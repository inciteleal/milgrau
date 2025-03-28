"""
Radiosounding data acquisition - RADIODATA
Adapted on Mon Feb 07 09:46:18 2022
@author: Alexandre Yoshida and Alexandre Cacheffo
adapted by Fábio Lopes

"""

import os
import urllib3
import pandas as pd
from bs4 import BeautifulSoup
from datetime import  datetime
from functions import milgrau_function as mf

#function to remove the first ans last line of radiosounding data
def solve_fast(s):
    ind1 = s.find('\n')
    ind2 = s.rfind('\n')
    return s[ind1+1:ind2]

'''
############### Initial setup ###############
'''
rootdir_name = os.getcwd()
rawinsonde_folder = '07-rawinsonde'
datadir_name = os.path.join(rootdir_name, rawinsonde_folder)

initial_date = '2024/06/20' # time range to download data yyyy/mm/dd
final_date = '2024/06/20'
station = '83779' # Radiosounding Station number identification
rstime = ['00','12'] # Radiosounding launch time (00 --> 00 UTC and 12 --> 12 UTC as string). If only one launch time is desired put the time as list with one string e.g. ['00'] or ['12']

time_interval = pd.date_range(initial_date, final_date, freq = 'D')
for date in time_interval:
    year = str(date.year)
    month = str(date.month)
    day = str(date.day)
    for rs in rstime:
        hour = rs 
        http = urllib3.PoolManager()
        url = 'http://weather.uwyo.edu/cgi-bin/sounding?region=samer&TYPE=TEXT%3ALIST&YEAR=' + year \
              + '&MONTH=' + month + '&FROM=' + day + hour + '&TO=' + day + hour + '&STNM=' + station
        radiosounde_url = http.request('GET', url)
        
        radiosounde_data = BeautifulSoup(radiosounde_url.data,'html.parser')  #data acquisition in html format

        if radiosounde_data.find('h2') is None:
            print('Sorry :( --> ' + BeautifulSoup(radiosounde_url.data,'lxml').get_text().split('\n')[1])
        else:
            print('Downloading --> ' + BeautifulSoup(radiosounde_url.data,'lxml').get_text().split('\n')[3])
            n_line2 = solve_fast(radiosounde_data.find('pre').text).splitlines()[:-1] #data organization to save file as cvs
            
            # Setting file name according to date and station name and the saving folder
            title=radiosounde_data.find('h2').text
            datename = datetime.strptime(''.join([title.split(' ')[-1],'/',title.split(' ')[-2],'/',title.split(' ')[-3]]), '%Y/%b/%d')
            filename = ''.join([title.split(' ')[0],'_',title.split(' ')[1],'_',datename.strftime('%Y_%m_%d'),'_',title.split(' ')[-4],'.csv'])
            saving_folder = os.path.join(datadir_name,''.join([title.split(' ')[0],'_',title.split(' ')[1]]))
            savingfilename = os.path.join(saving_folder,filename)
            mf.folder_creation(saving_folder)
            with open(savingfilename, "wt", encoding="utf-8") as csvfile:
                csvfile.write(BeautifulSoup(radiosounde_url.data,'lxml').get_text().split('\n')[3] + '\n')
                for line in n_line2[:]:
                    csvfile.write(line + '\n')          




            
