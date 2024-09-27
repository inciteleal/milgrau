import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from scipy.signal import find_peaks,argrelextrema

#user input
data_folder = '/home/lidarnet-compaq/Documents/milgrau_HTHH/05-data_level1'
start_year = 2020
stop_year = 2020

bins =['1064AN',
# '1064PC',
'532AN',
# '532PC',
# '530AN',
# '530PC',
'355AN',
# '355PC',
# '387AN',
# '387PC',
# '408AN',
# '408PC',
]

bincolor={'1064AN':'crimson',
# '1064PC':'crimson',
'532AN':'green',
# '532PC':'green',
# '530AN':'mediumseagreen',
# '530PC':'mediumseagreen',
'355AN':'rebeccapurple',
# '355PC':'rebeccapurple',
# '387AN':'darkviolet',
# '387PC':'darkviolet',
# '408AN':'darkmagenta',
# '408PC':'darkmagenta',
}


# getting the data files
for yyyy in range(start_year,stop_year+1):
    for dd in sorted(os.listdir(os.path.join(data_folder, str(yyyy)))):
        colorfactor=0
        maxs=[[]]*len(bins)
        for file in sorted(os.listdir(os.path.join(data_folder,str(yyyy),str(dd),'02-preprocessed_corrected'))):
            file_path = os.path.join(data_folder,str(yyyy),str(dd),'02-preprocessed_corrected',file)
            file = open(file_path,'rb')
            aux = file.readlines()
            res = float(aux[7][8:])  # vertical resolution
            h0 = float(aux[1][8:])   # altitude
            df = pd.read_csv(file_path,skiprows=[i for i in range(0,10)]) #bin signals
            dfnp = df.to_numpy()
            h = [h0+x*res for x in range(len(df['1064AN']))] # height above sea level
            h= np.array(h)
            
            #normalization
            df2=df.copy()
            for e,bin in enumerate(bins):
                maxs[e]+=[df[bin].max()]
                
            

