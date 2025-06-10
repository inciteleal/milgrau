"""
Radiosounding data acquisition - RADIODATA
@author: Alexandre Yoshida and Alexandre Cacheffo
Adapted on Mon Feb 07 09:46:18 2022 by Fabio Lopes
Adapted on 2025 by Luisa Mello
"""

import os
import urllib3
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import time
import csv
from functions import milgrau_function as mf

"""Initial setup/input"""
rootdir_name = os.getcwd()
rawinsonde_folder = "07-rawinsonde"
datadir_name = os.path.join(rootdir_name, rawinsonde_folder)
initial_date = "2024/06/06"  # time range to download data yyyy/mm/dd
final_date = "2024/06/07"
station = "83779"  # Radiosounding Station number identification
rstime = ["00", "12"]  # Radiosounding launch time ('00' and/or '12' (UTC))
time_interval = pd.date_range(initial_date, final_date, freq="D")


def solve_fast(s):
    """function to remove the first ans last line of radiosounding data"""
    ind1 = s.find("\n")
    ind2 = s.rfind("\n")
    return s[ind1 + 1 : ind2]


def save_log(log, log_file="log.log"):
    """saves log of files downloaded and files with download errors"""

    file_exists = os.path.isfile(log_file)

    with open(log_file, "a", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["date", "hour", "station", "status"])

        for data in sorted(log):
            writer.writerow(data)


def download_radiosonde_data(date, hour, station, datadir_name, max_retries=5):
    """attempts to download radiosonde data for a specific date and time, with retries in case of error."""
    year, month, day = str(date.year), str(date.month), str(date.day)
    http = urllib3.PoolManager()
    url = f"http://weather.uwyo.edu/cgi-bin/sounding?region=samer&TYPE=TEXT%3ALIST&YEAR={year}&MONTH={month}&FROM={day}{hour}&TO={day}{hour}&STNM={station}"

    for attempt in range(max_retries):
        response = http.request("GET", url)
        data_html = BeautifulSoup(response.data, "html.parser")

        if data_html.find("h2") is None:
            erro = BeautifulSoup(response.data, "lxml").get_text().split("\n")[1]
            if attempt == 0:
                print(f"Error : {erro}")
            if "try again later" in erro.lower():
                print(f"   Trying again... (attempt {attempt+1}/{max_retries})")
                time.sleep(2**attempt)
                continue
            else:
                return erro

        title = data_html.find("h2").text
        datename = datetime.strptime(
            f"{title.split(' ')[-1]}/{title.split(' ')[-2]}/{title.split(' ')[-3]}",
            "%Y/%b/%d",
        )
        filename = f"{title.split(' ')[0]}_{title.split(' ')[1]}_{datename.strftime('%Y_%m_%d')}_{title.split(' ')[-4]}.csv"
        saving_folder = os.path.join(
            datadir_name, f"{title.split(' ')[0]}_{title.split(' ')[1]}"
        )
        savingfilename = os.path.join(saving_folder, filename)

        if os.path.exists(savingfilename):
            print(f"Already downloaded: {filename}")
            return "OK- downloaded"

        mf.folder_creation(saving_folder)
        with open(savingfilename, "wt", encoding="utf-8") as csvfile:
            csvfile.write(
                BeautifulSoup(response.data, "lxml").get_text().split("\n")[3] + "\n"
            )
            for line in solve_fast(data_html.find("pre").text).splitlines()[:-1]:
                csvfile.write(line + "\n")

        print(f"Download successful: {filename}")
        return "OK- downloaded"

    return "Max retries for 'Please try again later.'"


if __name__ == "__main__":
    log = []
    for date in time_interval:
        for hour in rstime:
            log_message = download_radiosonde_data(
                date, hour, station, datadir_name, max_retries=3
            )
            log.append([date.strftime("%Y/%m/%d"), hour, station, log_message])

    save_log(log, log_file=datadir_name + "/log.txt")
