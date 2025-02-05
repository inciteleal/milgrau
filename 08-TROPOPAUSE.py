import os
from datetime import date
 
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

years = mdates.YearLocator()  # every year
months = mdates.MonthLocator()  # every month
days = mdates.DayLocator()
hours = mdates.HourLocator()
minutes = mdates.MinuteLocator()
seconds = mdates.SecondLocator()
yearsFmt = mdates.DateFormatter("%b")

sns.set_theme(font="Anonymous Pro")

sns.set_theme()

# User Input
initial_date = "2013/01/01"  # Time range data yyyy/mm/dd
final_date = "2023/12/31"
station = "83779_SBMT"  # Radiosounding Station number and identifier
rstime = [
    "00",
    "12",
]  # Radiosounding launch time (00 --> 00 UTC and 12 --> 12 UTC as string). If only one
# launch time is desired put the time as list with one string e.g. ['00'] or ['12']
profile = False  # True --> plots and saves day height profile, False --> doesn't plot
# day height profile
rawinsonde_folder = "07-rawinsonde"  # Folder containing radiosounding data
rootdir_name = os.getcwd()
datadir_name = os.path.join(rootdir_name, rawinsonde_folder)
color_CPT = [
    "#d9ed92",
    "#b5e48c",
    "#99d98c",
    "#76c893",
    "#52b69a",
    "#34a0a4",
    "#168aad",
    "#1a759f",
    "#1e6091",
    "#184e77",
    "#0E2C44",
]
color_LRT = [
    "#F8BD4F",
    "#F6AA1C",
    "#D97212",
    "#BC3908",
    "#A82A0A",
    "#941B0C",
    "#7B190A",
    "#621708",
    "#421005",
    "#320D03",
    "#220901",
]


def get_radiodata(file):

    try:
        df = pd.read_table(
            datadir_name + "/" + station + "/" + file,
            delim_whitespace=True,
            header=2,
            skiprows=4,
            engine="python",
        )
        a = np.array(df)
        a = np.transpose(a)
        a = a[1:3]

        heights, temperatures = a
        heights = heights / 1000  # convert to km
        temperatures += 273.15  # convert to K

        dt = np.diff(temperatures) / np.diff(heights)  # temperature gradient

        return heights, temperatures, dt

    except:  # noqa: E722
        print("No radiosound data for", file)
        return False


def CPT(h, t, dt):
    """''The cold-point tropopause (CPT) is the height where the minimum temperature is 
    found below 20 km. """ ""

    CPT = h[np.where(t == np.min(t))[0][0]]
    if CPT <= 20 and CPT >= 13:
        temp = round(t[np.where(t == np.min(t))[0][0]], 2)
        return CPT, temp
    else:
        return np.nan, np.nan


def LRT(h, t, dt):
    """The lapse rate tropopause (LRT) is the lowest height at which the temperature
    gradient is greater than or equal to -2 K/km, provided that the averaged temperature
    gradient between this level and all the higher levels within 2 km does not exceed
    -2 K/km."""

    LRT, temp = np.nan, np.nan
    hs = np.where(dt >= -2)[0]  ##height indexes where temp gradient is >= -2K/km

    for i in hs:
        if h[i] >= 10:
            initial = np.where(h > h[i])
            final = np.where(h <= h[i] + 2)
            ind = np.intersect1d(
                initial, final
            )  ##height indexes for 2km above initial height
            try:
                dtmed = np.mean(dt[ind[0] : ind[-1]])  ##mean temp gradient in these 2km
            except:  # noqa: E722
                break
            if dtmed >= -2:
                LRT = h[i]
                temp = round(t[i], 2)
                break
    return LRT, temp


def plot_profile(h, t, dt, day, data):
    fig = plt.figure(figsize=(5, 7))

    ax = fig.add_subplot(111, label="1")
    ax2 = ax.twiny()

    ax.plot(t, h, marker=".", linestyle="--", color="k")

    # para plotar a tropopausa calculada do dia
    ax2.axhline(y=data[1], color="g", linestyle="-", label="Rawinsonde CPT")
    ax2.axhline(y=data[3], color="purple", linestyle="-", label="Rawinsonde LRT")

    ax.set_xlabel("Temperature(K)", color="k")
    ax.tick_params(axis="x", colors="k")
    ax2.legend(loc="lower right", fancybox=True, shadow=True)

    ax2.plot(dt, h[1:], marker=".", linestyle=":", color="r")
    ax2.xaxis.tick_top()
    ax2.set_xlabel("dT/dz(K/km)", color="r")
    ax2.xaxis.set_label_position("top")
    ax2.tick_params(axis="x", colors="r")

    ax.set_ylabel("Height (km)")
    plt.title("Height Profile of Temperature " + str(day))

    plt.savefig("Height_prof_" + str(day), dpi=300, bbox_inches="tight")


def plot_tropopause(dados, title):

    dados["day"] = pd.to_datetime(dados["day"], format="ISO8601")
    weekly = dados.groupby(pd.Grouper(key="day", freq="W")).mean().reset_index()

    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(111, label="1")

    # data CPT
    valid_data = weekly["CPT(km)"].notna() & ~np.isinf(weekly["CPT(km)"])
    ax.plot(
        weekly["day"][valid_data],
        weekly["CPT(km)"][valid_data],
        marker=".",
        label="Radiosounding CPT",
        c="darkcyan",
        lw=0.5,
    )

    # dados LRT
    valid_data = weekly["LRT(km)"].notna() & ~np.isinf(weekly["LRT(km)"])
    ax.plot(
        weekly["day"][valid_data],
        weekly["LRT(km)"][valid_data],
        marker=".",
        label="Radiosounding LRT",
        c="darkmagenta",
        lw=0.5,
    )

    ax.set_xlabel("Date")
    plt.legend(loc="upper right", fancybox=True, shadow=True, ncols=2)

    # ax.tick_params(axis='x')
    # ax.xaxis.set_major_formatter(yearsFmt)
    # legend = plt.legend(bbox_to_anchor=(1, 1),loc='upper left', fancybox=True, shadow=True)

    plt.ylabel("Height (km)")
    plt.title("Tropopause over São Paulo")
    plt.tight_layout()
    plt.savefig("Height_tropop", dpi=300, transparent=False)
    plt.show()


def plot_tropo_temp(dados, title):
    dados["day"] = pd.to_datetime(dados["day"], format="ISO8601")
    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_subplot(111, label="1")

    ax.scatter(dados["day"], dados["CPT_temp(K)"], s=1, c=color_CPT[-5], label="CPT")
    ax.scatter(dados["day"], dados["LRT_temp(K)"], s=1, c=color_LRT[-5], label="LRT")

    ax.set_ylim(190, 215)

    plt.legend(loc="upper right", fancybox=True, shadow=True)
    ax.set_xlabel("Year")
    plt.ylabel("Temperature (K)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig("temp_tropop", dpi=300, bbox_inches="tight", transparent=False)
    plt.show()


files=[]
days=[]
time_interval = pd.date_range(initial_date, final_date, freq = 'D')
for date in time_interval:
    year = str(date.year)
    month = str(date.month)
    if len(month)==1:
        month='0'+month
    dayd = str(date.day)
    if len(dayd)==1:
        dayd='0'+dayd
    for rs in rstime:
        hour = rs
        file_name=station+'_'+year+'_'+month+'_'+dayd+'_'+hour+'Z.csv'
        files.append(file_name)
        days.append(date.date())


## Writes CSV with data
header = ['day', 'CPT(km)', 'CPT_temp(K)', 'LRT(km)', 'LRT_temp(K)']
with open('Tropopause.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for i,file in enumerate(files):
        dadoss = get_radiodata(file)
        if dadoss!=False and len(dadoss[0])>0:
            h,t,dt=dadoss
            data = [days[i],CPT(h,t,dt)[0], CPT(h,t,dt)[1], LRT(h,t,dt)[0], LRT(h,t,dt)[1]]
            writer.writerow(data)
            if profile:
                plot_profile(h,t,dt, days[i],data)

## Plots
# dados = pd.read_csv(rootdir_name + "/Tropopause.csv")
# plot_tropopause(dados, "Height of the tropopause São Paulo")
# plot_tropo_temp(dados,'Temperature of the tropopause over São Paulo')
