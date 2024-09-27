"""
Functions to be used on LIdar RAnge COrrected Signal - LIRACOS
Created on Wed Jul 21 06:43:53 2021
@author: Fábio J. S. Lopes

"""

import locale
import os
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MultipleLocator

from .python_colormap import labview_colormap

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")


def meanrcs(
    rcsignalmean,
    alt,
    lamb,
    channelmode,
    dfdict,
    maxscale_alt,
    fileinfo,
    rootdir_name,
    version,
    rcsmeangraphics_dir,
):

    mdpi = 100

    dateinstr = datetime.strptime(dfdict["starttime"][0], "%d/%m/%Y-%H:%M:%S").strftime(
        "%d %b %Y-%H:%M"
    )
    dateendstr = datetime.strptime(
        dfdict["starttime"].iloc[-1], "%d/%m/%Y-%H:%M:%S"
    ).strftime("%d %b %Y-%H:%M")
    measurement_title = (
        "Mean RCS - "
        + dateinstr
        + " to "
        + dateendstr
        + " UTC \n SPU Lidar Station - São Paulo"
    )

    measperiod = Path(os.path.relpath(fileinfo, rootdir_name)).parts[-1][-2:]
    dateinname = datetime.strptime(
        dfdict["starttime"][0], "%d/%m/%Y-%H:%M:%S"
    ).strftime("%Y%m%d%H%M")
    dateendname = datetime.strptime(
        dfdict["starttime"].iloc[-1], "%d/%m/%Y-%H:%M:%S"
    ).strftime("%Y%m%d%H%M")
    meanrcsfigurename = "".join(
        [
            dateinname,
            "_",
            dateendname,
            "_",
            measperiod,
            "_",
            str(int(maxscale_alt / 1000)).zfill(2),
            "km_",
            dfdict["station"][0],
            "_meanrcsfigure_",
            channelmode,
            "_",
            version,
            ".png",
        ]
    )

    levelinfo = Path(os.path.relpath(fileinfo, rootdir_name)).parts[0][-1]
    # im_license = mpl.image.imread(os.path.join(rootdir_name, 'img', 'by-sa.png'))
    logoleal = mpl.image.imread(os.path.join(rootdir_name, "img", "logo_leal.png"))
    logoincite = mpl.image.imread(
        os.path.join(rootdir_name, "img", "Logo_InCite_blue_site.png")
    )

    sns.set(style="darkgrid")
    fig, ax = plt.subplots(
        1, 1, sharey="row", figsize=(800 / mdpi, 960 / mdpi), dpi=mdpi
    )
    fig.suptitle(measurement_title, fontsize=14, fontweight="bold")
    fig.subplots_adjust(top=0.91)
    # fig.subplot(111)

    (p1,) = ax.plot(
        rcsignalmean[str(lamb[0]) + channelmode]
        .rolling(50, min_periods=1)
        .mean()
        .to_numpy(),
        alt["altitude"].to_numpy(),
        color="rebeccapurple",
        linestyle="-",
        label="355 nm " + channelmode,
        zorder=2,
    )
    (p2,) = ax.plot(
        rcsignalmean[str(lamb[1]) + channelmode]
        .rolling(50, min_periods=1)
        .mean()
        .to_numpy(),
        alt["altitude"].to_numpy(),
        color="forestgreen",
        linestyle="-",
        label="532 nm " + channelmode,
        zorder=3,
    )
    #    p2, = ax.plot(rcsignalmean[str(lamb[1])+'AN'].to_numpy(), alt['altitude'].to_numpy(), color='forestgreen',
    #                  linestyle='-', label='532 nm', zorder=3)
    (p3,) = ax.plot(
        rcsignalmean[str(lamb[2]) + "AN"].rolling(120, min_periods=1).mean().to_numpy(),
        alt["altitude"].to_numpy(),
        color="crimson",
        linestyle="-",
        label="1064 nm AN",
        zorder=1,
    )

    ax.set_xlabel(r"mean RCS [a.u.]", fontsize=15, fontweight="bold")
    ax.set_xscale("log")
    ax.set_ylabel("Height (m a.g.l.)", fontsize=15, fontweight="bold")
    #    ax.tick_params(axis='both', which='major', labelsize=12, width=1, length=5, color='black', direction='in')
    for label in ax.get_xticklabels():
        label.set_fontweight(500)
    for label in ax.get_yticklabels():
        label.set_fontweight(500)
    ax.legend(fontsize=15, loc="best", markerscale=1.5, handletextpad=0.2)
    #    ax.grid(which = 'minor', alpha = 0.5)
    #    ax.grid(which = 'major', alpha = 1.0)
    ax.axis(xmin=1e2, xmax=1e9)
    ax.axis(ymin=0, ymax=maxscale_alt)

    # newax_license = fig.add_axes([0.64, 0.005, 0.14, 0.03], zorder=11)
    # newax_license.imshow(im_license, alpha=0.8, aspect='equal')
    # newax_license.axis('off')

    newax_incite = fig.add_axes([0.0, 0.03, 0.14, 0.06], zorder=12)
    newax_incite.imshow(logoincite, alpha=0.9, aspect="equal")
    newax_incite.axis("off")

    newax_logo = fig.add_axes([0.87, 0.006, 0.12, 0.08], zorder=11)
    newax_logo.imshow(logoleal, alpha=1, aspect="equal")
    newax_logo.axis("off")

    # fig.text(0.77, 0.005, u"\u00A9 {1} {0}.\nCC BY SA 4.0 License.".format(datetime.now().strftime('%Y'), 'LEAL-IPEN-LALINET'),
    #     fontweight='bold', fontsize=8, color='black', ha='left',
    #     va='bottom', alpha=1, zorder=10)

    fig.text(
        0.63,
        0.005,
        "LEAL-IPEN-LALINET",
        fontweight="bold",
        fontsize=12,
        color="black",
        ha="left",
        va="bottom",
        alpha=1,
        zorder=10,
    )

    fig.text(
        0.02,
        0.015,
        datetime.strptime(dfdict["starttime"][0], "%d/%m/%Y-%H:%M:%S").strftime(
            "%d %b %Y"
        ),
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.17,
        0.015,
        "".join(["Level: ", str(float(levelinfo))]),
        fontsize=12,
        fontweight="bold",
    )

    plt.savefig(os.path.join(rcsmeangraphics_dir, meanrcsfigurename))
    #    plt.show()
    plt.close(fig)

    return meanrcsfigurename, measperiod


def ql(
    new_rcslambda,
    alt,
    rcstime,
    lamb,
    qlchannelmode,
    dfdict,
    maxscale_altql,
    fileinfo,
    rootdir_name,
    version,
    quicklook_graphics_dir,
):

    #    if lamb == 355:
    #        colorfactor = 9e6
    #    elif lamb == 532:
    #        colorfactor = 9e6
    #    elif lamb == 1064:
    #        colorfactor = 5e6
    if qlchannelmode == "AN":
        if lamb == 355:
            colorfactor = 5e6
        elif lamb == 532:
            colorfactor = 7e6
        elif lamb == 1064:
            colorfactor = 4e7

    ##    elif qlchannelmode == 'PC':
    ##        if lamb == 355:
    ##            colorfactor = 2e8
    ##        elif lamb == 532:
    ##            colorfactor = 4e8

    ###daytime color configuration for PC channels
    elif qlchannelmode == "PC":
        if lamb == 355:
            colorfactor = 2e8
        elif lamb == 532:
            colorfactor = 3e8

    if maxscale_altql <= 5250:
        (ymajorfactor, yminorfactor) = (100, 50)
    elif maxscale_altql > 5250 and maxscale_altql <= 18000:
        (ymajorfactor, yminorfactor) = (400, 100)
    elif maxscale_altql > 18000:
        (ymajorfactor, yminorfactor) = (500, 250)

    timediff = datetime.strptime(
        dfdict["starttime"].iloc[-1], "%d/%m/%Y-%H:%M:%S"
    ) - datetime.strptime(dfdict["starttime"].iloc[0], "%d/%m/%Y-%H:%M:%S")

    if int(timediff.total_seconds() / 60) <= 120:
        (xmajorfactor, xminorfactor) = (10, 5)
    elif (
        int(timediff.total_seconds() / 60) > 120
        and int(timediff.total_seconds() / 60) <= 360
    ):
        (xmajorfactor, xminorfactor) = (30, 10)
    elif int(timediff.total_seconds() / 60) > 360:
        (xmajorfactor, xminorfactor) = (60, 30)

    dateinstr = datetime.strptime(dfdict["starttime"][0], "%d/%m/%Y-%H:%M:%S").strftime(
        "%d %b %Y-%H:%M"
    )
    dateendstr = datetime.strptime(
        dfdict["starttime"].iloc[-1], "%d/%m/%Y-%H:%M:%S"
    ).strftime("%d %b %Y-%H:%M")
    measurement_title = (
        "RCS at "
        + str(lamb)
        + " nm "
        + qlchannelmode
        + " - "
        + dateinstr
        + " to "
        + dateendstr
        + " \n SPU Lidar Station - São Paulo"
    )
    levelinfo = Path(os.path.relpath(fileinfo, rootdir_name)).parts[0][-1]
    # im_license = mpl.image.imread(os.path.join(rootdir_name, 'img', 'by-sa.png'))
    logoleal = mpl.image.imread(os.path.join(rootdir_name, "img", "logo_leal.png"))
    logoincite = mpl.image.imread(
        os.path.join(rootdir_name, "img", "Logo_InCite_blue_site.png")
    )

    measperiod = Path(os.path.relpath(fileinfo, rootdir_name)).parts[-1][-2:]
    dateinname = datetime.strptime(
        dfdict["starttime"][0], "%d/%m/%Y-%H:%M:%S"
    ).strftime("%Y_%m_%d_")
    quicklook_figname = "".join(
        [
            dateinname,
            measperiod,
            "_",
            str(int(maxscale_altql / 1000)).zfill(2),
            "km_",
            str(lamb),
            "nm",
            qlchannelmode,
            "_",
            dfdict["station"][0],
            "_QL_",
            version,
            ".png",
        ]
    )

    fig = plt.figure(figsize=[14, 7])
    ax = fig.add_axes([0.11, 0.15, 0.78, 0.74])
    pcmesh = ax.pcolormesh(
        list(new_rcslambda.columns.values.astype(str)),
        list(new_rcslambda.index.values.astype(str)),
        new_rcslambda,
        vmin=0e0,
        vmax=colorfactor,
        cmap=labview_colormap(),
        rasterized=True,
        shading="nearest",
    )
    ax.set_xlabel("Time UTC", fontsize=20, fontweight="bold")
    ax.set_ylabel("Height (m a.g.l.)", fontsize=20, fontweight="bold")

    ax.yaxis.set_major_locator(MultipleLocator(ymajorfactor))
    ax.yaxis.set_minor_locator(MultipleLocator(yminorfactor))
    ax.xaxis.set_major_locator(MultipleLocator(xmajorfactor))
    ax.xaxis.set_minor_locator(MultipleLocator(xminorfactor))
    ax.set_ylim([1, np.ceil(maxscale_altql / float(dfdict["vert_res"][0]))])
    #    ax.set_ylim([np.ceil(15000/float(dfdict['vert_res'][0])), np.ceil(maxscale_altql/float(dfdict['vert_res'][0]))])
    # ax.set_xlim(xmin=0, xmax=new_rcslambda.shape[1] - 1)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=16,
        bottom=True,
        left=True,
        width=2,
        length=5,
    )
    ax.tick_params(
        axis="both", which="minor", width=1.5, length=3.5, bottom=True, left=True
    )
    for label in ax.get_xticklabels():
        label.set_fontweight(550)
    for label in ax.get_yticklabels():
        label.set_fontweight(550)
    ax.set_title(measurement_title, fontsize=18, fontweight="bold")

    cb_ax = fig.add_axes([0.893, 0.20, 0.02, 0.65])
    cbar = fig.colorbar(
        pcmesh,
        cax=cb_ax,
        ticks=np.linspace(0e0, colorfactor, 5),
        orientation="vertical",
        format="%.0e",
    )
    cbar.ax.get_yaxis().labelpad = 14
    cbar.ax.yaxis.set_offset_position("left")
    cbar.ax.set_ylabel(ylabel="Intensity [a.u.]", fontsize=16, fontweight="bold")
    cbar.ax.tick_params(
        axis="y",
        direction="out",
        labelsize=14,
        pad=5,
    )
    for cbarlabel in cbar.ax.get_yticklabels():
        cbarlabel.set_fontweight(400)

    # newax_license = fig.add_axes([0.71, 0.006, 0.14, 0.06], zorder=10)
    # newax_license.imshow(im_license, alpha=0.8, aspect='equal')
    # newax_license.axis('off')

    newax_logo = fig.add_axes([0.87, 0.006, 0.20, 0.08], zorder=12)
    newax_logo.imshow(logoleal, alpha=1, aspect="equal")
    newax_logo.axis("off")

    newax_incite = fig.add_axes([0.84, 0.006, 0.14, 0.07], zorder=12)
    newax_incite.imshow(logoincite, alpha=0.9, aspect="equal")
    newax_incite.axis("off")

    # fig.text(0.84, 0.003, u"\u00A9 {1} {0}.\nCC BY SA 4.0 License.".format(datetime.now().strftime('%Y'), 'LEAL-IPEN-LALINET'),
    #     fontweight='bold', fontsize=11, color='black', ha='left',
    #     va='bottom', alpha=1, zorder=10)

    # fig.text(0.73, 0.003, u"{1} {0}".format(datetime.now().strftime('%Y'), 'LEAL-IPEN-LALINET'),
    #     fontweight='bold', fontsize=11, color='black', ha='left',
    #     va='bottom', alpha=1, zorder=10)

    fig.text(
        0.76,
        0.003,
        "LEAL-IPEN-LALINET",
        fontweight="bold",
        fontsize=12,
        color="black",
        ha="left",
        va="bottom",
        alpha=1,
        zorder=10,
    )

    fig.text(
        0.05,
        0.015,
        datetime.strptime(dfdict["starttime"][0], "%d/%m/%Y-%H:%M:%S").strftime(
            "%d %b %Y"
        ),
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.15,
        0.015,
        "".join(["Level: ", str(float(levelinfo))]),
        fontsize=13,
        fontweight="bold",
    )

    plt.savefig(os.path.join(quicklook_graphics_dir, quicklook_figname))
    #    plt.show()
    plt.close(fig)
    return None
