"""
LIdar RAnge COrrected Signal - LIRACOS
This script provides tools to handle with range corrected signal graphics averaged and RCS maps, so-called Quick-looks graphics
This script uses range corrected data with all corrections applyied by LIPANCORA scripts
Created on Fri Jul 16 08:05:10 2021
@author: Fábio J. S. Lopes, Alexandre C. Yoshida and Alexandre Cacheffo
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from functions import liracos_function as lrc
from functions import milgrau_function as mf
from functions import qlhtml_function_amanda as qlf

rootdir_name = os.getcwd()
files_dir_level1 = "05-data_level1"
files_dir_to_read = "03-rcsignal"
file_dir_quicklooks = "04-quicklooks_graphics"
file_dir_meanrcs = "05-rcsignalmean_graphics"
html_dir = "10-ql_htmlfiles"
rcsscale = "11-RCS_scales"
datadir_name = os.path.join(rootdir_name, files_dir_level1)
rcsmax_dirname = os.path.join(rootdir_name, rcsscale, "01-maxrcs")
rcsmean_dirname = os.path.join(rootdir_name, rcsscale, "01-meanrcs")
rcsmin_dirname = os.path.join(rootdir_name, rcsscale, "01-minrcs")
version = "level1"

"""Reading all measurements directory"""
fileinfo, subfolderinfo = mf.readfiles_meastype(datadir_name)

lamb = [355, 532, 1064]
maxscale_alt = 15000  # max scale altitude for mean RCS graphics
maxscale_altql = 15000  # max scale altitude for quicklook RCS graphics
channelmode = "AN"
dfmaxrcstotal = pd.DataFrame()
dfmeanrcstotal = pd.DataFrame()
dfminrcstotal = pd.DataFrame()

# for i in trange(len(fileinfo),file=sys.stdout, desc='outer loop'):
# for i in trange(2,file=sys.stdout, desc='outer loop'):
for i in range(len(fileinfo)):
    alt = []
    rcsignal = []
    datafiles = []
    filenameheader = []
    rcstime = []
    maxrcs = []
    meanrcs = []
    minrcs = []
    dfmaxrcs = pd.DataFrame()
    dfmeanrcs = pd.DataFrame()
    dfminrcs = pd.DataFrame()

    for j in range(len(subfolderinfo)):
        datafiles.append(
            mf.readfiles_generic(os.path.join(fileinfo[i], subfolderinfo[j]))
        )
        if subfolderinfo[j] == files_dir_to_read:
            for filename in datafiles[j]:
                rcsignal.append(pd.read_csv(filename, sep=",", skiprows=range(0, 10)))
                filenameheader.append(mf.readdown_header(filename))
                dfdict = pd.DataFrame(filenameheader)

    for k in range(len(rcsignal)):
        alt = pd.DataFrame(list(range(len(rcsignal[k].index)))).mul(
            float(dfdict["vert_res"][k])
        ) + float(dfdict["vert_res"][k])
        alt.columns = ["altitude"]
        rcsignalmean = pd.concat(rcsignal).groupby(level=0).mean()
        rcstotal = pd.concat(rcsignal, axis=1)
        rcstime.append(
            datetime.strptime(dfdict["starttime"][k], "%d/%m/%Y-%H:%M:%S").strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )

    yeardir = Path(os.path.relpath(filename, datadir_name)).parts[-4]
    datedir = Path(os.path.relpath(filename, datadir_name)).parts[-3]
    rcsmeangraphics_dir = os.path.join(
        rootdir_name, files_dir_level1, yeardir, datedir, file_dir_meanrcs
    )
    mf.folder_creation(rcsmeangraphics_dir)
    quicklook_graphics_dir = os.path.join(
        rootdir_name, files_dir_level1, yeardir, datedir, file_dir_quicklooks
    )
    mf.folder_creation(quicklook_graphics_dir)

    for ii in range(len(lamb)):

        if lamb[ii] == 1064:
            qlchannelmode = "AN"
        else:
            qlchannelmode = channelmode

        rcslambda = rcstotal[str(lamb[ii]) + qlchannelmode].set_index(
            [pd.Index(alt["altitude"])]
        )
        rcslambda.columns = rcstime
        rcsheatmap = rcslambda.T.reset_index().rename(columns={"index": "Time_UTC"})
        rcsheatmap["Time_UTC"] = pd.to_datetime(
            rcsheatmap["Time_UTC"], format="%Y-%m-%d %H:%M:%S"
        )
        time_values = pd.DataFrame(columns=[rcslambda.index])
        time_values_list = []
        for ix in range(0, len(rcsheatmap["Time_UTC"]) - 1):
            dif = divmod(
                (
                    rcsheatmap["Time_UTC"][ix + 1] - rcsheatmap["Time_UTC"][ix]
                ).total_seconds(),
                1,
            )
            if dif[0] > float(61.0):
                nfactor = int(
                    dif[0]
                    // int(
                        (float(dfdict["shotnumber"][ix]) - 1)
                        / float(dfdict["laser_freq"][ix])
                    )
                )
                for jx in range(1, nfactor):
                    time_values_list.append(
                        rcsheatmap["Time_UTC"][ix] + jx * timedelta(seconds=60)
                    )

        missing_time_values = pd.concat(
            [time_values.T, pd.DataFrame(columns=time_values_list)], ignore_index=True
        )

        new_rcslambda = rcslambda.join(missing_time_values).fillna(0.0)

        new_rcslambda = new_rcslambda.T.reset_index()
        new_rcslambda.insert(loc=1, column="0", value=0)
        new_rcslambda["index"] = pd.to_datetime(
            new_rcslambda["index"], format="%Y-%m-%d %H:%M:%S"
        )
        new_rcslambda = new_rcslambda.sort_values(by="index").set_index("index")
        new_index_rcslambda = new_rcslambda.index.to_series().transform(
            lambda x: x.strftime("%H:%M")
        )
        new_rcslambda = new_rcslambda.set_index(new_index_rcslambda).T
        maxrcs.append(max(new_rcslambda.max()))
        meanrcs.append(new_rcslambda.mean().mean())
        minrcs.append(min(new_rcslambda.min()))
        dfmaxrcs[str(lamb[ii])] = [max(new_rcslambda.max())]
        dfmeanrcs[str(lamb[ii])] = [new_rcslambda.mean().mean()]
        dfminrcs[str(lamb[ii])] = [min(new_rcslambda.min())]

        lrc.ql(
            new_rcslambda,
            alt,
            rcstime,
            lamb[ii],
            qlchannelmode,
            dfdict,
            maxscale_altql,
            fileinfo[i],
            rootdir_name,
            version,
            quicklook_graphics_dir,
        )
    dfmaxrcs["data"] = pd.to_datetime(
        rcsheatmap["Time_UTC"][0], format="%Y-%m-%d"
    ).strftime("%Y-%m-%d")
    dfmaxrcstotal = pd.concat([dfmaxrcstotal, dfmaxrcs], ignore_index=True)
    dfmeanrcs["data"] = pd.to_datetime(
        rcsheatmap["Time_UTC"][0], format="%Y-%m-%d"
    ).strftime("%Y-%m-%d")
    dfmeanrcstotal = pd.concat([dfmeanrcstotal, dfmeanrcs], ignore_index=True)
    dfminrcs["data"] = pd.to_datetime(
        rcsheatmap["Time_UTC"][0], format="%Y-%m-%d"
    ).strftime("%Y-%m-%d")
    meanrcsfigurename, measperiod = lrc.meanrcs(
        rcsignalmean,
        alt,
        lamb,
        channelmode,
        dfdict,
        maxscale_alt,
        fileinfo[i],
        rootdir_name,
        version,
        rcsmeangraphics_dir,
    )
    qlf.qlhtml(
        dfdict,
        fileinfo[i],
        rootdir_name,
        html_dir,
        version,
        meanrcsfigurename,
        measperiod,
        channelmode,
        lamb,
    )


# dfmaxrcstotal.to_csv(rcsmax_dirname, index=False, float_format="%.4f")
# dfmeanrcstotal.to_csv(rcsmean_dirname, index=False, float_format="%.4f")
# dfminrcstotal.to_csv(rcsmin_dirname, index=False, float_format="%.4f")


# del i, ii, j, k, datedir, yeardir, lamb
