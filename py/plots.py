#!/usr/bin/env python

"""
    plotFoV.py: module to plot Fan plots with various transformation
"""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import sys
sys.path.extend(["py/", "py/fetch/", "py/geo/"])
import datetime as dt
import pandas as pd
import numpy as np

from plotFoV import Fan
from flare import FlareTS
from darn import FetchData
from hamsci import HamSci
from plotRTI import GOESSDOPlot, RTI, joyplot, DIPlot
from analysis import Stats

def plot_fov_plots():
    nodes = get_all_nodes()
    date = dt.datetime(2023,12,14,17,2)
    rads = ["fhe","fhw", "bks", "gbr", "kap", "sas", "cve", "cvw", "pgr"]
    rads = ["fhe","fhw"]
    fan = Fan(rads, date)
    fan.generate_fov(nodes)
    fan.save("data/analysis/fov.png")
    fan.close()
    return

def get_all_nodes():
    import glob
    date_folders = glob.glob("data/20*")
    node_dic = {}
    for d in date_folders:
        files = glob.glob(d + "/hamsci/*.csv")
        for file in files:
            with open(file,"r") as fp:
                print(file)
                lines = fp.readlines()[:16]
                if len(lines) > 10:
                    node = list(filter(None, lines[4].replace("\n", "").split(" ")))[4]
                    call = list(filter(None, lines[5].replace("\n", "").split(" ")))[2]
                    geo = list(filter(None, lines[7].replace("\n", "").split("         ")))
                    geo = geo[-1].split(",")
                    lat, lon, elv = np.round(float(geo[0]), 2), np.round(float(geo[1]), 2), np.round(float(geo[2]), 2)
                    id = call.upper()
                    if (not id in node_dic) and node!="N0000000":
                        node_dic[id] = dict(
                            id=id, node=node, call=call.upper(), lat=lat, lon=lon, elv=elv
                        )
                        print(node, call.upper(), f"{lat} N, {lon} E")
    node_dic["WWV"] = dict(
        node="N/1", call="WWV/H", 
        lat=40.67836111111111, lon=-105.04052777777778, 
        elv=0
    )
    return node_dic

def plot_goes_sdo_observations():
    dates = [dt.datetime(2023,12,14,16), dt.datetime(2023,12,14,20)]
    fts = FlareTS(dates)
    time = fts.dfs["goes"].time.tolist()
    xs, xl = (fts.dfs["goes"].xrsa, fts.dfs["goes"].xrsb)
    sdo_time, sdo_euv = (
        fts.dfs["eve"].time.tolist(),
        fts.dfs["eve"]["0.1-7ESPquad"]
    )
    vlines, colors = (
        [fts.flare["event_peaktime"], 
        fts.flare["event_starttime"], 
        fts.flare["event_endtime"]],
        ["r", "orange", "g"],
    )
    GOESSDOPlot(
        time, xl, xs, sdo_time, sdo_euv, "data/analysis/goessdo.png",
        vlines, colors, drange=dates
    )
    return

def plot_sd_event():
    rads = ["fhe","fhw"]
    ev = dt.datetime(2023,12,14,17,2)
    base = "data/{Y}-{m}-{d}-{H}-{M}/".format(
        Y=ev.year,
        m="%02d" % ev.month,
        d="%02d" % ev.day,
        H="%02d" % ev.hour,
        M="%02d" % ev.minute,
    )
    dates = [dt.datetime(2023,12,14,16), dt.datetime(2023,12,14,20)]
    fts = FlareTS(dates)
    vlines, colors = (
        [fts.flare["event_peaktime"], 
        fts.flare["event_starttime"], 
        fts.flare["event_endtime"]],
        ["r", "orange", "g"],
    )
    darns = FetchData.fetch(base, ["fhe","fhw"], dates)
    f0 = (darns["fhe"].records["tfreq"]/1000).mean().round(1)
    rti = RTI(
        100, dates, num_subplots=2,
        fig_title=fr"FHE/W, {dates[0].strftime('%Y-%m-%d')}, Beam: 7, $f_0$= {f0}MHz",
    )
    ax = rti.addParamPlot(darns["fhe"].records, 7, "", cbar=True, p_max=30, p_min=-30, xlabel="")
    rti.add_vlines(ax, vlines, colors)
    ax.text(0.05, 0.95, "(a)", ha="left", va="center", transform=ax.transAxes)
    ax = rti.addParamPlot(darns["fhw"].records, 7, "", cbar=True, p_max=30, p_min=-30)            
    rti.add_vlines(ax, vlines, colors)
    ax.text(0.05, 0.95, "(b)", ha="left", va="center", transform=ax.transAxes)
    rti.save("data/analysis/RTI.png")
    rti.close()
    return

def plot_hamsci_event():
    ev = dt.datetime(2023,12,14,17,2)
    base = "data/{Y}-{m}-{d}-{H}-{M}/".format(
        Y=ev.year,
        m="%02d" % ev.month,
        d="%02d" % ev.day,
        H="%02d" % ev.hour,
        M="%02d" % ev.minute,
    )
    dates = [dt.datetime(2023,12,14,16), dt.datetime(2023,12,14,17,30)]
    fts = FlareTS(dates)
    vlines, colors = (
        [fts.flare["event_peaktime"], 
        fts.flare["event_starttime"], 
        fts.flare["event_endtime"]],
        ["r", "orange", "g"],
    )
    ham = HamSci(base, dates, None)
    ham.setup_plotting()
    flare_timings = pd.Series(
        [
            fts.flare["event_starttime"].iloc[0],
            fts.flare["event_peaktime"].iloc[0],
            fts.flare["event_endtime"].iloc[0],
        ]
    )
    flare_timings = flare_timings.dt.tz_localize("UTC")
    gds = [ham.gds[1]]
    joyplot(
            ham.gds[1:],
            f"data/analysis/joyplot.png",
            flare_timings,
            vlines=vlines,
            colors=colors,
            drange=dates,
            ccolors=["r", "b", "k"]
        )
    return

def plot_time_hamsci_dI():
    ev = dt.datetime(2023,12,14,17,2)
    base = "data/{Y}-{m}-{d}-{H}-{M}/".format(
        Y=ev.year,
        m="%02d" % ev.month,
        d="%02d" % ev.day,
        H="%02d" % ev.hour,
        M="%02d" % ev.minute,
    )
    dates = [dt.datetime(2023,12,14,16), dt.datetime(2023,12,14,17,30)]
    fts = FlareTS(dates)
    time = fts.dfs["goes"].time.tolist()
    xs, xl = (fts.dfs["goes"].xrsa, fts.dfs["goes"].xrsb)
    sdo_time, sdo_euv = (
        fts.dfs["eve"].time.tolist(),
        fts.dfs["eve"]["0.1-7ESPquad"]
    )
    vlines, colors = (
        [fts.flare["event_peaktime"], 
        fts.flare["event_starttime"], 
        fts.flare["event_endtime"]],
        ["r", "orange", "g"],
    )
    ham = HamSci(base, dates, None)
    ham.setup_plotting()
    flare_timings = pd.Series(
        [
            fts.flare["event_starttime"].iloc[0],
            fts.flare["event_peaktime"].iloc[0],
            fts.flare["event_endtime"].iloc[0],
        ]
    )
    flare_timings = flare_timings.dt.tz_localize("UTC")
    
    DIPlot(
        time, xl, xs, sdo_time, sdo_euv, ham.gds[3], "data/analysis/goessdoDI.png",
        vlines, colors, drange=dates
    )
    return

def plot_statistics(hamsci_event_list="config/hamsci_events.csv"):
    records = pd.read_csv(
        hamsci_event_list, 
        parse_dates=["event", "start", "end", "s_time", "e_time"]
    ).to_dict(orient="records")
    for rec in records:
        rec["call_sign"] = rec["call_sign"].split("-")
        rec["summary_file"] = f"data/stage/{rec['event'].strftime('%Y-%m-%d-%H-%M')}/stage0.mat"
    stat = Stats(records)
    stat.run_hamsci_stats("data/analysis/scatter.png")
    return

if __name__ == "__main__":
    plot_time_hamsci_dI()
    #plot_statistics()
    #plot_hamsci_event()
    #plot_sd_event()
    #plot_goes_sdo_observations()
    #plot_fov_plots()
    #get_all_nodes()
