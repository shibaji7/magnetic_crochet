#!/usr/bin/env python

"""analysis.py: module is dedicated to fetch, filter, and save data for post_processing."""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import datetime as dt
import os
import sys
import utils
import json

import numpy as np
import pandas as pd
from loguru import logger
from pysolar.solar import get_altitude

sys.path.extend(["py/", "py/fetch/", "py/geo/"])
from plotFoV import Fan
from plotRTI import RTI, GOESSDOPlot, HamSciParamTS, HamSciTS, joyplot

class Hopper(object):
    """
    This class is responsible for following
    operations for each flare event.
        i) Fetching HamSci, SD, SM, GOES dataset using .fetch module.
        ii) Summary plots for the dataset.
        iii) Store the data for post-processing.
    """

    def __init__(
        self,
        base,
        dates,
        rads,
        event,
        event_start,
        event_end,
        uid="shibaji7",
        mag_stations=None,
        stg=True,
    ):
        """
        Populate all data tables from GOES, FISM2, SuperDARN, and SuperMAG.
        """

        from flare import FlareTS
        from hamsci import HamSci

        self.base = base
        self.dates = dates
        self.rads = rads
        self.event = event
        self.event_start = event_start
        self.event_end = event_end
        self.uid = uid
        self.mag_stations = mag_stations
        if not os.path.exists(base):
            os.makedirs(base)
        self.stage_base = self.base.replace("data/", "data/stage/") 
        self.stage_filename = self.stage_base + "stage.json"
        if not os.path.exists(self.stage_filename):
            self.flareTS = FlareTS(self.dates)
            self.GOESplot()
            self.hamSci = HamSci(self.base, self.dates, None)
            self.GrepMultiplot()
            if stg:
                self.stage_analysis()
        return

    def GOESplot(self):
        """
        Create GOES plots
        """
        base = self.base + "figures/"
        os.makedirs(base, exist_ok=True)
        fname = f"{base}GOES.png"
        time = self.flareTS.dfs["goes"].time.tolist()
        xs, xl = (self.flareTS.dfs["goes"].xrsa, self.flareTS.dfs["goes"].xrsb)
        vlines, colors = (
            [self.event, self.event_start, self.event_end],
            ["k", "r", "r"],
        )
        sdo_time, sdo_euv = (
            self.flareTS.dfs["eve"].time.tolist(),
            self.flareTS.dfs["eve"]["0.1-7ESPquad"]
        ) if len(self.flareTS.dfs["eve"]) > 0 else (np.array([np.nan]), np.array([np.nan]))
        GOESSDOPlot(time, xl, xs, sdo_time, sdo_euv, fname, vlines, colors, drange=self.dates)
        return
    
    def GrepMultiplot(self):
        """
        Create Latitude longitude dependent plots in Grape
        """
        parse_fill_nodelist()
        vlines, colors = (
            [self.event, self.event_start, self.event_end],
            ["k", "r", "r"],
        )
        base = self.base + "figures/"
        os.makedirs(base, exist_ok=True)
        fname = f"{base}hamsci.png"
        self.hamSci.setup_plotting()
        HamSciTS(self.hamSci.gds, fname, drange=self.dates)
        flare_timings = pd.Series(
            [
                self.event_start,
                self.event,
                self.event_end,
            ]
        )
        flare_timings = flare_timings.dt.tz_localize("UTC")
        self.hamSci.extract_parameters(flare_timings)
        for i in range(len(self.hamSci.gds)):
            fname = f"{base}hamsci_{i}.png"
            HamSciParamTS(
                self.hamSci.gds,
                fname,
                flare_timings,
                drange=self.dates,
                index=i,
                vlines=vlines,
                colors=colors,
            )
        joyplot(
            self.hamSci.gds,
            f"{base}joyplot.png",
            flare_timings,
            vlines=vlines,
            colors=colors,
            drange=self.dates,
        )
        return

    def stage_analysis(self):
        """
        Stage dataset for next analysis
        """
        
        os.makedirs(self.stage_base, exist_ok=True)
        self.stage = dict(hamsci=None, flare=None)
        self.stage["hamsci"] = self.hamSci.extract_stagging_data()
        self.stage["flare"] = self.flareTS.extract_stagging_data()
        self.stage["flare"]["rise_time"] = (
            self.event - self.event_start
        ).total_seconds()
        self.stage["flare"]["fall_time"] = (self.event_end - self.event).total_seconds()
        logger.info(f"file: {self.stage_filename}")
        with open(self.stage_filename, "w") as f:
            f.write(json.dumps(self.stage, indent=4, sort_keys=True))
        return

def parse_fill_nodelist():
    import glob
    folders = glob.glob("data/20*")
    df = pd.read_csv("config/nodelist.csv")
    nodes = df["Node #"].tolist()
    dlist = []
    for folder in folders:
        files = glob.glob(folder + "/hamsci/*.csv")
        for file in files:
            node = int(file.split("_")[1].replace("N", ""))
            if not node in nodes:
                with open(file, "r") as fp:
                    lines = fp.readlines()[:20]
                d = dict()
                d["Node #"] = node
                d["Callsign"] = lines[5].replace("# Callsign", "").replace("\n", "").strip()
                d["Name"] = ""
                d["Grid Square"] = lines[6].replace("# Grid Square", "").replace("\n", "").strip()
                geo = lines[7].replace("# Lat, Long, Elv", "").replace("\n", "").strip()
                d["Latitude"], d["Longitude"], d["Elevation (M)"] = (
                    float(geo.split(",")[0]), 
                    float(geo.split(",")[1]), 
                    float(geo.split(",")[2])
                )
                d["Radio"] = lines[9].replace("# Radio1", "").replace("\n", "").strip()
                d["Antenna"] = lines[11].replace("# Antenna", "").replace("\n", "").strip()
                d["System"] = lines[13].replace("# System Info", "").replace("\n", "").strip()
                d["Magnetometer"] = ""
                d["Temperature Sensor"] = ""
                dlist.append(d)
    do = pd.DataFrame.from_records(dlist)
    df = pd.concat([df, do])
    df = df.drop_duplicates(keep="first", inplace=False)
    df.to_csv("config/nodelist.csv", header=True, index=False)
    return

def create_event_list(year=2023):
    fname = f"config/events_{year}.csv"
    if not os.path.exists(fname):
        def parse(r):
            r["rads"], r["is_analyzable"] = "-", 0
            r["s_time"] = (r["event_starttime"] - dt.timedelta(minutes=30)).replace(minute=0)
            r["e_time"] = (r["event_endtime"] + dt.timedelta(minutes=90)).replace(minute=0)
            return r

        from flare import FlareInfo
        start, end = (dt.datetime(year, 1, 1), dt.datetime(year, 12, 31))
        days = 1  + (end-start).days
        dates = [start + dt.timedelta(d) for d in range(days)]
        flares = pd.DataFrame()
        for d in dates:
            logger.info(f"Date: {d}")
            fi = FlareInfo([d, d+dt.timedelta(1)])
            if len(fi.flare)>0:
                flares = pd.concat([flares, fi.flare])
        flares["hour"] = flares["event_peaktime"].apply(lambda x: x.hour)
        flares = flares[(flares.hour>13)]
        flares.drop(columns=["hour"], inplace=True)
        flares = flares.apply(parse, axis=1)
        flares.rename(
            columns={
                "event_starttime": "start",
                "event_peaktime": "event",
                "event_endtime": "end",
                "fl_goescls": "fclass"
            },
            inplace=True
        )
        flares.to_csv(fname, index=False, header=True)
    return

def load_stagged_events():
    fname = "data/events.csv"
    if not os.path.exists(fname):
        import glob
        folders = glob.glob("data/stage/*")
        files = [f + "/stage.json" for f in folders]
        dataset, number_of_events = [], 0
        for f in files:
            if os.path.exists(f):
                number_of_events += 1
                with open(f,"r") as fp:
                    o = json.loads("\n".join(fp.readlines()))
                    for h in o["hamsci"]:
                        dx = {}
                        dx["flare_rise_time"], dx["flare_fall_time"] = (
                            o["flare"]["rise_time"],
                            o["flare"]["fall_time"]
                        )
                        dx["energy_ESPquad"] = o["flare"]["energy"]["ESPquad"]
                        dx["energy_xray_a"] = o["flare"]["energy"]["xray_a"]
                        dx["energy_xray_b"] = o["flare"]["energy"]["xray_b"]
                        dx["peak_ESPquad"] = o["flare"]["peaks"]["ESPquad"]
                        dx["peak_xray_a"] = o["flare"]["peaks"]["xray_a"]
                        dx["peak_xray_b"] = o["flare"]["peaks"]["xray_b"]
                        dx["peak_dI_ESPquad"] = o["flare"]["peak_of_dI"]["ESPquad"]
                        dx["peak_dI_xray_a"] = o["flare"]["peak_of_dI"]["xray_a"]
                        dx["peak_dI_xray_b"] = o["flare"]["peak_of_dI"]["xray_b"]
                        dx["lat"], dx["lon"], dx["freq"], dx["call_sign"], dx["rise_area"],\
                            dx["peak"], dx["fall_area"] = (
                            h["lat"], h["lon"], h["freq"], h["call_sign"], 
                            h["rise_area"], h["peak"], h["fall_area"]
                        )
                        dataset.append(dx)
        df = pd.DataFrame.from_records(dataset)
        df["number_of_events"] = number_of_events
        logger.warning(f"Long list of PSWS/events: {len(df)}/{number_of_events}")
        df.to_csv(fname, index=False, float_format="%g")
    else:
        df = pd.read_csv(fname)
        logger.warning(f"Long list of events: {len(df)}")
    return

def run_event_countdown(i, row, L):
    try:
        ev = row["event"]
        logger.warning(f"Event number: {i} of {L} <:> {ev}")        
        row["e_time"], row["s_time"] = (
            pd.to_datetime(row["e_time"]),
            pd.to_datetime(row["s_time"])
        )
        base = "data/{Y}-{m}-{d}-{H}-{M}/".format(
            Y=ev.year,
            m="%02d" % ev.month,
            d="%02d" % ev.day,
            H="%02d" % ev.hour,
            M="%02d" % ev.minute,
        )
        dates = [row["s_time"], row["e_time"]]
        row["rads"] = row["rads"].replace(" ", "")
        rads = row["rads"].split("-") if len(str(row["rads"])) > 0 else []
        Hopper(base, dates, rads, ev, row["start"], row["end"])
    except:
        logger.error("System error!!")
    return 0

def run_event(file="config/events_2023.csv", ftype="X", runstart=0, runend=-1):
    """
    Load all the events from
    events list files and fork Hopper
    to pre-process and store the
    dataset.
    """
    o = pd.read_csv(
        file, 
        parse_dates=["event", "start", "end", "s_time", "e_time"],
    )
    o = o[o.fclass.str.contains(ftype)]
    o = o.reset_index()
    o["stage_json_fname"] = o.event.apply(
        lambda ev: "data/stage/{Y}-{m}-{d}-{H}-{M}/stage.json".format(
            Y=ev.year,
            m="%02d" % ev.month,
            d="%02d" % ev.day,
            H="%02d" % ev.hour,
            M="%02d" % ev.minute,
        )
    )
    logger.info(f"Total detected flares: {len(o)}")
    o = o.drop([i for i, row in o.iterrows() if os.path.exists(row["stage_json_fname"])])
    o = o.reset_index().iloc[runstart:runend]
    logger.info(f"Total flares: {len(o)}")
    from joblib import Parallel, delayed
    _ = Parallel(n_jobs=8)(delayed(run_event_countdown)(i, row, len(o)) for i, row in o.iterrows())    
    return

if __name__ == "__main__":
    create_event_list(2021)
    create_event_list(2022)
    create_event_list(2023)
    run_event("config/events_2021.csv", "X")
    run_event("config/events_2022.csv", "X")
    run_event("config/events_2023.csv", "X")
    run_event("config/events_2021.csv", "M")
    run_event("config/events_2022.csv", "M")
    run_event("config/events_2023.csv", "M")
    run_event("config/events_2021.csv", "C")
    run_event("config/events_2022.csv", "C", runstart=17, runend=-1)
    run_event("config/events_2023.csv", "C", runstart=18)
    load_stagged_events()
    
