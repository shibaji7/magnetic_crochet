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
import utils


def fit_OLS(y, X, alphas=[0.05], set_log=True):
    import statsmodels.api as sm
    from scipy.stats import t
    X = sm.add_constant(X)
    if set_log: y = np.log10(y)
    model = sm.OLS(y, X)
    results = model.fit()
    params = results.params
    n = len(y)
    y_hat = 10**results.predict(X)
    solve = dict(
        model=model,
        results=results,
        X=X, y=y, y_hat=y_hat,
    )
    if X.shape[1] > 1:
        for alpha in alphas:
            t_critical = t.ppf(1 - alpha / 2, df=n-2)
            print(results.bse)
            slope_std_error = results.bse[1]
            confidence_interval = (
                params[1] - t_critical * slope_std_error, 
                params[1] + t_critical * slope_std_error
            )
            prediction_std_error = np.sqrt(results.scale)
            upper_bound = 10**(np.log10(y_hat) + t_critical * prediction_std_error)
            lower_bound = 10**(np.log10(y_hat) - t_critical * prediction_std_error)
            out = dict(
                alpha=alpha,
                confidence_interval=confidence_interval,
                y_upper=upper_bound,
                y_lower=lower_bound
            )
            solve[alpha] = out
    return solve

class Stats(object):
    """
    This class is responsible for
    generateing the statistics for HamSCI
    and(or) SuperDARN HF observations
    """

    def __init__(self, records, **keywrds):
        self.records = records
        for k, v in keywrds:
            setattr(self, k, v)
        return

    def run_hamsci_stats(self):
        """
        Conduct the analysis
        """
        #######################
        # "flare_rise_time", "flare_fall_time"
        # "peak_ESPquad", "peak_xray_a", "peak_xray_b",
        # "peak_dI_ESPquad", "peak_dI_xray_a", 
        # "peak_dI_xray_b", "sza"
        #######################
        Xs, ys = [
            "sza", "peak_xray_a", 
            "peak_xray_b", "peak_ESPquad"
        ], ["doppler_peak"]
        logger.info(f"Dataset lengths [{len(self.records)}]")
        events = self.records[Xs+ys].dropna()
        events = events[
            (events.doppler_peak >= 0.1)
            & (events.doppler_peak <= 10)
            & (events.sza >= 0)
            & (events.sza <= 90)
        ]
        logger.info(f"non-NAN, filtered lengths [{len(events)}]")
        uq = 0.32
        import matplotlib as mpl
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(dpi=300, figsize=(3*len(Xs),3), nrows=1, ncols=len(Xs), sharey=True)
        kinds = ["linear", "log", "log", "log"]
        for i, p, ax in zip(range(len(axs)), Xs, axs):
            e = events.copy()
            f_ols = fit_OLS(events[ys[0]], events[p], [uq])
            ax.scatter(events[p], events[ys[0]], s=0.1, color="k", label="Observations")
            ax.set_yscale("log")
            if kinds[i] == "linear":
                x, y, yerr = self.binning_linearX(e, p, ys[0], nbins=30)
            else:
                x, y, yerr = self.binning_logX(e, p, ys[0], nbins=30)
            nmi = utils.compute_normalized_MI(x[~np.isnan(y)], y[~np.isnan(y)], state=0)
            ax.text(
                0.9, 1.05, r"$\mathcal{N}$=%.2f"%(nmi),
                ha="right", va="center", transform=ax.transAxes
            )
            ax.errorbar(
                x, y, yerr, markersize=1.8, ecolor="red", fmt="o",
                elinewidth=0.5, capsize=0.5, capthick=0.5,
                mec="red", ls="None", zorder=3
            )
            if i > 0: ax.set_xscale("log")
            # e["fit"], e["ub"], e["lb"] = (
            #     f_ols["y_hat"],
            #     f_ols[uq]["y_upper"],
            #     f_ols[uq]["y_lower"],
            # )
            # e = e.sort_values(by=p)
            # ax.plot(e[p], e.fit, "r-", lw=1.2, label="Best Fit")
            # ax.plot(e[p], e.ub, "b-", lw=1.2, label=r"1-$\sigma$")
            # ax.plot(e[p], e.lb, "b-", lw=1.2)

        axs[0].set_ylabel(fr"Doppler Peak, $D_H$ (Hz)")
        axs[0].set_xlabel("SZA (Deg)")
        axs[1].set_xlabel(r"X-ray ($\lambda_0$), $Wm^{-2}$")
        axs[2].set_xlabel(r"X-ray ($\lambda_1$), $Wm^{-2}$")
        axs[3].set_xlabel(r"EUV (0.1-7 nm), $Wm^{-2}$")
        fig.subplots_adjust(wspace=0.15, hspace=0.3)
        fig.savefig("data/analysis/scatter.png", bbox_inches="tight")

        Xs, ys = [
            "peak_dI_xray_a", "peak_dI_xray_b", 
            "peak_dI_ESPquad",
        ], ["doppler_peak"]
        kinds = ["log", "log", "log", "linear"]
        logger.info(f"Dataset lengths [{len(self.records)}]")
        events = self.records[["sza"]+Xs+ys].dropna()
        events = events[
            (events.doppler_peak >= 0.1)
            & (events.doppler_peak <= 10)
            & (events.sza >= 0)
            & (events.sza <= 90)
        ]
        logger.info(f"non-NAN, filtered lengths [{len(events)}]")
        fig, axs = plt.subplots(dpi=300, figsize=(3*len(Xs),3), nrows=1, ncols=len(Xs), sharey=True)
        for i, p, ax in zip(range(len(axs)), Xs, axs):
            e = events.copy()
            f_ols = fit_OLS(events[ys[0]], events[p], [uq])
            ax.scatter(events[p], events[ys[0]], s=0.1, color="k", label="Observations")
            if i < 3: ax.set_yscale("log")
            if kinds[i] == "linear":
                x, y, yerr = self.binning_linearX(e, p, ys[0], nbins=30)
            else:
                x, y, yerr = self.binning_logX(e, p, ys[0], nbins=30)
            nmi = utils.compute_normalized_MI(x[~np.isnan(y)], y[~np.isnan(y)], state=0)
            ax.text(
                0.9, 1.05, r"$\mathcal{N}$=%.2f"%(nmi),
                ha="right", va="center", transform=ax.transAxes
            )
            ax.errorbar(
                x, y, yerr, markersize=1.8, ecolor="red", fmt="o",
                elinewidth=0.5, capsize=0.5, capthick=0.5,
                mec="red", ls="None", zorder=3
            )
            ax.set_xscale("log")

        axs[0].set_ylabel(fr"Doppler Peak, $D_H$ (Hz)")
        axs[0].set_xlabel(r"$\left[\frac{\partial\Phi_0}{\partial t}\right]_{P}$ ($\lambda_0$), $Wm^{-2}s^{-1}$")
        axs[1].set_xlabel(r"$\left[\frac{\partial\Phi_0}{\partial t}\right]_{P}$ ($\lambda_1$), $Wm^{-2}s^{-1}$")
        axs[2].set_xlabel(r"$\left[\frac{\partial\Phi_0}{\partial t}\right]_{P}$ (EUV, 0.1-7 nm), $Wm^{-2}s^{-1}$")
        fig.subplots_adjust(wspace=0.15, hspace=0.3)
        fig.savefig("data/analysis/scatter_dl.png", bbox_inches="tight")

        Xs, ys = ["lon"], ["doppler_peak"]
        kinds = ["linear"]
        logger.info(f"Dataset lengths [{len(self.records)}]")
        events = self.records[["sza"]+Xs+ys].dropna()
        events = events[
            (events.doppler_peak >= 0.1)
            & (events.doppler_peak <= 10)
            & (events.sza >= 0)
            & (events.sza <= 90)
        ]
        logger.info(f"non-NAN, filtered lengths [{len(events)}]")
        fig, axs = plt.subplots(dpi=300, figsize=(3*len(Xs),3), nrows=1, ncols=len(Xs), sharey=True)
        for i, p, ax in zip(range(len([axs])), Xs, [axs]):
            e = events.copy()
            f_ols = fit_OLS(events[ys[0]], events[p], [uq])
            ax.scatter(events[p], events[ys[0]], s=0.1, color="k", label="Observations")
            if i < 3: ax.set_yscale("log")
            if kinds[i] == "linear":
                x, y, yerr = self.binning_linearX(e, p, ys[0], nbins=30)
            else:
                x, y, yerr = self.binning_logX(e, p, ys[0], nbins=30)
            nmi = utils.compute_normalized_MI(x[~np.isnan(y)], y[~np.isnan(y)], state=0)
            ax.text(
                0.9, 1.05, r"$\mathcal{N}$=%.2f"%(nmi),
                ha="right", va="center", transform=ax.transAxes
            )
            ax.errorbar(
                x, y, yerr, markersize=1.8, ecolor="red", fmt="o",
                elinewidth=0.5, capsize=0.5, capthick=0.5,
                mec="red", ls="None", zorder=3
            )
            #ax.set_xscale("log")

        axs.set_ylabel(fr"Doppler Peak, $D_H$ (Hz)")
        axs.set_xlabel(r"Lon (Deg)")
        # axs[1].set_xlabel(r"$\left[\frac{\partial\Phi_0}{\partial t}\right]_{P}$ ($\lambda_1$), $Wm^{-2}s^{-1}$")
        # axs[2].set_xlabel(r"$\left[\frac{\partial\Phi_0}{\partial t}\right]_{P}$ (EUV, 0.1-7 nm), $Wm^{-2}s^{-1}$")
        fig.subplots_adjust(wspace=0.15, hspace=0.3)
        fig.savefig("data/analysis/scatter_distance.png", bbox_inches="tight")

        Xs, ys = [
            "energy_xray_a", "energy_xray_b", "energy_ESPquad"
        ], ["doppler_rise_area"]
        kinds = ["log", "log", "log"]
        logger.info(f"Dataset lengths [{len(self.records)}]")
        events = self.records[["sza"]+Xs+ys].dropna()
        events = events[
            (events.doppler_rise_area >= 10)
            & (events.doppler_rise_area <= 10000)
            & (events.sza >= 0)
            & (events.sza <= 90)
        ]
        logger.info(f"non-NAN, filtered lengths [{len(events)}]")
        print(events.columns)
        fig, axs = plt.subplots(dpi=300, figsize=(3*len(Xs),3), nrows=1, ncols=len(Xs), sharey=True)
        for i, p, ax in zip(range(len(axs)), Xs, axs):
            e = events.copy()
            ax.scatter(events[p], events[ys[0]], s=0.1, color="k", label="Observations")
            if i < 3: ax.set_yscale("log")
            x, y, yerr = self.binning_logX(e, p, ys[0], nbins=30)
            # nmi = utils.compute_normalized_MI(x[~np.isnan(y)], y[~np.isnan(y)], state=0)
            # ax.text(
            #     0.9, 1.05, r"$\mathcal{N}$=%.2f"%(nmi),
            #     ha="right", va="center", transform=ax.transAxes
            # )
            ax.errorbar(
                x, y, yerr, markersize=1.8, ecolor="red", fmt="o",
                elinewidth=0.5, capsize=0.5, capthick=0.5,
                mec="red", ls="None", zorder=3
            )
            ax.set_xscale("log")

        # axs[0].set_ylabel(fr"Doppler Peak, $D_H$ (Hz)")
        # axs[0].set_xlabel(r"$\left[\frac{\partial\Phi_0}{\partial t}\right]_{P}$ ($\lambda_0$), $Wm^{-2}s^{-1}$")
        # axs[1].set_xlabel(r"$\left[\frac{\partial\Phi_0}{\partial t}\right]_{P}$ ($\lambda_1$), $Wm^{-2}s^{-1}$")
        # axs[2].set_xlabel(r"$\left[\frac{\partial\Phi_0}{\partial t}\right]_{P}$ (EUV, 0.1-7 nm), $Wm^{-2}s^{-1}$")
        fig.subplots_adjust(wspace=0.15, hspace=0.3)
        fig.savefig("data/analysis/scatter_area.png", bbox_inches="tight")
        return

    def binning_linearX(self, records, Xname, yname, nbins=20):
        from scipy.stats import median_abs_deviation
        xbins = np.linspace(records[Xname].min(), records[Xname].max(), nbins+1)
        xs = [(xbins[i] + xbins[i+1])/2 for i in range(nbins)]
        ys, yerr = [], []
        for i in range(nbins):
            o = records[
                (records[Xname] >= xbins[i])
                & (records[Xname] <= xbins[i+1])
            ]
            ys.append(o[yname].median())
            yerr.append(median_abs_deviation(o[yname]))
        return np.array(xs), np.array(ys), np.array(yerr)

    def binning_logX(self, records, Xname, yname, nbins=40):
        from scipy.stats import median_abs_deviation
        log_xbins = np.linspace(
            0.01 if records[Xname].min() == 0 else np.log10(records[Xname].min()), 
            np.log10(records[Xname].max()), 
            nbins+1
        )
        log_xs = [(log_xbins[i] + log_xbins[i+1])/2 for i in range(nbins)]
        ys, yerr = [], []
        print(log_xbins, records[Xname].min(), records[Xname].max())
        for i in range(nbins):
            o = records[
                (records[Xname] >= 10**log_xbins[i])
                & (records[Xname] <= 10**log_xbins[i+1])
            ]
            ys.append(o[yname].median())
            yerr.append(median_abs_deviation(o[yname]))
        return 10**np.array(log_xs), np.array(ys), np.array(yerr)

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

def get_bearing(start_loc, stop_loc):
    """
    Bearing between two lat/long coordinates: 
        (lat1, lon1), (lat2, lon2)
    """
    import math
    dLon = stop_loc[1] - start_loc[1]
    y = math.sin(dLon) * math.cos(stop_loc[0])
    x = (
        (math.cos(start_loc[0])*math.sin(stop_loc[0])) - 
        (math.sin(start_loc[0])*math.cos(stop_loc[0])*math.cos(dLon))
    )
    brng = np.rad2deg(math.atan2(y, x))
    if brng < 0: brng+= 360
    return brng

def calculate_zenith_angle(start_loc, stop_loc, d):
    """
    Thing to try for SZA: instead of computing it for 
    the lat/lon of the station, we might try computing 
    the halfway point between the station lat/lon and WWV 
    using the haversine formula, and calculating the SZA 
    for that point 600 km up.
    """
    from geopy import distance
    dist = distance.distance(start_loc, stop_loc).km/2
    brng = get_bearing(start_loc, stop_loc)    
    destination = distance.distance(kilometers=dist).destination(start_loc, brng)
    za = get_altitude(
        destination.latitude,
        destination.longitude,
        d
    )
    return za

def load_stagged_events():
    fname = "data/events.csv"
    start_loc = (
        40.014984,
        -105.270546
    )
    if not os.path.exists(fname):
        import glob
        folders = glob.glob("data/stage/*")
        files = [f + "/stage.json" for f in folders]
        dataset, number_of_events = [], 0
        for f in files:
            if os.path.exists(f):
                d = dt.datetime.strptime(f.split("/")[-2], "%Y-%m-%d-%H-%M")
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
                        dx["lat"], dx["lon"], dx["freq"], dx["call_sign"], dx["doppler_rise_area"],\
                            dx["doppler_peak"], dx["doppler_fall_area"] = (
                            h["lat"], h["lon"], h["freq"], h["call_sign"], 
                            h["rise_area"], h["peak"], h["fall_area"]
                        )
                        d = d.replace(tzinfo=dt.timezone.utc)
                        dx["sza"] = calculate_zenith_angle(start_loc, (h["lat"], h["lon"]), d)
                        dx["date"] = d
                        dx["distance"] = utils.great_circle(h["lon"], h["lat"], start_loc[1], start_loc[0])
                        dataset.append(dx)
        df = pd.DataFrame.from_records(dataset)
        df["number_of_events"] = number_of_events
        logger.warning(f"Long list of PSWS/events: {len(df)}/{number_of_events}")
        df.to_csv(fname, index=False, float_format="%g")
    else:
        df = pd.read_csv(fname, parse_dates=["date"])
        logger.warning(f"Long list of events: {len(df)}")
    Stats(df).run_hamsci_stats()
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
    # create_event_list(2021)
    # create_event_list(2022)
    # create_event_list(2023)
    # run_event("config/events_2021.csv", "X")
    # run_event("config/events_2022.csv", "X")
    # run_event("config/events_2023.csv", "X")
    # run_event("config/events_2021.csv", "M")
    # run_event("config/events_2022.csv", "M")
    # run_event("config/events_2023.csv", "M")
    # run_event("config/events_2021.csv", "C")
    # run_event("config/events_2022.csv", "C", runstart=17, runend=-1)
    # run_event("config/events_2023.csv", "C", runstart=18)
    load_stagged_events()
    
