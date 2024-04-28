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

import numpy as np
import pandas as pd
from loguru import logger
from scipy.io import savemat, loadmat
from pysolar.solar import get_altitude

sys.path.extend(["py/", "py/fetch/", "py/geo/"])

from plotFoV import Fan
from plotRTI import RTI, GOESPlot, HamSciParamTS, HamSciTS, joyplot

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

def calculate_zenith_angle_by_location(loc, d):
    za = get_altitude(
        loc[0], loc[1], d
    )
    return za

def fit_OLS(y, X, alphas=[0.05]):
    import statsmodels.api as sm
    from scipy.stats import t
    X = sm.add_constant(X)
    y = np.log10(y)
    model = sm.OLS(y, X)
    results = model.fit()
    _, slope = results.params
    n = len(y)
    y_hat = 10**results.predict(X)
    solve = dict(
        model=model,
        results=results,
        X=X, y=y, y_hat=y_hat,
    )
    for alpha in alphas:
        t_critical = t.ppf(1 - alpha / 2, df=n-2)
        slope_std_error = results.bse[1]
        confidence_interval = (
            slope - t_critical * slope_std_error, 
            slope + t_critical * slope_std_error
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

    def __init__(self, records, kind="HamSCI", **keywrds):
        self.records = records
        self.kind = kind
        for k, v in keywrds:
            setattr(self, k, v)
        for rec in self.records:
            rec["details"] = loadmat(rec["summary_file"])
        return

    def run_hamsci_stats(self, fname):
        """
        Conduct the analysis
        """
        events = []
        for rec in self.records:
            flare, hamsci = (
                rec["details"]["flare"],
                rec["details"]["hamsci"]
            )
            for i in range(rec["details"]["hamsci"].shape[1]):
                x, d = (
                    hamsci[0,i], 
                    rec["event"]
                )
                d = d.replace(tzinfo=dt.timezone.utc).to_pydatetime()
                call = x["call_sign"][0][0][0]+"/"+str(x["node"][0][0][0][0])
                if call in rec["call_sign"]:
                    start_loc = (
                        40.014984,
                        -105.270546
                    )
                    stop_loc = (
                        x["lat"][0][0][0][0], 
                        x["lon"][0][0][0][0]
                    )
                    events.append(dict(
                        rise_area=x["rise_area"][0][0][0][0],
                        fall_area=x["fall_area"][0][0][0][0],
                        area=abs(x["rise_area"][0][0][0][0]) + abs(x["fall_area"][0][0][0][0]),
                        peak=x["peak"][0][0][0][0],
                        freq=x["freq"][0][0][0][0]/1e6,
                        node=x["node"][0][0][0][0],
                        call_sign=x["call_sign"][0][0][0],
                        lat=x["lat"][0][0][0][0],
                        lon=x["lon"][0][0][0][0],
                        #sza=calculate_zenith_angle(start_loc, stop_loc, d),
                        sza=calculate_zenith_angle_by_location(stop_loc, d),
                        call=call,
                        flare_peaks_xray_a=flare["peaks"][0,0]["xray_a"][0,0][0,0]*1e6,
                        flare_peaks_xray_b=flare["peaks"][0,0]["xray_b"][0,0][0,0]*1e5,
                        flare_peaks_ESPquad=flare["peaks"][0,0]["ESPquad"][0,0][0,0]*1e3,
                        # Add
                        flare_peaks_xray_a_dI=flare["peak_of_dI"][0,0]["xray_a"][0,0][0,0]*1e6,
                        flare_peaks_xray_b_dI=flare["peak_of_dI"][0,0]["xray_b"][0,0][0,0]*1e5,
                        flare_peaks_ESPquad_dI=flare["peak_of_dI"][0,0]["ESPquad"][0,0][0,0]*1e3,
                    ))
        events = pd.DataFrame.from_records(events)
        events = events.dropna()
        uq = 0.32
        print(events.head())
        import matplotlib as mpl
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(dpi=300, figsize=(12,3), nrows=1, ncols=4, sharey=True)
        params = [
            "sza", "flare_peaks_xray_a", 
            "flare_peaks_xray_b", "flare_peaks_ESPquad"
        ]
        y_param, label = "peak", "Peak"
        i = 0
        for p, ax in zip(params, axs):
            e = events.copy()
            fm = fit_OLS(events[y_param], events[p], [uq])
            x, y = np.array(events[y_param]), np.log10(events[p])
            nmi = utils.compute_normalized_MI(x, y, state=0)
            print(f"{p}-------------",nmi)

            e["fit"], e["ub"], e["lb"] = (
                fm["y_hat"],
                fm[uq]["y_upper"],
                fm[uq]["y_lower"],
            )
            e = e.sort_values(by=p)
            ax.scatter(events[p], events[y_param], s=0.8, color="k", label="Observations")
            ax.plot(e[p], e.fit, "r-", lw=1.2, label="Best Fit")
            ax.plot(e[p], e.ub, "b-", lw=1.2, label=r"1-$\sigma$")
            ax.plot(e[p], e.lb, "b-", lw=1.2)
            ax.text(
                0.9, 1.05, 
                r"$\mathcal{N}$=%.2f"%(nmi),#fm["results"].params[1]*1e2), 
                ha="right", va="center", transform=ax.transAxes
            )
            ax.set_yscale("log")
            if i==0:
                ax.legend(loc=3)
            i+=1
        axs[0].set_ylabel(fr"Doppler/{label}, $D_H$ (Hz)")
        #axs[1,0].set_ylabel(f"Doppler/{label} (Hz)")
        axs[0].set_xlabel("SZA (Deg)")
        axs[2].set_xlabel(r"X-ray [$\lambda_1$] ($\times 10^{-5}$ $Wm^{-2}$)")
        axs[1].set_xlabel(r"X-ray [$\lambda_0$] ($\times 10^{-6}$ $Wm^{-2}$)")
        axs[3].set_xlabel(r"EUV [0.1-7 nm] $\times (10^{-3}$ $Wm^{-2}$)")
        fig.subplots_adjust(wspace=0.15, hspace=0.3)
        fig.savefig(fname, bbox_inches="tight")

        fig, axs = plt.subplots(dpi=300, figsize=(9,3), nrows=1, ncols=3, sharey=True)
        params = [
            "flare_peaks_xray_a_dI", 
            "flare_peaks_xray_b_dI", "flare_peaks_ESPquad_dI"
        ]
        y_param, label = "peak", "Peak"
        i = 0
        for p, ax in zip(params, axs): 
            e = events.copy()
            fm = fit_OLS(events[y_param], np.log10(events[p]), [uq])
            x, y = np.array(events[y_param]), np.log10(events[p])
            nmi = utils.compute_normalized_MI(x, y, state=0)
            print(f"{p}-------------",nmi)
            
            e["fit"], e["ub"], e["lb"] = (
                fm["y_hat"],
                fm[uq]["y_upper"],
                fm[uq]["y_lower"],
            )
            e = e.sort_values(by=p)
            ax.scatter(events[p], events[y_param], s=0.8, color="k", label="Observations")
            ax.plot(e[p], e.fit, "r-", lw=1.2, label="Best Fit")
            ax.plot(e[p], e.ub, "b-", lw=1.2, label=r"1-$\sigma$")
            ax.plot(e[p], e.lb, "b-", lw=1.2)
            ax.text(
                0.9, 1.05, 
                r"$\mathcal{N}$=%.2f"%(nmi), 
                ha="right", va="center", transform=ax.transAxes
            )
            ax.set_yscale("log")
            if i==0:
                ax.legend(loc=3)
            #else: ax.set_xscale("log")
            i+=1
        axs[0].set_ylabel(fr"Doppler/{label}, $D_H$ (Hz)")
        axs[1].set_xlabel(r"$\frac{\partial\Phi_{X}}{\partial t}$ [$\lambda_1$] ($\times 10^{-5}$ $Wm^{-2}$)")
        axs[0].set_xlabel(r"$\frac{\partial\Phi_{X}}{\partial t}$ [$\lambda_0$] ($\times 10^{-6}$ $Wm^{-2}$)")
        axs[2].set_xlabel(r"$\frac{\partial\Phi_{EUV}}{\partial t}$ [0.1-7 nm] ($\times 10^{-3}$ $Wm^{-2}$)")
        fig.subplots_adjust(wspace=0.15, hspace=0.3)
        fig.savefig(fname.replace(".","_dI."), bbox_inches="tight")
        return


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

        from darn import FetchData
        from flare import FlareTS
        from hamsci import HamSci
        from smag import SuperMAG

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
        self.flareTS = FlareTS(self.dates)
        self.GOESplot()
        self.darns = FetchData.fetch(base, self.rads, self.dates)
        self.magObs = SuperMAG(self.base, self.dates, stations=mag_stations)
        self.hamSci = HamSci(self.base, self.dates, None)
        self.GrepMultiplot()
        if stg:
            # self.GenerateRadarFoVPlots()
            # self.GenerateRadarRTIPlots()
            self.stage_analysis()
        return

    def CompileSMJSummaryPlots(
        self, id_scanx, fname_tag="SM-SD.%04d.png", plot_sm=True
    ):
        """
        Create SM/J plots overlaied SD data
        """
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
        GOESPlot(time, xl, xs, fname, vlines, colors, drange=self.dates)
        return

    def GrepMultiplot(self):
        """
        Create Latitude longitude dependent plots in Grape
        """
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

    def GenerateRadarRTIPlots(self):
        """
        Generate RTI summary plots.
        """
        base = self.base + "figures/rti/"
        os.makedirs(base, exist_ok=True)
        for rad in self.rads:
            if hasattr(self.darns[rad], "records") > 0:
                ffd = self.darns[rad].records
                for b in ffd.bmnum.unique():
                    rti = RTI(
                        100,
                        self.dates,
                        fig_title=f"{rad.upper()} / {self.dates[0].strftime('%Y-%m-%d')} / {b}",
                    )
                    ax = rti.addParamPlot(ffd, b, "", cbar=True)
                    rti.add_vlines(ax, [self.event, self.event_start], ["k", "r"])
                    rti.save(base + f"{rad}-{'%02d'%b}.png")
                    rti.close()
        return

    def GenerateRadarFoVPlots(self):
        """
        Generate FoV summary plots.
        """
        base = self.base + "figures/fan/"
        os.makedirs(base, exist_ok=True)
        for rad in self.rads:
            if hasattr(self.darns[rad], "records") > 0:
                dmin = np.rint(self.darns[rad].records.scan_time.iloc[0] / 60.0)
                dN = int(
                    np.rint((self.dates[1] - self.dates[0]).total_seconds() / 60.0)
                )
                dates = [
                    self.dates[0] + dt.timedelta(minutes=i * dmin) for i in range(dN)
                ]
                for d in dates:
                    fan = Fan([rad], d)
                    fan.generate_fov(self.darns)
                    fan.save(base + f"{rad}-{d.strftime('%H-%M')}.png")
                    fan.close()
        return

    def stage_analysis(self):
        """
        Stage dataset for next analysis
        """
        base = "data/stage/{Y}-{m}-{d}-{H}-{M}/".format(
            Y=self.event.year,
            m="%02d" % self.event.month,
            d="%02d" % self.event.day,
            H="%02d" % self.event.hour,
            M="%02d" % self.event.minute,
        )
        os.makedirs(base, exist_ok=True)
        self.stage = dict(darn=[], hamsci=None, flare=None)
        for rad in self.rads:
            self.stage["darn"].append(
                self.darns[rad].extract_stagging_data(self.event_start, self.event_end),
            )
        self.stage["hamsci"] = self.hamSci.extract_stagging_data()
        self.stage["flare"] = self.flareTS.extract_stagging_data()
        self.stage["flare"]["rise_time"] = (
            self.event - self.event_start
        ).total_seconds()
        self.stage["flare"]["fall_time"] = (self.event_end - self.event).total_seconds()
        logger.info(f"file: {base}stage0.mat")
        savemat(base + "stage0.mat", self.stage)
        return


def fork_event_based_mpi(file="config/events.csv"):
    """
    Load all the events from
    events list files and fork Hopper
    to pre-process and store the
    dataset.
    """
    o = pd.read_csv(file, parse_dates=["event", "start", "end", "s_time", "e_time"])
    for i, row in o.iterrows():
        ev = row["event"]
        base = "data/{Y}-{m}-{d}-{H}-{M}/".format(
            Y=ev.year,
            m="%02d" % ev.month,
            d="%02d" % ev.day,
            H="%02d" % ev.hour,
            M="%02d" % ev.minute,
        )
        dates = [row["s_time"], row["e_time"]]
        rads = row["rads"].split("-")
        Hopper(base, dates, rads, ev, row["start"], row["end"])
        break
    return


if __name__ == "__main__":
    fork_event_based_mpi()
