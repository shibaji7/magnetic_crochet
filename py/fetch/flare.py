#!/usr/bin/env python

"""flare.py: module is dedicated to fetch solar irradiance data."""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"


import numpy as np
import pandas as pd
from loguru import logger
from sunpy import timeseries as ts
from sunpy.net import Fido
from sunpy.net import attrs as a


class FlareInfo(object):
    """
    This class is dedicated to plot GOES
    from the repo using SunPy
    """

    def __init__(self, dates, fl_class="C1.0"):
        """
        Parameters
        ----------
        dates: list of datetime object for start and end of TS
        """
        self.flare = pd.DataFrame()
        self.dates = dates
        result = Fido.search(
            a.Time(
                self.dates[0].strftime("%Y-%m-%d %H:%M"),
                self.dates[1].strftime("%Y-%m-%d %H:%M"),
            ),
            a.hek.FL,
            a.hek.FL.GOESCls > fl_class,
            a.hek.OBS.Observatory == "GOES",
        )
        # Retrieve HEKTable from the Fido result and then load
        hek_results = result["hek"]
        if len(hek_results) > 0:
            self.flare = hek_results[
                "event_starttime",
                "event_peaktime",
                "event_endtime",
                "fl_goescls",
            ].to_pandas()
        return

class FlareTS(object):
    """
    This class is dedicated to plot GOES, RHESSI, and SDO data
    from the repo using SunPy
    """

    def __init__(self, dates):
        """
        Parameters
        ----------
        dates: list of datetime object for start and end of TS
        """
        self.dates = dates
        self.dfs = {}
        self.__loadGOES__()
        self.__loadEVE__()
        return

    def __loadGOES__(self):
        """
        Load GOES data from remote/local repository
        """
        self.dfs["goes"], self.goes, self.flareHEK = pd.DataFrame(), [], None
        result = Fido.search(
            a.Time(
                self.dates[0].strftime("%Y-%m-%d %H:%M"),
                self.dates[1].strftime("%Y-%m-%d %H:%M"),
            ),
            a.Instrument("XRS") | a.hek.FL & (a.hek.FRM.Name == "SWPC"),
        )
        if len(result) > 0:
            logger.info(f"Fetching GOES ...")
            tmpfiles = Fido.fetch(result)
            for tf in tmpfiles:
                self.goes.append(ts.TimeSeries(tf))
                self.dfs["goes"] = pd.concat(
                    [self.dfs["goes"], self.goes[-1].to_dataframe()]
                )
            self.dfs["goes"].index.name = "time"
            self.dfs["goes"] = self.dfs["goes"].reset_index()
            self.dfs["goes"] = self.dfs["goes"][
                (self.dfs["goes"].time >= self.dates[0])
                & (self.dfs["goes"].time <= self.dates[1])
            ]
            # Retrieve HEKTable from the Fido result and then load
            hek_results = result["hek"]
            if len(hek_results) > 0:
                self.flare = hek_results[
                    "event_starttime",
                    "event_peaktime",
                    "event_endtime",
                    "fl_goescls",
                    "ar_noaanum",
                ].to_pandas()
                self.flare["cls"] = self.flare.fl_goescls.apply(lambda x: x[0])
        return

    def __loadEVE__(self):
        """
        Load EVE data from remote/local repository
        """
        self.dfs["eve"], self.eve = pd.DataFrame(), []
        result = Fido.search(
            a.Time(
                self.dates[0].strftime("%Y-%m-%d %H:%M"),
                self.dates[1].strftime("%Y-%m-%d %H:%M"),
            ),
            a.Instrument("EVE"),
        )
        if len(result) > 0:
            logger.info(f"Fetching EVE ...")
            tmpfiles = Fido.fetch(result)
            for tf in tmpfiles:
                if "EVE" in tf:
                    self.eve.append(ts.TimeSeries(tf, source="EVE"))
                    self.dfs["eve"] = pd.concat(
                        [self.dfs["eve"], self.eve[-1].to_dataframe()]
                    )
        self.dfs["eve"].index.name = "time"
        self.dfs["eve"] = self.dfs["eve"].reset_index()
        if len(self.dfs["eve"]) > 10:
            self.dfs["eve"] = self.dfs["eve"][
                (self.dfs["eve"].time >= self.dates[0])
                & (self.dfs["eve"].time <= self.dates[1])
            ]
        return

    def extract_stagging_data(self):
        """
        Stagging dataset
        """
        etc = dict(
            rise_time=np.nan,
            fall_time=np.nan,
            peak_of_dI=dict(
                xray_a=np.float64(np.diff(self.dfs["goes"].xrsa).max()), # Store derivative
                xray_b=np.float64(np.diff(self.dfs["goes"].xrsb).max()),
                ESPquad=np.float64(np.diff(self.dfs["eve"]["0.1-7ESPquad"]).max()) if len(self.dfs["eve"]) > 0 else np.nan,
            ),
            peaks=dict(
                xray_a=np.float64(self.dfs["goes"].xrsa.max()),
                xray_b=np.float64(self.dfs["goes"].xrsb.max()),
                ESPquad=np.float64(self.dfs["eve"]["0.1-7ESPquad"].max()) if len(self.dfs["eve"]) > 0 else np.nan,
                ESP171=np.float64(self.dfs["eve"]["17.1ESP"].max()) if len(self.dfs["eve"]) > 0 else np.nan,
                ESP257=np.float64(self.dfs["eve"]["25.7ESP"].max()) if len(self.dfs["eve"]) > 0 else np.nan,
                ESP304=np.float64(self.dfs["eve"]["30.4ESP"].max()) if len(self.dfs["eve"]) > 0 else np.nan,
                ESP366=np.float64(self.dfs["eve"]["36.6ESP"].max()) if len(self.dfs["eve"]) > 0 else np.nan,
            ),
            energy=dict(
                xray_a=np.float64(np.trapz(self.dfs["goes"].xrsa.fillna(0), dx=60)),
                xray_b=np.float64(np.trapz(self.dfs["goes"].xrsb.fillna(0), dx=60)),
                ESPquad=np.float64(np.trapz(self.dfs["eve"]["0.1-7ESPquad"].fillna(0), dx=60)) if len(self.dfs["eve"]) > 0 else np.nan,
                ESP171=np.float64(np.trapz(self.dfs["eve"]["17.1ESP"].fillna(0), dx=60)) if len(self.dfs["eve"]) > 0 else np.nan,
                ESP257=np.float64(np.trapz(self.dfs["eve"]["25.7ESP"].fillna(0), dx=60)) if len(self.dfs["eve"]) > 0 else np.nan,
                ESP304=np.float64(np.trapz(self.dfs["eve"]["30.4ESP"].fillna(0), dx=60)) if len(self.dfs["eve"]) > 0 else np.nan,
                ESP366=np.float64(np.trapz(self.dfs["eve"]["36.6ESP"].fillna(0), dx=60)) if len(self.dfs["eve"]) > 0 else np.nan,
            ),
        )
        return etc

    def __loadRHESSI__(self):
        """
        Load RHESSI data from remote/local repository
        """
        self.rhessi, self.dfs["rhessi"] = [], pd.DataFrame()
        result = Fido.search(
            a.Time(self.dates[0], self.dates[1]), a.Instrument("RHESSI")
        )
        if len(result) > 0:
            logger.info(f"Fetched RHESSI: \n {result}")
            tmpfiles = Fido.fetch(result)
            for tf in tmpfiles:
                if "obssum" in tf:
                    self.rhessi.append(ts.TimeSeries(tf))
                    self.dfs["rhessi"] = pd.concat(
                        [self.dfs["rhessi"], self.rhessi[-1].to_dataframe()]
                    )
            self.dfs["rhessi"].index.name = "time"
            self.dfs["rhessi"] = self.dfs["rhessi"].reset_index()
        logger.info(f"Data from RHESSI XRS: \n {self.dfs['rhessi'].head()}")
        return
