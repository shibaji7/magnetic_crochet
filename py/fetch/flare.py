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


import pandas as pd
from loguru import logger
from sunpy import timeseries as ts
from sunpy.net import Fido
from sunpy.net import attrs as a


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
        # self.__loadRHESSI__()
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
            # Retrieve HEKTable from the Fido result and then load
            self.flareHEK = result["hek"]
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
        return

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
