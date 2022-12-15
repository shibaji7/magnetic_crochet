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

import os
import sys

import pandas as pd

sys.path.extend(["py/", "py/fetch/"])


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
        uid="shibaji7",
        mag_stations=None,
    ):
        """
        Populate all data tables from GOES, FISM2, SuperDARN, and SuperMAG.
        """

        from darn import FetchFitData
        from flare import FlareTS
        from smag import SuperMAG

        self.base = base
        self.dates = dates
        self.rads = rads
        self.uid = uid
        self.mag_stations = mag_stations

        if not os.path.exists(base):
            os.makedirs(base)
        self.flareTS = FlareTS(self.dates)
        self.darns = {}
        for rad in self.rads:
            self.darns[rad] = FetchFitData(
                self.base,
                self.dates[0],
                self.dates[1],
                rad,
            )
        self.magObs = SuperMAG(self.base, self.dates, stations=mag_stations)

        self.GenerateSummaryPlots()
        return

    def CompileSMJSummaryPlots(
        self, id_scanx, fname_tag="SM-SD.%04d.png", plot_sm=True
    ):
        """
        Create SM/J plots overlaied SD data
        """

        return

    def GenerateSummaryPlots(self):
        """
        Generate Fan plots and other summary plots.
        """
        return


def fork_event_based_mpi(file="config/events.csv"):
    """
    Load all the events from
    events list files and fork Hopper
    to pre-process and store the
    dataset.
    """
    o = pd.read_csv(file, parse_dates=["event", "s_time", "e_time"])
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
        Hopper(base, dates, rads)
    return


if __name__ == "__main__":
    fork_event_based_mpi()
