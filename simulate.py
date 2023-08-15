#!/usr/bin/env python

"""simulate.py: module is dedicated to fetch, filter, and save data for post_processing."""

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
from analysis import Hopper
import argparse
import pandas as pd


def run_event(idx, file="config/events.csv"):
    """
    Load all the events from
    events list files and fork Hopper
    to pre-process and store the
    dataset.
    """
    o = pd.read_csv(file, parse_dates=["event", "start", "end", "s_time", "e_time"])
    row = o.iloc[idx]
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
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--index", default=0, type=int, help="Index (0)"
    )
    parser.add_argument(
        "-cf", "--cfg_file", default="config/events.csv", type=str, help="Configuration file"
    )
    args = parser.parse_args()
    for k in vars(args).keys():
        print("     ", k, "->", str(vars(args)[k]))
    run_event(args.index, args.cfg_file)