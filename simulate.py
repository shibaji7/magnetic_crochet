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
from analysis import Stats
from hopper import Hopper
import argparse
import datetime as dt
import pandas as pd


def run_event(idx, file="config/events.csv"):
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
    row = o.iloc[idx]
    ev = row["event"]
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
    return

def run_hamsci_stats(args, hamsci_event_list="config/hamsci_events.csv"):
    """
    Run HamSCI statistics from Stage:0
    """
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--index", default=0, type=int, help="Index (0)"
    )
    parser.add_argument(
        "-cf", "--cfg_file", default="config/events.csv", type=str, help="Configuration file"
    )
    parser.add_argument(
        "-m", "--method", default="EA", type=str, help="Methods to run (EA:Event Analysis/HS: HamSCI Stats)"
    )
    args = parser.parse_args()
    for k in vars(args).keys():
        print("     ", k, "->", str(vars(args)[k]))
    if args.method == "EA":
        run_event(args.index, args.cfg_file)
    elif args.method == "HS":
        run_hamsci_stats(args)
    else:
        print(f"Invalid method / not implemented {args.method}")