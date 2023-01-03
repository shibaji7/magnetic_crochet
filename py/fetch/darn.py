#!/usr/bin/env python

"""darn.py: module is dedicated to fetch superdarn data."""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import bz2
import datetime as dt
import glob
import os

import numpy as np
import pandas as pd
import pydarnio as pydarn
from loguru import logger


class FetchFitData(object):
    """
    This class uses pyDarn to access local repository for radar observations.
    """

    def __init__(
        self,
        base,
        sDate,
        eDate,
        rad,
        ftype="fitacf",
        files=None,
        regex="/sd-data/{year}/{ftype}/{rad}/{date}.*{ftype}*.bz2",
    ):
        """
        Parameters:
        -----------
        base = Base folder to store data
        sDate = Start datetime of analysis
        eDate = End datetime of analysis
        rad = Radar code
        ftype = SD 'fit' file type [fitacf/fitacf3]
        files = List of files to load the data from (optional)
        regex = Regular expression to locate files
        """
        self.base = base
        os.makedirs(base, exist_ok=True)
        self.rad = rad
        self.sDate = sDate
        self.eDate = eDate
        self.files = files
        self.regex = regex
        self.ftype = ftype
        if (rad is not None) and (sDate is not None) and (eDate is not None):
            self.__createFileList__()
        return

    def __createFileList__(self):
        """
        Create file names from date and radar code
        """
        if self.files is None:
            self.files = []
        reg_ex = self.regex
        days = (self.eDate - self.sDate).days + 2
        for d in range(-1, days):
            e = self.sDate + dt.timedelta(days=d)
            fnames = glob.glob(
                reg_ex.format(
                    year=e.year,
                    rad=self.rad,
                    ftype=self.ftype,
                    date=e.strftime("%Y%m%d"),
                )
            )
            fnames.sort()
            for fname in fnames:
                tm = fname.split(".")[1]
                sc = fname.split(".")[2]
                d0 = dt.datetime.strptime(
                    fname.split(".")[0].split("/")[-1] + tm + sc, "%Y%m%d%H%M%S"
                )
                d1 = d0 + dt.timedelta(hours=2)
                if (self.sDate <= d0) and (d0 <= self.eDate):
                    self.files.append(fname)
                elif d0 <= self.sDate <= d1:
                    self.files.append(fname)
        self.files = list(set(self.files))
        self.files.sort()
        self.fetch_data()
        return

    def fetch_data(
        self,
        scalerParams=[
            "bmnum",
            "noise.sky",
            "tfreq",
            "scan",
            "nrang",
            "intt.sc",
            "intt.us",
            "mppul",
            "nrang",
            "rsep",
            "cp",
            "frang",
            "smsep",
            "lagfr",
            "channel",
        ],
        vectorParams=["v", "w_l", "gflg", "p_l", "slist", "v_e"],
    ):
        """
        Fetch data from file list and return the dataset
        scalerParams = Scaler parameter list to fetch
        vectorParams = Vector parameter list to fetch
        """
        if not self.__loadLocal__():
            records = []
            for f in self.files:
                with bz2.open(f) as fp:
                    fs = fp.read()
                logger.info(f"Read file - {f}")
                reader = pydarn.SDarnRead(fs, True)
                records += reader.read_fitacf()
            self.records = pd.DataFrame()
            self.echoRecords = []
            for rec in records:
                time = dt.datetime(
                    rec["time.yr"],
                    rec["time.mo"],
                    rec["time.dy"],
                    rec["time.hr"],
                    rec["time.mt"],
                    rec["time.sc"],
                    rec["time.us"],
                )
                eRec = {
                    "time": time,
                    "bmnum": rec["bmnum"],
                    "eCount": len(rec["slist"]) if "slist" in rec.keys() else 0,
                }
                o = pd.DataFrame()
                for p in vectorParams:
                    if p in rec.keys():
                        o[p] = rec[p]
                    else:
                        o[p] = [np.nan]
                for p in scalerParams:
                    o[p] = rec[p]
                o["time"] = time
                self.records = pd.concat([self.records, o])
                self.echoRecords.append(eRec)
            self.echoRecords = pd.DataFrame.from_records(self.echoRecords)
            if len(self.echoRecords) > 0:
                self.echoRecords.to_csv(
                    self.base + "%s.eRec.csv" % self.rad,
                    header=True,
                    index=False,
                    float_format="%g",
                )
            if len(self.records) > 0:
                self.records.to_csv(
                    self.base + "%s.DFRec.csv" % self.rad,
                    header=True,
                    index=False,
                    float_format="%g",
                )
        return

    def __loadLocal__(self):
        """
        Load data from local csv files
        """
        isLocal = False
        eRecFile, DFRecFile = (
            self.base + "%s.eRec.csv" % self.rad,
            self.base + "%s.DFRec.csv" % self.rad,
        )
        if os.path.exists(eRecFile) and os.path.exists(DFRecFile):
            isLocal = True
            self.records = pd.read_csv(
                self.base + "%s.DFRec.csv" % self.rad, parse_dates=["time"]
            )
            self.echoRecords = pd.read_csv(
                self.base + "%s.eRec.csv" % self.rad, parse_dates=["time"]
            )
        return isLocal
