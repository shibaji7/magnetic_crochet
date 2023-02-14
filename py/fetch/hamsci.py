#!/usr/bin/env python

"""hamsci.py: module is dedicated to fetch HamSci database."""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = ""
__status__ = "Research"


import datetime as dt
import json
import os
from ftplib import FTP

import numpy as np
import pandas as pd
import pytz
from cryptography.fernet import Fernet
from hamsci_psws import grape1
from loguru import logger


class Conn2Remote(object):
    def __init__(self, host, user, password, port=22, passcode=None):
        self.host = host
        self.user = user
        self.password = password
        self.passcode = passcode
        self.port = port
        self.con = False
        if passcode:
            self.decrypt()
        self.conn()
        return

    def decrypt(self):
        passcode = bytes(self.passcode, encoding="utf8")
        cipher_suite = Fernet(passcode)
        self.user = cipher_suite.decrypt(bytes(self.user, encoding="utf8")).decode(
            "utf-8"
        )
        self.host = cipher_suite.decrypt(bytes(self.host, encoding="utf8")).decode(
            "utf-8"
        )
        self.password = cipher_suite.decrypt(
            bytes(self.password, encoding="utf8")
        ).decode("utf-8")
        return

    def conn(self):
        if not self.con:
            self.ftp = FTP(self.host, self.user, self.password)
            self.con = True
        return

    def close(self):
        if self.con:
            self.ftp.quit()
        return


def encrypt(host, user, password, filename="config/passcode.json"):
    passcode = Fernet.generate_key()
    cipher_suite = Fernet(passcode)
    host = cipher_suite.encrypt(bytes(host, encoding="utf8"))
    user = cipher_suite.encrypt(bytes(user, encoding="utf8"))
    password = cipher_suite.encrypt(bytes(password, encoding="utf8"))
    with open(filename, "w") as f:
        f.write(
            json.dumps(
                {
                    "user": user.decode("utf-8"),
                    "host": host.decode("utf-8"),
                    "password": password.decode("utf-8"),
                    "passcode": passcode.decode("utf-8"),
                },
                sort_keys=True,
                indent=4,
            )
        )
    return


def get_session(filename="config/passcode.json", isclose=False):
    with open(filename, "r") as f:
        obj = json.loads("".join(f.readlines()))
        conn = Conn2Remote(
            obj["host"],
            obj["user"],
            obj["password"],
            passcode=obj["passcode"],
        )
    if isclose:
        conn.close()
    return conn


class HamSci(object):
    """
    This class is help to extract the dataset from HamSci database and plot.
    """

    def __init__(self, base, dates, fList, close=True):
        """
        Parameters:
        -----------
        base: Base location
        fList: Frequency of operation in MHz (list)
        dates: Start and end dates
        close: Close FTP connection
        """
        self.fList = fList
        self.dates = self.parse_dates(dates)
        self.date_range = [
            dates[0].to_pydatetime().replace(tzinfo=pytz.utc),
            dates[1].to_pydatetime().replace(tzinfo=pytz.utc),
        ]
        self.base = base + "hamsci/"
        if not os.path.exists(self.base):
            os.makedirs(self.base)
        logger.info("Loging into remote FTP")
        self.conn = get_session()
        self.fetch_files()
        if close:
            logger.info("System logging out from remote.")
            self.conn.close()
        return

    def parse_dates(self, dates):
        """
        Parsing dates
        """
        da = [
            dates[0].to_pydatetime().replace(minute=0, hour=0, second=0),
            dates[1].to_pydatetime().replace(minute=0, hour=0, second=0),
        ]
        return da

    def fetch_files(self):
        """
        Fetch all the available files on the given date/time range and frequencies.
        Compile and store to one location under other files.
        """
        o = []
        self.conn.ftp.cwd(str(self.dates[0].year))
        files = self.conn.ftp.nlst()
        for file in files:
            if ".csv" in file:
                info = file.split("_")
                date = dt.datetime.strptime(info[0].split("T")[0], "%Y-%m-%d")
                node, frq = info[1], info[-1].replace(".csv", "").replace(
                    "WWV", ""
                ).replace("CHU", "")
                if "p" in frq:
                    frq = frq.replace("p", ".")
                if frq.isnumeric():
                    frq = float(frq)
                    o.append({"node": node, "frq": frq, "fname": file, "date": date})
        o = pd.DataFrame.from_records(o)
        o.date = o.date.apply(lambda x: x.to_pydatetime())
        logger.info(f"Number of files {len(o)}")
        if self.fList:
            o = o.query("frq in @self.fList")
        o = o[(o.date >= self.dates[0]) & (o.date <= self.dates[1])]
        logger.info(f"Number of files after {len(o)}")
        logger.info(f"Start retreiveing Bin")
        for fn in o.fname:
            if not os.path.exists(self.base + fn):
                with open(self.base + fn, "wb") as fp:
                    self.conn.ftp.retrbinary(f"RETR {fn}", fp.write)
        return

    def load_nodes(self, freq):
        """
        Load files using grape1 library
        """
        inv = grape1.DataInventory(data_path=self.base)
        inv.filter(
            freq=freq,
            sTime=self.date_range[0],
            eTime=self.date_range[1],
        )
        gn = grape1.GrapeNodes(
            fpath="config/nodelist.csv", logged_nodes=inv.logged_nodes
        )
        return inv, gn

    def setup_plotting(
        self,
        fname,
        freq=10e6,
        solar_loc=(40.6683, -105.0384),
        color_dct={"ckey": "lon"},
        xkey="UTC",
        events=[
            {"datetime": dt.datetime(2021, 10, 28, 15, 35), "label": "X1 Solar Flare"}
        ],
    ):
        """
        Plot dataset in multipoint plot
        """
        import matplotlib.pyplot as plt
        plt.style.use(["science", "ieee"])
        plt.rcParams.update(
            {
                "figure.figsize": np.array([8, 6]),
                "text.usetex": True,
                "font.family": "sans-serif",
                "font.sans-serif": [
                    "Tahoma",
                    "DejaVu Sans",
                    "Lucida Grande",
                    "Verdana",
                ],
                "font.size": 10,
            }
        )
            
        gds = []
        inv, gn = self.load_nodes(freq)
        node_nrs = inv.get_nodes()
        for node in node_nrs:
            gd = grape1.Grape1Data(
                node,
                freq,
                self.date_range[0],
                self.date_range[1],
                inventory=inv,
                grape_nodes=gn,
                data_path=self.base,
            )
            gd.process_data()
            gds.append(gd)
        mp = grape1.GrapeMultiplot(gds)
        palet = mp.multiplot(
            "filtered",
            color_dct=color_dct,
            xkey=xkey,
            solar_lat=solar_loc[0],
            solar_lon=solar_loc[1],
            events=events,
            plot_GOES=False,
            fig_width=8,
            panel_height=3,
        )
        palet["fig"].savefig(fname, bbox_inches="tight")
        return mp
    
    def extract_stagging_data(
        self, 
        date_range, 
        frqs=[], 
    ):
        """
        Extract observations for only analysis
        """
        for freq in freqs:
            inv, gn = self.load_nodes(freq)
            node_nrs = inv.get_nodes()
        return


if __name__ == "__main__":
    dates = [
        dt.datetime(2021, 10, 28),
        dt.datetime(2021, 10, 29),
    ]
    fList = [10, 5, 2.5]
    base = "data/2021-10-28/"
    HamSci(base, dates, fList)
