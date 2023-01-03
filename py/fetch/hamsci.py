#!/usr/bin/env python

"""hamsci.py: module is dedicated to fetch HamSci database."""

__author__ = "Collins, K."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Collins, K."
__email__ = "TODO"
__status__ = "Research"


import datetime as dt
import json
import os
from ftplib import FTP

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

    def __init__(self, fList, dates, base, close=True):
        """
        Parameters:
        -----------
        fList: Frequency of operation in MHz (list)
        dates: Start and end dates
        base: Base location
        close: Close FTP connection
        """
        self.fList = fList
        self.dates = dates
        self.base = base
        if not os.path.exists(base):
            os.makedirs(base)
        logger.info("Loging into remote FTP")
        self.conn = get_session()
        self.fetch_files()
        self.load_files()
        if close:
            logger.info("System logging out from remote.")
            self.conn.close()
        return

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

    def load_files(self):
        """
        Load files using grape1 library
        """
        self.inventory = grape1.DataInventory(data_path=self.base)
        self.inventory.filter(
            sTime=self.dates[0].astimezone(pytz.utc),
            eTime=self.dates[1].astimezone(pytz.utc),
        )
        self.grape_nodes = grape1.GrapeNodes(
            fpath="config/nodelist.csv", logged_nodes=self.inventory.logged_nodes
        )
        return


if __name__ == "__main__":
    dates = [
        dt.datetime(2021, 10, 28),
        dt.datetime(2021, 10, 29),
    ]
    fList = [10, 5, 2.5]
    base = "data/2021-10-28/"
    HamSci(fList, dates, base)
