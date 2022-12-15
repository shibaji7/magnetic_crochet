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


import datetime

import pytz
from cryptography.fernet import Fernet
#import paramiko
import json
from hamsci_psws import grape1


class Conn2Remote(object):
    def __init__(self, host, user, password, key_filename, port=22, passcode=None):
        self.host = host
        self.user = user
        self.key_filename = key_filename
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
        return

    def conn(self):
        if not self.con:
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect(
                hostname=self.host,
                port=self.port,
                username=self.user,
                key_filename=self.key_filename,
            )
            self.scp = paramiko.SFTPClient.from_transport(self.ssh.get_transport())
            self.con = True
        return

    def close(self):
        if self.con:
            self.scp.close()
            self.ssh.close()
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


class HamSci(object):
    """
    This class is help to extract the dataset from HamSci database and plot.
    """

    def __init__(self, f0, dates):
        """
        Parameters:
        -----------
        f0: Frequency of operation in Hz
        dates: Start and end dates
        """
        self.f0 = f0
        self.dates = dates

        self.inventory = grape1.DataInventory(data_path="scripts/data/")
        self.inventory.filter(freq=self.f0, sTime=self.dates[0], eTime=self.dates[1])
        self.grape_nodes = grape1.GrapeNodes(
            fpath="scripts/nodelist.csv", logged_nodes=self.inventory.logged_nodes
        )
        return

    def summary_plots(self):
        return


if __name__ == "__main__":
    dates = [
        datetime.datetime(2021, 10, 28, 0, tzinfo=pytz.UTC),
        datetime.datetime(2021, 10, 29, 0, tzinfo=pytz.UTC),
    ]
    f0 = 10e6
    HamSci(f0, dates)
    
