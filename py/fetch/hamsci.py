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
import numpy as np

from hamsci_psws import grape1


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
        self.inventory.filter(freq=self.f0,sTime=self.dates[0],eTime=self.dates[1])
        self.grape_nodes = grape1.GrapeNodes(fpath="scripts/nodelist.csv", logged_nodes=self.inventory.logged_nodes)
        return
    
    def summary_plots(self):
        return
    
    
if __name__ == "__main__":
    dates = [
        datetime.datetime(2021,10,28,0, tzinfo=pytz.UTC),
        datetime.datetime(2021,10,29,0, tzinfo=pytz.UTC)
    ]
    f0 = 10e6
    HamSci(f0, dates)