#!/usr/bin/env python

"""zenodo.py: module is dedicated to upload files to Zenodo and create a dataset."""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import glob
import json

import requests


def fetch_API_key():
    """
    Get the ACCESS_TOKEN
    """
    with open("~/.zenodo_token", "r") as f:
        zenodo = json.loads(f.readlines().join("\n"))
    ACCESS_TOKEN = zenodo["ACCESS_TOKEN"]
    return ACCESS_TOKEN

class Zenodo(object):
    """
    This class interacts with zenondo using REST API.
    1. Fetch existing project
    2. Upload dataset into project
    """

    def __init__(self, ev):
        """
        param: Name of the parameter file
        """
        self.event = ev
        base = "data/{Y}-{m}-{d}-{H}-{M}/".format(
            Y=ev.year,
            m="%02d" % ev.month,
            d="%02d" % ev.day,
            H="%02d" % ev.hour,
            M="%02d" % ev.minute,
        )
        self.dataset = f"data/{base}/"
        self.ACCESS_TOKEN = fetch_API_key()
        self.__setup__()
        self.upload_files()
        return

    def __setup__(self):
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.ACCESS_TOKEN}",
        }
        self.baseurl = f"https://zenodo.org/api"
        # TODO get project details by name
        return
    
    def upload_files(self):
        """
        Upload data and files
        """
        params = {"access_token": self.ACCESS_TOKEN}
        # TODO get all files under that dataset directory
        # TODO get all files stored
        # TODO upload only non esisting files
        files = glob.glob(self.dataset + "/*")
        for f in files:
            with open(f, "rb") as fp:
                filename = f.split("/")[-1]
                r = requests.put(
                    f"{self.bucket_link}/{filename}",
                    params=params,
                    data=fp,
                )
        return
