#!/usr/bin/env python

"""
    plotFoV.py: module to plot Fan plots with various transformation
"""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import cartopy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER


class Fan(object):
    """
    This class holds plots for all radars FoVs
    """

    def __init__(
        self, rads, date, fig_title=None, nrows=1, ncols=1, coord="geo", science=True
    ):
        if science:
            #plt.style.use(["science", "ieee"])
            plt.rcParams.update(
                {
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
        else:
            mpl.rcParams["font.size"] = 16
            mpl.rcParams["font.weight"] = "bold"
            mpl.rcParams["axes.labelweight"] = "bold"
            mpl.rcParams["axes.titleweight"] = "bold"
            mpl.rcParams["axes.grid"] = True
            mpl.rcParams["grid.linestyle"] = ":"
            mpl.rcParams["figure.figsize"] = np.array([15, 8])
            mpl.rcParams["axes.xmargin"] = 0

        self.rads = rads
        self.date = date
        self.nrows, self.ncols = nrows, ncols
        self._num_subplots_created = 0
        self.fig = plt.figure(figsize=(4.5 * ncols, 4.5 * nrows), dpi=240)
        self.coord = coord
        plt.suptitle(
            f"{self.date_string()} / {fig_title}"
            if fig_title
            else f"{self.date_string()}",
            x=0.1,
            y=0.85,
            ha="left",
            fontweight="bold",
            fontsize=8,
        )
        return

    def add_axes(self):
        """
        Instatitate figure and axes labels
        """
        from carto import SDCarto
        self._num_subplots_created += 1
        proj = cartopy.crs.Stereographic(central_longitude=-90.0, central_latitude=45.0)
        ax = self.fig.add_subplot(
            100 * self.nrows + 10 * self.ncols + self._num_subplots_created,
            projection="SDCarto",
            map_projection=proj,
            coords=self.coord,
            plot_date=self.date,
        )
        ax.overaly_coast_lakes(lw=0.4, alpha=0.4)
        ax.set_extent([-130, -50, 20, 80], crs=cartopy.crs.PlateCarree())
        plt_lons = np.arange(-180, 181, 15)
        mark_lons = np.arange(-180, 181, 30)
        plt_lats = np.arange(40, 90, 10)
        gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), linewidth=0.5)
        gl.xlocator = mticker.FixedLocator(plt_lons)
        gl.ylocator = mticker.FixedLocator(plt_lats)
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.n_steps = 90
        #ax.mark_latitudes(plt_lats, fontsize="small", color="darkred")
        #ax.mark_longitudes(mark_lons, fontsize="small", color="darkblue")
        self.proj = proj
        self.geo = cartopy.crs.PlateCarree()
        ax.text(
            -0.02,
            0.99,
            "Coord: Geo",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize="small",
            rotation=90,
        )
        ax.draw_DN_terminator(self.date)
        gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), linewidth=0.8, color="r")
        gl.xlocator = mticker.FixedLocator([-72.2])
        gl.ylocator = mticker.FixedLocator([])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), linewidth=0.8, color="orange")
        gl.xlocator = mticker.FixedLocator([-102])
        gl.ylocator = mticker.FixedLocator([])
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        return ax

    def date_string(self, label_style="web"):
        # Set the date and time formats
        dfmt = "%d/%b/%Y" if label_style == "web" else "%d %b %Y,"
        tfmt = "%H:%M"
        stime = self.date
        date_str = "{:{dd} {tt}} UT".format(stime, dd=dfmt, tt=tfmt)
        return date_str

    def generate_fov(self, stations, beams=[]):
        """
        Generate plot with dataset overlaid
        """
        ax = self.add_axes()
        for k in stations.keys():
            if k=="WWV":
                ax.overlay_station(
                    stations[k], markerColor="darkred", 
                    zorder=4, markerSize=10, drawline=False
                )
            else:
                ax.overlay_station(stations[k], markerSize=5)
        for rad in self.rads:
            ax.overlay_radar(rad, font_color="b")
            ax.overlay_fov(rad)
            #ax.overlay_data(rad, fds[rad].records, self.proj)
            if beams and len(beams) > 0:
                [ax.overlay_fov(rad, beamLimits=[b, b + 1], ls="--") for b in beams]
        return

    def save(self, filepath):
        self.fig.savefig(filepath, bbox_inches="tight", facecolor=(1, 1, 1, 1))
        return

    def close(self):
        self.fig.clf()
        plt.close()
        return
