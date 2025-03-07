#!/usr/bin/env python

"""plot.py: module is dedicated RTI plotting tools, summary plots."""

__author__ = "Chakraborty, S."
__copyright__ = ""
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"

import datetime as dt

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import utils
from pysolar.solar import get_altitude_fast


def setup(font_size=11, science=True):
    if science:
        #plt.style.use(["science", "ieee"])
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
                "font.size": font_size,
            }
        )
    else:
        mpl.rcParams["font.size"] = font_size
        mpl.rcParams["font.weight"] = "bold"
        mpl.rcParams["axes.labelweight"] = "bold"
        mpl.rcParams["axes.titleweight"] = "bold"
        mpl.rcParams["axes.grid"] = True
        mpl.rcParams["grid.linestyle"] = ":"
        mpl.rcParams["figure.figsize"] = np.array([15, 8])
        mpl.rcParams["axes.xmargin"] = 0
    return


class Stackplots(object):

    def __init__(
        self,
        drange,
        fig_title=None,
        num_subplots=3,
        font_size=14.
    ):
        self.drange = drange
        self.num_subplots = num_subplots
        self._num_subplots_created = 0
        self.fig_title = fig_title
        self.font_size=font_size
        setup(self.font_size)
        self.fig = plt.figure(figsize=(8, 2.5 * num_subplots), dpi=240)
        return

    def lay_vlines(
        self,
        vlines=[], 
        colors=[],
    ):
        all_axes = self.fig.axes
        for ax in all_axes:
            for v, c in zip(vlines, colors):
                ax.axvline(v, color=c, ls="--", lw=0.6)
        return

    def GOESSDOPlot(
        self, time, xl, xs, 
        sdo_time, sdo_euv, 
    ):
        """ """
        ax = self._add_axis()
        ax.set_xlim(self.drange)
        ax.set_ylim(1e-7, 1e-3)
        ax.axhline(1e-6, color="k", ls=":", lw=0.6)
        ax.axhline(1e-5, color="k", ls=":", lw=0.6)
        ax.axhline(1e-4, color="k", ls=":", lw=0.6)
        ax.set_ylabel(
            r"Irradiance(GOES) [$Wm^{-2}$]", fontdict={"size": self.font_size, "fontweight": "bold"}
        )
        ax.semilogy(
            time, xl, color="r", ls="-", lw=1, label=r"$\lambda_1=0.1-0.8 nm$"
        )
        ax.semilogy(
            time, xs, color="b", ls="-", lw=1, label=r"$\lambda_0=0.05-0.4 nm$"
        )
        ax.set_xticklabels([])
        ax.legend(loc=1)
        ax = ax.twinx()
        ax.set_xlabel("Time [UT]", fontdict={"size": self.font_size, "fontweight": "bold"})
        ax.set_xlim(self.drange)
        ax.set_ylabel(
            "Irradiance (SDO, 0.1-7 nm)\n"+ r"[$\times 10^{-3}$ $Wm^{-2}$]", fontdict={"size": self.font_size, "fontweight": "bold"}
        )
        ax.plot(
            sdo_time, sdo_euv*1e3, color="k", ls="-", lw=1,
        )
        return

    def addParamPlot(
        self,
        df,
        beam,
        title="",
        p_max=100,
        p_min=-100,
        xlabel="Time [UT]",
        zparam="v",
        label="Velocity [m/s]",
        yscale="srange",
        cmap=plt.cm.Spectral,
        cbar=True,
        nGates=80,
    ):
        if yscale == "srange":
            yrange, ylab = (
                nGates * df.rsep.tolist()[0] + df.frang.tolist()[0],
                "Slant Range [km]",
            )
        else:
            yrange, ylab = (nGates, "Range Gates")
        ax = self._add_axis()
        df = df[df.bmnum == beam]
        X, Y, Z = utils.get_gridded_parameters(
            df, xparam="time", yparam=yscale, zparam=zparam, 
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("$%H^{%M}$"))
        if (self.drange[1] - self.drange[0]).total_seconds() / 3600 <= 1.0:
            major_locator = mdates.MinuteLocator(byminute=range(0, 60, 10))
        elif (self.drange[1] - self.drange[0]).total_seconds() / 3600 <= 6.0:
            major_locator = mdates.HourLocator(byhour=range(0, 24, 1))
        else:
            major_locator = mdates.HourLocator(byhour=range(0, 24, 4))
        ax.xaxis.set_major_locator(major_locator)
        ax.set_xlabel(xlabel, fontdict={"size": self.font_size, "fontweight": "bold"})
        ax.set_xlim([mdates.date2num(self.drange[0]), mdates.date2num(self.drange[1])])
        ax.set_ylim(0, yrange)
        ax.set_ylabel(ylab, fontdict={"size": self.font_size, "fontweight": "bold"})
        im = ax.pcolormesh(X, Y, Z.T, lw=0.01, edgecolors="None", cmap=cmap,
                        vmax=p_max, vmin=p_min, shading="nearest", zorder=3)
        if cbar:
            self._add_colorbar(self.fig, ax, im, label=label)
        if title:
            ax.set_title(title, loc="left", fontdict={"fontweight": "bold"})
        ax.set_xticklabels([])
        return ax

    def HamSciTS(
        self,
        gds,
        color_dct={"ckey": "lon"},
        title="",
    ):
        """ """
        ax = self._add_axis()
        cmap = color_dct.get("cmap", "viridis")
        vmax, vmin = (color_dct.get("vmax", -70), color_dct.get("vmin", -110))
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        mpbl = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        
        ax.set_xlabel("Time [UT]", fontdict={"size": self.font_size, "fontweight": "bold"})
        ax.set_ylabel(r"Power [dB]", fontdict={"size": self.font_size, "fontweight": "bold"})

        for gs in [gds[3],gds[5],gds[9]]:
            o = gs.data["filtered"]["df"]
            print(o.head())
            # o = gs.data["raw"]["df"]
            val = gs.meta["solar_lon"]
            # ax.plot(o.UTC, 20*np.log10(o.Vpk), ls="-", lw=0.8, color=mpbl.cmap(mpbl.norm(val)))
            ax.plot(o.UTC, o.Power_dB, ls="-", lw=0.8, color=mpbl.cmap(mpbl.norm(val)))
        ax.set_xlim(self.drange)
        ax.set_ylim(-100, -20)
        if title:
            ax.set_title(title, loc="left", fontdict={"fontweight": "bold"})
        # self.fig.colorbar(mpbl, ax=ax, label="Longitude")
        return

    def _add_axis(self):
        tag = chr(97+self._num_subplots_created)
        self._num_subplots_created += 1
        ax = self.fig.add_subplot(self.num_subplots, 1, self._num_subplots_created)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("$%H^{%M}$"))
        major_locator = mdates.HourLocator(byhour=range(0, 24, 1))
        ax.xaxis.set_major_locator(major_locator)
        ax.text(
            0.05, 0.95, f"({tag})", ha="left", 
            va="center", transform=ax.transAxes,
            fontdict={"size": self.font_size, "fontweight": "bold"}
        )
        return ax

    def _add_colorbar(self, fig, ax, im, label=""):
        """
        Add a colorbar to the right of an axis.
        """
        cpos = [1.04, 0.1, 0.025, 0.8]
        cax = ax.inset_axes(cpos, transform=ax.transAxes)
        cb = fig.colorbar(im, ax=ax, cax=cax)
        cb.set_label(label)
        return

    def save(self, filepath):
        self.fig.savefig(filepath, bbox_inches="tight", facecolor=(1, 1, 1, 1))
        return

    def close(self):
        self.fig.clf()
        plt.close()
        return