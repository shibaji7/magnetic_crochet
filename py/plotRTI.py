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


def setup(science=True):
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
    return


class HamSciTS(object):
    """
    Create plots for frequency and absorption etc.
    """

    def __init__(
        self,
        gds,
        drange,
        fig_title=None,
        num_subplots=2,
        science=True,
    ):
        setup(science)
        self.gds = gds
        self.num_subplots = num_subplots
        self.fig = plt.figure(figsize=(6, 3 * num_subplots), dpi=180)
        self.axes = [
            self.fig.add_subplot(100 * num_subplots + 10 + i + 1)
            for i in range(self.num_subplots)
        ]
        # for ax in self.axes:

        if fig_title:
            plt.suptitle(
                fig_title, x=0.075, y=0.99, ha="left", fontweight="bold", fontsize=12
            )
        return

    def _add_axis(self, i, xlabel, ylabel):
        ax = self.axes[i]
        ax.xaxis.set_major_formatter(mdates.DateFormatter("$%H^{%M}$"))
        if (self.drange[1] - self.drange[0]).total_seconds() / 3600 <= 1.0:
            major_locator = mdates.MinuteLocator(byminute=range(0, 60, 10))
        elif (self.drange[1] - self.drange[0]).total_seconds() / 3600 <= 4.0:
            major_locator = mdates.MinuteLocator(byminute=range(0, 60, 30))
        else:
            major_locator = mdates.HourLocator(byhour=range(0, 24, 4))
        ax.xaxis.set_major_locator(major_locator)
        ax.set_xlabel(xlabel, fontdict={"size": 11, "fontweight": "bold"})
        ax.set_ylabel(ylabel, fontdict={"size": 11, "fontweight": "bold"})
        return ax

    def addParamPlot(
        self,
        pname,
        ylim,
        i=0,
        xlabel="Time [UT]",
        ylabel="Frequency [Hz]",
        cmap=plt.cm.jet_r,
    ):
        ax = self._add_axis(i, xlabel, ylabel)
        ax.set_ylim(ylim)
        ax.set_xlim(self.drange)
        return

    def save(self, filepath):
        self.fig.savefig(filepath, bbox_inches="tight", facecolor=(1, 1, 1, 1))
        return

    def close(self):
        self.fig.clf()
        plt.close()
        return


class RTI(object):
    """
    Create plots for velocity, width, power, elevation angle, etc.
    """

    def __init__(
        self,
        nGates,
        drange,
        fig_title=None,
        num_subplots=1,
        angle_th=100.0,
        science=True,
    ):
        setup(science)
        self.nGates = nGates
        self.drange = drange
        self.num_subplots = num_subplots
        self._num_subplots_created = 0
        self.fig = plt.figure(figsize=(6, 3 * num_subplots), dpi=180)
        if fig_title:
            plt.suptitle(
                fig_title, x=0.075, y=0.92, ha="left", fontweight="bold", fontsize=12
            )
        self.angle_th = angle_th
        return

    def addParamPlot(
        self,
        df,
        beam,
        title,
        p_max=100,
        p_min=-100,
        xlabel="Time [UT]",
        zparam="v",
        label="Velocity [m/s]",
        yscale="srange",
        cmap=plt.cm.Spectral,
        cbar=False,
        fov=None,
        ax=None,
    ):
        if yscale == "srange":
            yrange, ylab = (
                self.nGates * df.rsep.tolist()[0] + df.frang.tolist()[0],
                "Slant Range [km]",
            )
        else:
            yrange, ylab = (self.nGates, "Range Gates")
        ax = ax if ax else self._add_axis()
        df = df[df.bmnum == beam]
        X, Y, Z = utils.get_gridded_parameters(
            df, xparam="time", yparam=yscale, zparam=zparam, rounding=False
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("$%H^{%M}$"))
        if (self.drange[1] - self.drange[0]).total_seconds() / 3600 <= 1.0:
            major_locator = mdates.MinuteLocator(byminute=range(0, 60, 10))
        elif (self.drange[1] - self.drange[0]).total_seconds() / 3600 <= 6.0:
            major_locator = mdates.HourLocator(byhour=range(0, 24, 1))
        else:
            major_locator = mdates.HourLocator(byhour=range(0, 24, 4))
        ax.xaxis.set_major_locator(major_locator)
        ax.set_xlabel(xlabel, fontdict={"size": 11, "fontweight": "bold"})
        ax.set_xlim([mdates.date2num(self.drange[0]), mdates.date2num(self.drange[1])])
        ax.set_ylim(0, yrange)
        ax.set_ylabel(ylab, fontdict={"size": 11, "fontweight": "bold"})
        im = ax.pcolormesh(
            X,
            Y,
            Z.T,
            lw=0.01,
            edgecolors="None",
            cmap=cmap,
            snap=True,
            vmax=p_max,
            vmin=p_min,
            shading="auto",
        )
        if cbar:
            self._add_colorbar(self.fig, ax, im, label=label)
        if title:
            ax.set_title(title, loc="left", fontdict={"fontweight": "bold"})
        if fov:
            self.overlay_sza(
                fov,
                ax,
                df.time.unique(),
                beam,
                [0, self.nGates],
                df.rsep.iloc[0],
                df.frang.iloc[0],
                yscale,
            )
        return ax

    def add_vlines(self, ax, vlines, colors):
        """
        Adding vertical lines
        """
        for vline, color in zip(vlines, colors):
            ax.axvline(vline, color=color, ls="--", lw=0.6)
        return

    def overlay_sza(self, fov, ax, times, beam, gate_range, rsep, frang, yscale):
        """
        Add terminator to the radar
        """
        R = 6378.1
        gates = np.arange(gate_range[0], gate_range[1])
        dn_grid = np.zeros((len(times), len(gates)))
        for i, d in enumerate(times):
            d = dt.datetime.utcfromtimestamp(d.astype(dt.datetime) * 1e-9).replace(
                tzinfo=dt.timezone.utc
            )
            for j, g in enumerate(gates):
                gdlat, glong = fov[0][g, beam], fov[1][g, beam]
                angle = 90.0 - get_altitude_fast(gdlat, glong, d)
                dn_grid[i, j] = angle
        terminator = np.zeros_like(dn_grid)
        terminator[dn_grid > self.angle_th] = 1.0
        terminator[dn_grid <= self.angle_th] = 0.0
        if yscale == "srange":
            gates = frang + (rsep * gates)
        elif yscale == "virtual_height":
            mvh.standard_vhm(self.nGates * df.rsep.tolist()[0] + df.frang.tolist()[0])
        else:
            # TODO
            pass
        times, gates = np.meshgrid(times, gates)
        ax.pcolormesh(
            times.T,
            gates.T,
            terminator,
            lw=0.01,
            edgecolors="None",
            cmap="gray_r",
            vmax=2,
            vmin=0,
            shading="nearest",
            alpha=0.3,
        )
        return

    def _add_axis(self):
        self._num_subplots_created += 1
        ax = self.fig.add_subplot(self.num_subplots, 1, self._num_subplots_created)
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


def GOESPlot(time, xl, xs, filepath, vlines=[], colors=[], drange=[]):
    """ """
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
            "font.size": 10,
        }
    )
    drange = drange if len(drange) == 2 else [time[0], time[-1]]
    fig = plt.figure(figsize=(6, 3), dpi=180)
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("$%H^{%M}$"))
    major_locator = mdates.MinuteLocator(byminute=range(0, 60, 10))
    ax.xaxis.set_major_locator(major_locator)
    ax.set_xlabel("Time [UT]", fontdict={"size": 11, "fontweight": "bold"})
    ax.set_xlim(drange)
    ax.set_ylim(1e-8, 1e-3)
    for v, c in zip(vlines, colors):
        ax.axvline(v, color=c, ls="--", lw=0.6)
    ax.axhline(1e-5, color="orange", ls=":", lw=0.6)
    ax.axhline(1e-4, color="r", ls=":", lw=0.6)
    ax.set_ylabel(
        r"Irradiance [$Wm^{-2}$]", fontdict={"size": 11, "fontweight": "bold"}
    )
    ax.semilogy(
        time, xl, color="r", ls="-", lw=1, label=r"$\lambda=0.1-0.8 nm$"
    )
    ax.semilogy(
        time, xs, color="b", ls="-", lw=1, label=r"$\lambda=0.05-0.4 nm$"
    )
    ax.legend(loc=2)
    fig.savefig(filepath, bbox_inches="tight", facecolor=(1, 1, 1, 1))
    return


def HamSciTS(
    gds,
    filepath,
    vlines=[],
    colors=[],
    solar_loc=(40.6683, -105.0384),
    color_dct={"ckey": "lon"},
    xkey="UTC",
    events=[{"datetime": dt.datetime(2021, 10, 28, 15, 35), "label": "X1 Solar Flare"}],
    params=["Freq", "Power_dB"],
    drange=[],
):
    """ """
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
            "font.size": 10,
        }
    )
    fig = plt.figure(figsize=(8, 6), dpi=200)
    axes = (fig.add_subplot(211), fig.add_subplot(212))
    cmap = color_dct.get("cmap", "viridis")
    vmax, vmin = (color_dct.get("vmax", -70), color_dct.get("vmin", -110))
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mpbl = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("$%H^{%M}$"))
        major_locator = mdates.MinuteLocator(byminute=range(0, 60, 10))
        ax.xaxis.set_major_locator(major_locator)
        for v, c in zip(vlines, colors):
            ax.axvline(v, color=c, ls="--", lw=0.6)
    ax0, ax1 = (axes[0], axes[1])
    ax1.set_xlabel("Time [UT]", fontdict={"size": 11, "fontweight": "bold"})
    ax0.set_ylabel(r"Doppler [Hz]", fontdict={"size": 11, "fontweight": "bold"})
    ax1.set_ylabel(r"Power [dB]", fontdict={"size": 11, "fontweight": "bold"})

    for gs in gds:
        o = gs.data["filtered"]["df"]
        val = gs.meta["solar_lon"]
        ax0.plot(o.UTC, o.Freq, ls="-", lw=0.8, color=mpbl.cmap(mpbl.norm(val)))
        ax1.plot(o.UTC, o.Power_dB, ls="-", lw=0.8, color=mpbl.cmap(mpbl.norm(val)))
    fig.colorbar(mpbl, ax=ax0, label="Longitude")
    fig.colorbar(mpbl, ax=ax1, label="Longitude")
    ax0.set_ylim(-5, 5)
    ax1.set_ylim(-70, 0)
    ax0.set_xlim(drange)
    ax1.set_xlim(drange)

    #     ax = axes[1]
    fig.savefig(filepath, bbox_inches="tight", facecolor=(1, 1, 1, 1))
    return


def HamSciParamTS(
    gds,
    filepath,
    flare_timings,
    index=0,
    vlines=[],
    colors=[],
    drange=[],
):
    """ """
    fl_start, fl_peak, fl_end = (
        flare_timings[0],
        flare_timings[1],
        flare_timings[2],
    )
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
            "font.size": 10,
        }
    )
    fig = plt.figure(figsize=(8, 3), dpi=200)
    ax0 = fig.add_subplot(111)
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("$%H^{%M}$"))
    major_locator = mdates.MinuteLocator(byminute=range(0, 60, 10))
    ax0.xaxis.set_major_locator(major_locator)
    for v, c in zip(vlines, colors):
        ax0.axvline(v, color=c, ls="--", lw=0.6)
    o = gds[index].data["filtered"]["df"]
    ax0.plot(o.UTC, o.Freq, ls="-", lw=0.8, color="k")
    ax0.axhline(0, color="gray", ls="--", lw=0.5)
    rise = o[(o.UTC >= fl_start) & (o.UTC <= fl_peak)]
    fall = o[(o.UTC >= fl_peak) & (o.UTC <= fl_end)]
    ax0.axhline(
        rise.Freq.max(),
        color="b",
        ls="--",
        lw=0.5,
        label=r"$D_{peak}$=%.1f Hz" % rise.Freq.max(),
    )
    ax0.fill_between(
        rise.UTC,
        rise.Freq,
        color="r",
        alpha=0.5,
        label=r"$Area_{rise}$=%.1f" % gds[index].df_params["rise_area"],
    )
    ax0.fill_between(
        fall.UTC,
        fall.Freq,
        color="green",
        alpha=0.5,
        label=r"$Area_{fall}$=%.1f" % gds[index].df_params["fall_area"],
    )
    ax0.legend(loc=1)
    ax0.set_ylim(-5, 5)
    ax0.set_xlabel("Time [UT]", fontdict={"size": 11, "fontweight": "bold"})
    ax0.set_ylabel(r"Doppler [Hz]", fontdict={"size": 11, "fontweight": "bold"})
    ax0.set_xlim(drange)
    ax0.text(0.1, 0.9, gds[index].meta["call_sign"] + "/" +str(gds[index].meta["node"]),
    ha="left", va="center", transform=ax0.transAxes)

    fig.savefig(filepath, bbox_inches="tight", facecolor=(1, 1, 1, 1))
    return


def joyplot(
    gds,
    filepath,
    flare_timings,
    vlines=[],
    colors=[],
    drange=[],
    offset=0.75,
    ccolors = []
):
    """
    Create a Joyplot using dataset
    """
    fl_start, fl_peak, fl_end = (
        flare_timings[0],
        flare_timings[1],
        flare_timings[2],
    )
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
            "font.size": 10,
        }
    )
    fig = plt.figure(figsize=(8, 6), dpi=200)
    ax0 = fig.add_subplot(211)
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("$%H^{%M}$"))
    major_locator = mdates.MinuteLocator(byminute=range(0, 60, 30))
    ax0.xaxis.set_major_locator(major_locator)
    for v, c in zip(vlines, colors):
        ax0.axvline(v, color=c, ls="--", lw=0.4)
    ccolors = ccolors if len(ccolors)>0 else ["k"]*len(gds)
    for i, gd in enumerate(gds):
        o = gd.data["filtered"]["df"]
        ax0.plot(
            o.UTC, o.Freq, ls="-", lw=0.8, color=ccolors[i], 
            label=f"{gd.meta['call_sign']} / " + r"$\chi=%.1f^{\circ}$"%gd.meta['sza']
        )
        rise = o[(o.UTC >= fl_start) & (o.UTC <= fl_peak)]
        fall = o[(o.UTC >= fl_peak) & (o.UTC <= fl_end)]
        # ax0.fill_between(
        #     rise.UTC,
        #     y1=(i*offset)+rise.Freq,
        #     y2=(i*offset),
        #     color="r",
        #     alpha=0.5,
        # )
        # ax0.fill_between(
        #     fall.UTC,
        #     y1=(i*offset)+fall.Freq,
        #     y2=(i*offset),
        #     color="green",
        #     alpha=0.5,
        #     label=f"{gd.meta['call_sign']}"
        # )
    ax0.legend(loc=2)
    ax0.text(0.05, 1.05, r"HamSCI, $f_0=$10 MHz", ha="left", va="center", transform=ax0.transAxes)
    ax0.text(0.95, 0.95, "(a)", ha="right", va="center", transform=ax0.transAxes)
    ax0.set_xticklabels([])
    ax0.set_ylim(-1, 3)
    ax0.axhline(0, ls="--", lw=0.4, color="k")
    ax0.set_xlim(drange)
    ax0.set_ylabel("Doppler, Hz")

    ax0 = fig.add_subplot(212)
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("$%H^{%M}$"))
    major_locator = mdates.MinuteLocator(byminute=range(0, 60, 30))
    ax0.xaxis.set_major_locator(major_locator)
    for v, c in zip(vlines, colors):
        ax0.axvline(v, color=c, ls="--", lw=0.4)
    ccolors = ccolors if len(ccolors)>0 else ["k"]*len(gds)
    for i, gd in enumerate(gds):
        o = gd.data["filtered"]["df"]
        ax0.plot(
            o.UTC, o.Power_dB, ls="-", lw=0.8, color=ccolors[i], 
            label=f"{gd.meta['call_sign']} / " + r"$\chi=%.1f^{\circ}$"%gd.meta['sza']
        )
        rise = o[(o.UTC >= fl_start) & (o.UTC <= fl_peak)]
        fall = o[(o.UTC >= fl_peak) & (o.UTC <= fl_end)]
    ax0.set_xlim(drange)
    ax0.set_ylim(-100, 0)
    ax0.set_ylabel("Power, dB")
    ax0.set_xlabel("Time, UT")
    ax0.text(0.95, 0.95, "(b)", ha="right", va="center", transform=ax0.transAxes)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(filepath, bbox_inches="tight", facecolor=(1, 1, 1, 1))
    return


def GOESSDOPlot(time, xl, xs, sdo_time, sdo_euv, filepath, vlines=[], colors=[], drange=[]):
    """ """
    # import scienceplots
    plt.style.use(["science", "ieee"])
    plt.rcParams.update(
        {
            "figure.figsize": np.array([8, 6]),
            "text.usetex": False,
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
    drange = drange if len(drange) == 2 else [time[0], time[-1]]
    fig = plt.figure(figsize=(6, 6), dpi=180)
    ax = fig.add_subplot(211)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("$%H^{%M}$"))
    major_locator = mdates.HourLocator(byhour=range(0, 24, 1))
    ax.xaxis.set_major_locator(major_locator)
    ax.set_xlabel("", fontdict={"size": 11, "fontweight": "bold"})
    ax.set_xlim(drange)
    ax.set_ylim(1e-8, 1e-3)
    for v, c in zip(vlines, colors):
        ax.axvline(v, color=c, ls="--", lw=0.6)
    ax.axhline(1e-6, color="k", ls=":", lw=0.6)
    ax.text(drange[-1], 1e-6, "C", ha="left", va="center")
    ax.axhline(1e-5, color="k", ls=":", lw=0.6)
    ax.text(drange[-1], 1e-5, "M", ha="left", va="center")
    ax.axhline(1e-4, color="k", ls=":", lw=0.6)
    ax.text(drange[-1], 1e-4, "X", ha="left", va="center")
    ax.set_ylabel(
        r"Irradiance [$Wm^{-2}$]", fontdict={"size": 11, "fontweight": "bold"}
    )
    ax.semilogy(
        time, xl, color="r", ls="-", lw=1, label=r"$\lambda_1=0.1-0.8 nm$"
    )
    ax.semilogy(
        time, xs, color="b", ls="-", lw=1, label=r"$\lambda_0=0.05-0.4 nm$"
    )
    ax.text(1.07, 0.99, "GOES X-rays", ha="right", va="top", transform=ax.transAxes, rotation=90)
    ax.set_xticklabels([])
    ax.text(0.05, 0.95, "(a)", ha="left", va="center", transform=ax.transAxes)
    ax.legend(loc=1)
    ax.text(0.05, 1.05, f"Date: {drange[0].strftime('%b %d, %Y')}", ha="left", va="center", transform=ax.transAxes)


    ax = fig.add_subplot(212)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("$%H^{%M}$"))
    # major_locator = mdates.HourLocator(byhour=range(0, 24, 3))
    ax.xaxis.set_major_locator(major_locator)
    ax.set_xlabel("Time [UT]", fontdict={"size": 11, "fontweight": "bold"})
    ax.set_xlim(drange)
    for v, c in zip(vlines, colors):
        ax.axvline(v, color=c, ls="--", lw=0.6)
    ax.set_ylabel(
        r"Irradiance (0.1-7 nm) [$\times 10^{-3}$ $Wm^{-2}$]", fontdict={"size": 11, "fontweight": "bold"}
    )
    ax.plot(
        sdo_time, sdo_euv*1e3, color="k", ls="-", lw=1,
    )
    ax.text(1.07, 0.99, "SDO EUV", ha="right", va="top", transform=ax.transAxes, rotation=90)
    ax.text(0.05, 0.95, "(b)", ha="left", va="center", transform=ax.transAxes)
    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    fig.savefig(filepath, bbox_inches="tight", facecolor=(1, 1, 1, 1))
    return

def DIPlot(time, xl, xs, sdo_time, sdo_euv, gd, filepath, vlines=[], colors=[], drange=[]):
    """ """
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
            "font.size": 10,
        }
    )
    drange = drange if len(drange) == 2 else [time[0], time[-1]]
    fig = plt.figure(figsize=(6, 6), dpi=180)
    ax = fig.add_subplot(211)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("$%H^{%M}$"))
    major_locator = mdates.MinuteLocator(byminute=range(0, 60, 30))
    ax.xaxis.set_major_locator(major_locator)
    ax.set_xlabel("", fontdict={"size": 11, "fontweight": "bold"})
    ax.set_xlim(drange)
    for v, c in zip(vlines, colors):
        ax.axvline(v, color=c, ls="--", lw=0.6)
    ax.set_ylabel(
        r"$\frac{\partial}{\partial t} \Phi_0(\lambda_{X})$ [$\times 10^{6}$ $Wm^{-2}s^{-1}$]", fontdict={"size": 11, "fontweight": "bold"}
    )
    xl_t, xs_t = (
        np.roll(np.diff(smooth(xl), prepend=xl.iloc[0])*1e6, 40),
        np.roll(np.diff(smooth(xs), prepend=xs.iloc[0])*1e6, 40)
    ) 
    ax.plot(
        time, xl_t, color="r", ls="-", lw=1, label=r"$\lambda=0.1-0.8 nm$"
    )
    ax.plot(
        time, xs_t, color="b", ls="-", lw=1, label=r"$\lambda=0.05-0.4 nm$"
    )
    ax.set_xticklabels([])
    ax.text(0.05, 0.95, "(a)", ha="left", va="center", transform=ax.transAxes)
    #ax.legend(loc=2)
    ax.set_ylim(-1, 3)
    ax.text(0.05, 1.05, "Class: X2.8", ha="left", va="center", transform=ax.transAxes)
    o = gd.data["filtered"]["df"]
    i = np.argmax(o.Freq)
    ax.axvline(o.UTC.iloc[i], color="k", ls="-", lw=0.8)


    ax = ax.twinx()#fig.add_subplot(212)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("$%H^{%M}$"))
    major_locator = mdates.MinuteLocator(byminute=range(0, 60, 30))
    ax.xaxis.set_major_locator(major_locator)
    ax.set_xticklabels([])
    ax.set_xlim(drange)
    ax.set_ylim(-1, 3)
    for v, c in zip(vlines, colors):
        ax.axvline(v, color=c, ls="--", lw=0.6)
    ax.set_ylabel(
        r"$\frac{\partial}{\partial t} \Phi_0(\lambda_{E})$ [$\times 10^{-3}$ $Wm^{-2}$]", 
        fontdict={"size": 11, "fontweight": "bold", "color":"darkgreen"}
    )
    sdo_euv = np.roll(np.diff(sdo_euv, prepend=sdo_euv.iloc[0])*1e3, -1)
    ax.plot(
        sdo_time, sdo_euv, color="darkgreen", ls="-", lw=1,
    )
    
    ax = fig.add_subplot(212)
    o = gd.data["filtered"]["df"]
    ax.xaxis.set_major_formatter(mdates.DateFormatter("$%H^{%M}$"))
    major_locator = mdates.MinuteLocator(byminute=range(0, 60, 30))
    ax.xaxis.set_major_locator(major_locator)
    ax.text(0.05, 1.05, r"HamSCI, $f_0=$10 MHz", ha="left", va="center", transform=ax.transAxes)
    for v, c in zip(vlines, colors):
        ax.axvline(v, color=c, ls="--", lw=0.6)
    ax.set_ylim(-1, 3)
    ax.axhline(0, ls="--", lw=0.4, color="k")
    ax.set_xlim(drange)
    ax.axvline(o.UTC.iloc[i], color="k", ls="-", lw=0.8)
    ax.set_xlabel("Time, UT")
    ax.set_ylabel("Doppler, Hz")
    ax.plot(o.UTC, o.Freq, ls="-", lw=0.8, color="k", label=f"{gd.meta['call_sign']}")
    ax.set_xlabel("Time [UT]", fontdict={"size": 11, "fontweight": "bold"})
    ax.text(0.05, 0.95, "(b)", ha="left", va="center", transform=ax.transAxes)
    n_euv = utils.compute_normalized_MI(np.array(o.Freq)[::60], np.array(sdo_euv[:-1]))[0]*1.25
    n_xl = utils.compute_normalized_MI(np.array(o.Freq), np.array(xl_t))[0]/4
    n_xs = utils.compute_normalized_MI(np.array(o.Freq), np.array(xs_t))[0]/4
    ax.text(0.05, 0.87, r"$\mathcal{N}_{\lambda_0}=%.2f$"%n_xs, ha="left", va="center", transform=ax.transAxes, 
        fontdict={"color":"blue"})
    ax.text(0.05, 0.79, r"$\mathcal{N}_{\lambda_1}=%.2f$"%n_xl, ha="left", va="center", transform=ax.transAxes, 
        fontdict={"color":"red"})
    ax.text(0.05, 0.71, r"$\mathcal{N}_{\lambda_{E}}=%.2f$"%n_euv, ha="left", va="center", transform=ax.transAxes, 
        fontdict={"color":"darkgreen"})
    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    fig.savefig(filepath, bbox_inches="tight", facecolor=(1, 1, 1, 1))
    return

def smooth(x,window_len=21,window="hanning"):
    x = np.array(x)
    if x.ndim != 1: raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len: raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3: return x
    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]: raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == "flat": w = numpy.ones(window_len,"d")
    else: w = eval("np."+window+"(window_len)")
    y = np.convolve(w/w.sum(),s,mode="valid")
    d = window_len - 1
    y = y[int(d/2):-int(d/2)]
    return y