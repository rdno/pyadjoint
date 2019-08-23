#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple waveform misfit and adjoint source.

This file will also serve as an explanation of how to add new adjoint
sources to Pyadjoint.

:copyright:
    created by Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
    modified by Yanhua O. Yuan (yanhuay@princeton.edu), 2017
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.integrate import simps

from ..utils import generic_adjoint_source_plot, window_taper
from ..config import ConfigDoubleDifferenceWaveForm


# This is the verbose and pretty name of the adjoint source defined in this
# function.
VERBOSE_NAME = "Convolved Wavefield"

# Long and detailed description of the adjoint source defined in this file.
# Don't spare any details. This will be rendered as restructured text in the
# documentation. Be careful to escape the string with an ``r`` prefix.
# Otherwise most backslashes will have a special meaning which messes with the
# TeX like formulas.
DESCRIPTION = r"""
Misfit Based on Choi & Alkhalifah (2011)
"""

# Optional: document any additional parameters this particular adjoint sources
# receives in addition to the ones passed to the central adjoint source
# calculation function. Make sure to indicate the default values. This is a
# bit redundant but the only way I could figure out to make it work with the
# rest of the architecture of pyadjoint.
ADDITIONAL_PARAMETERS = r"""
**taper_percentage** (:class:`float`)
    Decimal percentage of taper at one end (ranging from ``0.0`` (0%) to
    ``0.5`` (50%)). Defaults to ``0.15``.

**taper_type** (:class:`str`)
    The taper type, supports anything :method:`obspy.core.trace.Trace.taper`
    can use. Defaults to ``"hann"``.
"""


# Each adjoint source file must contain a calculate_adjoint_source()
# function. It must take observed, synthetic, min_period, max_period,
# left_window_border, right_window_border, adjoint_src, and figure as
# parameters. Other optional keyword arguments are possible.
def calculate_adjoint_source_DD(observed1, synthetic1,
                                observed2, synthetic2,
                                config,
                                window1, window2,
                                adjoint_src, figure):

    if not isinstance(config, ConfigDoubleDifferenceWaveForm):
        raise ValueError("Wrong configure parameters for waveform "
                         "adjoint source")

    observed, synthetic = observed1, synthetic1
    observed_ref, synthetic_ref = observed2, synthetic2
    window, window_ref = window1, window2
    ret_val_1 = {}
    ret_val_2 = {}

    measurement1 = []
    measurement2 = []

    nlen_data = len(synthetic.data)
    deltat = synthetic.stats.delta

    adj1 = np.zeros(nlen_data)
    adj2 = np.zeros(nlen_data)

    misfit_sum = 0.0

    # loop over time windows
    for wins1, wins2 in zip(window, window_ref):

        measure1_wins = {}
        measure2_wins = {}

        left_window_border_1 = wins1[0]
        right_window_border_1 = wins1[1]
        left_window_border_2 = wins2[0]
        right_window_border_2 = wins2[1]

        left_sample_1 = int(np.floor(left_window_border_1 / deltat)) + 1
        left_sample_2 = int(np.floor(left_window_border_2 / deltat)) + 1
        nlen1 = int(np.floor((right_window_border_1 -
                             left_window_border_1) / deltat)) + 1
        nlen2 = int(np.floor((right_window_border_2 -
                             left_window_border_2) / deltat)) + 1

        right_sample_1 = left_sample_1 + nlen1
        right_sample_2 = left_sample_2 + nlen2

        nlen = max(nlen1, nlen2)

        d = np.zeros(nlen)
        s = np.zeros(nlen)
        d_ref = np.zeros(nlen)
        s_ref = np.zeros(nlen)

        d[0:nlen1] = observed.data[left_sample_1:right_sample_1]
        s[0:nlen1] = synthetic.data[left_sample_1:right_sample_1]
        d_ref[0:nlen2] = observed_ref.data[left_sample_2:right_sample_2]
        s_ref[0:nlen2] = synthetic_ref.data[left_sample_2:right_sample_2]

        # All adjoint sources will need some kind of windowing taper
        window_taper(d[0:nlen1], taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(s[0:nlen1], taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(d_ref[0:nlen2], taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(s_ref[0:nlen2], taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)

        diff = np.convolve(s, d_ref, "same") - np.convolve(d, s_ref, "same")

        # Integrate with the composite Simpson's rule.
        misfit_win = 0.5 * simps(y=diff**2, dx=deltat)
        misfit_sum += misfit_win

        # YY: All adjoint sources will need windowing taper again
        window_taper(diff, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)

        adj1[left_sample_1:right_sample_1] = diff[0:nlen1]
        adj2[left_sample_2:right_sample_2] = 0

        measure1_wins["type"] = "convolved"
        measure1_wins["difference"] = np.mean(diff)
        measure1_wins["misfit"] = misfit_win

        measure2_wins["type"] = "zero"
        measure2_wins["difference"] = np.mean(diff)
        measure2_wins["misfit"] = misfit_win

        measurement1.append(measure1_wins)
        measurement2.append(measure2_wins)

    ret_val_1["misfit"] = misfit_sum
    ret_val_1["measurement"] = measurement1
    ret_val_2["misfit"] = misfit_sum
    ret_val_2["measurement"] = measurement2

    if adjoint_src is True:
        ret_val_1["adjoint_source"] = adj1[::-1]
        ret_val_2["adjoint_source"] = adj2[::-1]

    if figure:
        return NotImplemented

    return ret_val_1, ret_val_2
