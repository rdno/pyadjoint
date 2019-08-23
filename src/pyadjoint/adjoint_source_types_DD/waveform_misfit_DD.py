#!/usr/bin/env python
# -*- encoding: utf-8 -*-
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
VERBOSE_NAME = "Waveform Misfit DD"

# Long and detailed description of the adjoint source defined in this file.
# Don't spare any details. This will be rendered as restructured text in the
# documentation. Be careful to escape the string with an ``r`` prefix.
# Otherwise most backslashes will have a special meaning which messes with the
# TeX like formulas.
DESCRIPTION = r"""
This is the simplest of all misfits and is defined as the squared difference
between observed and synthetic data. The misfit :math:`\chi(\mathbf{m})` for a
given Earth model :math:`\mathbf{m}` and a single receiver and component is
given by

.. math::

    \chi (\mathbf{m}) = \frac{1}{2} \int_0^T \left| \mathbf{d}(t) -
    \mathbf{s}(t, \mathbf{m}) \right| ^ 2 dt

:math:`\mathbf{d}(t)` is the observed data and
:math:`\mathbf{s}(t, \mathbf{m})` the synthetic data.

The adjoint source for the same receiver and component is given by

.. math::

    f^{\dagger}(t) = - \left[ \mathbf{d}(T - t) -
    \mathbf{s}(T - t, \mathbf{m}) \right]

For the sake of simplicity we omit the spatial Kronecker delta and define
the adjoint source as acting solely at the receiver's location. For more
details, please see [Tromp2005]_ and [Bozdag2011]_.

This particular implementation here uses
`Simpson's rule <http://en.wikipedia.org/wiki/Simpson's_rule>`_
to evaluate the definite integral.
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

    ret_val_1 = {}
    ret_val_2 = {}

    measurement1 = []
    measurement2 = []

    nlen_data = len(synthetic1.data)
    deltat = synthetic1.stats.delta

    adj1 = np.zeros(nlen_data)
    adj2 = np.zeros(nlen_data)

    misfit_sum = 0.0

    # loop over time windows
    for wins1, wins2 in zip(window1, window2):

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

        d1 = np.zeros(nlen)
        s1 = np.zeros(nlen)
        d2 = np.zeros(nlen)
        s2 = np.zeros(nlen)

        d1[0:nlen1] = observed1.data[left_sample_1:right_sample_1]
        s1[0:nlen1] = synthetic1.data[left_sample_1:right_sample_1]
        d2[0:nlen2] = observed2.data[left_sample_2:right_sample_2]
        s2[0:nlen2] = synthetic2.data[left_sample_2:right_sample_2]

        # All adjoint sources will need some kind of windowing taper
        window_taper(d1[0:nlen1], taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(s1[0:nlen1], taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(d2[0:nlen2], taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        window_taper(s2[0:nlen2], taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)

        diff_syn = s1 - s2
        diff_obs = d1 - d2
        diff = diff_syn - diff_obs

        # Integrate with the composite Simpson's rule.
        misfit_win = 0.5 * simps(y=diff**2, dx=deltat)
        misfit_sum += misfit_win

        # YY: All adjoint sources will need windowing taper again
        window_taper(diff, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)

        adj1[left_sample_1:right_sample_1] = diff[0:nlen1]
        adj2[left_sample_2:right_sample_2] = -diff[0:nlen2]

        measure1_wins["type"] = "wf_dd_1"
        measure1_wins["difference"] = np.mean(diff)
        measure1_wins["misfit"] = misfit_win

        measure2_wins["type"] = "wf_dd_2"
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
