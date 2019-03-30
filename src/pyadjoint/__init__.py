#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:copyright:
    created by Lion Krischer (krischer@geophysik.uni-muenchen.de), 2015
    modified by Yanhua O. Yuan (yanhuay@princeton.edu), 2017
:license:
    BSD 3-Clause ("BSD New" or "BSD Simplified")
"""
from __future__ import absolute_import, division, print_function

import logging


class PyadjointError(Exception):
    """
    Base class for all Pyadjoint exceptions. Will probably be used for all
    exceptions to not overcomplicate things as the whole package is pretty
    small.
    """
    pass


class PyadjointWarning(UserWarning):
    """
    Base class for all Pyadjoint warnings.
    """
    pass


__version__ = "0.0.1a"


# setup the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
# Prevent propagating to higher loggers.
logger.propagate = 0
# Console log handler.
ch = logging.StreamHandler()
# Add formatter
FORMAT = "[%(asctime)s] - %(name)s - %(levelname)s: %(message)s"
formatter = logging.Formatter(FORMAT)
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

# Main objects and functions available at the top level.
from .adjoint_source import AdjointSource, calculate_adjoint_source, calculate_adjoint_source_DD  # NOQA
from .adjoint_source import calculate_attenuation_adjoint_source  # NOQA

# load configure
from .config import ConfigMultiTaper, ConfigWaveForm, ConfigCrossCorrelation, ConfigExponentiatedPhase  # NOQA
from .config import ConfigDoubleDifferenceCrossCorrelation, ConfigDoubleDifferenceMultiTaper   # NOQA
