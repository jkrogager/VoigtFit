"""
VoigtFit is a Python package designed to fit Voigt profiles to absorption
lines in spectral data. The package handles multiple spectra simultaneously,
and fits multiple component structure of several absorption lines using a
Levenberg--Marquardt minimization algorithm to identify the optimal parameters.

Written by Jens-Kristian Krogager.
"""

__author__ = 'Jens-Kristian Krogager'

from os import path
from sys import version_info

import warnings
import matplotlib
# The native MacOSX backend doesn't work for all:
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    matplotlib.use('TkAgg')

from VoigtFit.container.components import Component
from VoigtFit.container.dataset import DataSet
from VoigtFit.container.lines import Line, show_transitions
from VoigtFit.container.regions import Region
from VoigtFit.funcs import voigt
from VoigtFit.io.hdf5_save import load_dataset, save_dataset
from VoigtFit.io import parse_input, output
from VoigtFit.funcs import limits


code_dir = path.dirname(path.abspath(__file__))
with open(path.join(code_dir, 'VERSION')) as version_file:
    version = version_file.read().strip()
    if version_info[0] >= 3:
        v_items = version.split('.')
        v_items[0] = '3'
        version = '.'.join(v_items)
    __version__ = version
