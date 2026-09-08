"""
VoigtFit is a Python package designed to fit Voigt profiles to absorption
lines in spectral data. The package handles multiple spectra simultaneously,
and fits multiple component structure of several absorption lines using a
Levenberg--Marquardt minimization algorithm to identify the optimal parameters.

Written by Jens-Kristian Krogager.
"""

__author__ = 'Jens-Kristian Krogager'

# import warnings
# import matplotlib
# # The native MacOSX backend doesn't work for all:
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     matplotlib.use('TkAgg')

from VoigtFit.components import Component
from VoigtFit.dataset import DataSet
from VoigtFit.lines import Line, show_transitions
from VoigtFit.regions import Region
from VoigtFit import voigt
from VoigtFit.hdf5_save import load_dataset, save_dataset
from VoigtFit import parse_input, output
from VoigtFit import limits
from VoigtFit.main import run_voigtfit


import importlib.metadata
__version__ = importlib.metadata.version("VoigtFit")
