"""
VoigtFit is a Python package designed to fit Voigt profiles to absorption
lines in spectral data. The package handles multiple spectra simultaneously,
and fits multiple component structure of several absorption lines using a
Levenberg--Marquardt minimization algorithm to identify the optimal parameters.

Written by Jens-Kristian Krogager.
"""

__author__ = 'Jens-Kristian Krogager'

from os import path
from VoigtFit import *
from dataset import DataSet
from dataset import Line
import dataset
import hdf5_save as hdf5
import line_complexes
import molecules
import output
import regions
import voigt

code_dir = path.dirname(path.abspath(__file__))
with open(path.join(code_dir, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()
