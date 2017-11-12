"""
VoigtFit is a Python package designed to fit Voigt profiles to absorption
lines in spectral data. The package handles multiple spectra simultaneously,
and fits multiple component structure of several absorption lines using a
Levenberg--Marquardt minimization algorithm to identify the optimal parameters.

Written by Jens-Kristian Krogager.
"""
__author__ = 'Jens-Kristian Krogager'
__version__ = '1.0.0'

from VoigtFit import *
import regions
import output
import voigt
import dataset
import line_complexes
