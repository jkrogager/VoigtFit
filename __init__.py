"""
VoigtFit is a Python package designed to fit Voigt profiles to absorption
lines in spectral data. The package handles multiple spectra simultaneously,
and fits multiple component structure of several absorption lines using a
Levenberg--Marquardt minimization algorithm to identify the optimal parameters.

Written by Jens-Kristian Krogager.
"""
print "\n"
print "    VoigtFit"
print ""
print "    by Jens-Kristian Krogager"
print ""
print "    Institut d'Astrophysique de Paris"
print "    November 2017"
print ""
print "  ____  _           ___________________"
print "      \/ \  _/\    /                   "
print "          \/   \  / oigtFit            "
print "                \/                     "
print ""
print ""
__author__ = 'Jens-Kristian Krogager'

from VoigtFit import *
import regions
import output
import voigt
import dataset
import line_complexes
