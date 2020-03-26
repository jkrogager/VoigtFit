# -*- coding: UTF-8 -*-

__author__ = 'Jens-Kristian Krogager'

import numpy as np
import os

root_path = os.path.dirname(os.path.abspath(__file__))
datafile = root_path + '/static/Asplund2009.dat'

dt = [('element', 'U2'), ('N', 'f4'), ('N_err', 'f4'), ('N_m', 'f4'), ('N_m_err', 'f4')]
data = np.loadtxt(datafile, dtype=dt)

fname = root_path + '/static/Lodders2009.dat'
Lodders2009 = np.loadtxt(fname, usecols=(1, 2), dtype=str)

photosphere = dict()
meteorite = dict()
solar = dict()

for element, N_phot, N_phot_err, N_met, N_met_err in data:
    photosphere[element] = [N_phot, N_phot_err]
    meteorite[element] = [N_met, N_met_err]
    idx = (Lodders2009 == element).nonzero()[0][0]
    typeN = Lodders2009[idx][1]
    if typeN == 's':
        solar[element] = [N_phot, N_phot_err]
    elif typeN == 'm':
        solar[element] = [N_met, N_met_err]
    elif typeN == 'a':
        # Calculate the weighted average
        this_N = np.array([N_phot, N_met])
        this_e = np.array([N_phot_err, N_met_err])
        w = 1./this_e**2
        N_avg = np.sum(w*this_N) / np.sum(w)
        N_err = np.round(1./np.sqrt(np.sum(w)), 3)
        solar[element] = [N_avg, N_err]
