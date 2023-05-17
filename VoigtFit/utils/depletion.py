"""
Implementation of depletion sequences following De Cia et al. (2016)
using updated coefficients by Konstantopoulou et al. (2022)
"""

__author__ = 'Jens-Kristian Krogager'

from astropy.table import Table

import numpy as np
import os

root_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.sep.join(root_path.split(os.sep)[:-1])
datafile = os.path.join(root_path, 'static', 'Konstantopoulou2022.dat')

data = Table.read(datafile, format='csv', comment='#')

A2 = {}
B2 = {}
coeffs = {}
for row in data:
    A2[row['X']] = row['A2']
    B2[row['X']] = row['B2']
    coeffs[row['X']] = np.array([row['A2'], row['B2']])

