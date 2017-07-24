import numpy as np
from terminal_attributes import bold, reset
import os.path

dt = [('element', 'S2'), ('N', 'f4'), ('N_err', 'f4'), ('N_m', 'f4'), ('N_m_err', 'f4')]
file_path = os.path.abspath(os.path.dirname(__file__))
relative_path = os.path.join(file_path, "static/Asplund2009.dat")
data = np.loadtxt(relative_path, dtype=dt)

photosphere = dict()
meteorite = dict()

for element, N_phot, N_phot_err, N_met, N_met_err in data:
    photosphere[element] = [N_phot, N_phot_err]
    meteorite[element] = [N_met, N_met_err]

print " Loaded Solar abundances from Asplund et al. 2009"
print bold+"    The Chemical Composition of the Sun"+reset
print " Annual Review of Astronomy and Astrophysics"
print "             Vol. 47: 481-522"
print ""
print " Data available:  photosphere,  meteorite"
