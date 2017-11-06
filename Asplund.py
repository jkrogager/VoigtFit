import numpy as np
import os
# from terminal_attributes import bold, reset

if 'VFITDATA' in os.environ.keys():
    datafile = os.environ['VFITDATA']+'/Asplund2009.dat'

else:
    print("No VFITDATA in environment ... Using relative path to static data files")
    datafile = os.path.dirname(__file__) + '/static/Asplund2009.dat'

dt = [('element', 'S2'), ('N', 'f4'), ('N_err', 'f4'), ('N_m', 'f4'), ('N_m_err', 'f4')]
data = np.loadtxt(datafile, dtype=dt)

photosphere = dict()
meteorite = dict()

for element, N_phot, N_phot_err, N_met, N_met_err in data:
    photosphere[element] = [N_phot, N_phot_err]
    meteorite[element] = [N_met, N_met_err]

print "\n Loaded Solar abundances from Asplund et al. 2009  (photospheric)"
# print bold+"    The Chemical Composition of the Sun"+reset
# print " Annual Review of Astronomy and Astrophysics"
# print "             Vol. 47: 481-522"
# print ""
# print " Data available:  photosphere,  meteorite"
print ""
