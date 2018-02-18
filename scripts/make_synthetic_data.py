import numpy as np
import matplotlib.pyplot as plt
from VoigtFit.voigt import Voigt
from VoigtFit import show_transitions
from scipy.signal import fftconvolve, gaussian

# --- INPUT ---
SNR = 60.
wl = np.arange(1000., 3000., 0.010)
R = 45000.

elements = {'CII': 14.32, 'SiII': 13.95,
            'SII': 13.69, 'FeII': 13.59, 'ZnII': 11.54,
            'OI': 14.86, 'AlII': 12.47, 'AlIII': 11.90,
            'CrII': 12.10}

z1 = 0.0033
T1 = 4370.     # Kelvin
b_turb1 = 4.6  # km/s

z2 = 0.0036
T2 = 8600.     # Kelvin
b_turb2 = 5.2  # km/s

components = [[z1, T1, b_turb1, 3.0], [z2, T2, b_turb2, 1.0]]

K = 0.0166287

l = np.logspace(np.log10(wl.min()-50*0.03), np.log10(wl.max()+50*0.03), 3*(len(wl)+100))
tau = np.zeros_like(l)
pxs = np.diff(l)[0] / l[0] * 299792.458

# print " - Including following transitions:"
for ion, logN in elements.items():
    transitions = show_transitions(ion, lower=1000.)
    for trans in transitions:
        # print "  %s  %.3f" % (trans['trans'], trans['l0'])
        for z, T, beta, fN in components:
            b = np.sqrt(beta**2 + K*T/trans['mass'])
            tau += Voigt(l, trans['l0'], trans['f'], fN*10**logN, b*1.e5, trans['gam'], z=z)
            plt.axvline(trans['l0']*(z+1), ymin=0.9, ymax=0.97, color='crimson')

transmission = np.exp(-tau)

fwhm_instrumental = 299792.458/R                                     # in units of km/s
sigma_instrumental = fwhm_instrumental / 2.35482 / pxs    # in units of pixels
LSF = gaussian(len(l)/2, sigma_instrumental)
LSF = LSF/LSF.sum()
profile_broad = fftconvolve(transmission, LSF, 'same')

P_obs = np.interp(wl, l, profile_broad)

noise = np.random.normal(0., 1., len(P_obs))
f_obs = P_obs + P_obs/SNR*noise

plt.plot(wl, f_obs)

plt.ylabel("Normalized flux")
plt.xlabel("Wavelength  [Angstrom]")

data = np.column_stack([wl, f_obs, P_obs/SNR])
with open('thermal_model_2comp.dat', 'w') as output_file:
    output_file.write("# Synthetic Normalized Spectrum for VoigtFit\n")
    output_file.write("# Resolution: %.1f \n" % R)
    np.savetxt(output_file, data, fmt="%.3f  %.3e  %.3e")

with open('thermal_model_2comp.input', 'w') as par_file:
    par_file.write("\nElement Column Densities:\n")
    par_file.write("-------------------------\n\n")

    n_comp = len(components)
    header = "ion    "
    for num in range(n_comp):
        header += "comp%i" % (num+1)
        header += "  "
    par_file.write(header+"\n")

    fmt_str = "%5s" + (n_comp+1)*"  %5.2f"
    for ion, logN in elements.items():
        pars = [ion]
        N_tot = 0.
        for num in range(n_comp):
            fN = components[num][-1]
            pars.append(logN+np.log10(fN))
            N_tot += fN*10**logN
        pars.append(np.log10(N_tot))
        pars = tuple(pars)

        par_file.write(fmt_str % pars + "\n")

    par_file.write("\n\nVelocity Information:\n")
    par_file.write("---------------------\n\n")
    par_file.write("component   z         b_turb     T\n")
    fmt_str = "%2i          %8.6f  %5.2f    %6.0f"
    for num, comp in enumerate(components):
        pars = (num+1, comp[0], comp[2], comp[1])
        par_file.write(fmt_str % pars + "\n")
    par_file.write("")
