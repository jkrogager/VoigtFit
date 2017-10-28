# -*- coding: utf-8 -*-
#   VoigtFit
#   module to evaluate line profile

import numpy as np
from scipy.signal import fftconvolve, gaussian


# ==== VOIGT PROFILE ===============
def H(a, x):
    P = x**2
    H0 = np.exp(-x**2)
    Q = 1.5/x**2
    return H0 - a/np.sqrt(np.pi)/P * (H0*H0*(4.*P*P + 7.*P + 4. + Q) - Q - 1)


def Voigt(l, l0, f, N, b, gam, z=0):
    """Calculate the Voigt profile of transition with
    rest frame transition wavelength: 'l0'
    oscillator strength: 'f'
    column density: N  cm^-2
    velocity width: b  cm/s
    """
    # ==== PARAMETERS ==================

    c = 2.99792e10        # cm/s
    m_e = 9.1095e-28       # g
    e = 4.8032e-10        # cgs units

    # ==================================
    # Calculate Profile

    C_a = np.sqrt(np.pi)*e**2*f*l0*1.e-8/m_e/c/b
    a = l0*1.e-8*gam/(4.*np.pi*b)

    dl_D = b/c*l0
    l = l/(z+1.)
    x = (l - l0)/dl_D + 0.00001

    tau = np.float64(C_a) * N * H(a, x)
    return tau


def evaluate_continuum(x, pars, reg_num):
    """
    Function to evaluate the continuum model
    using Chebyshev polynomials.
    All regions are fitted with the same order
    of polynomials.
    """
    cheb_parnames = list()
    p_cont = list()

    # Find Chebyshev parameters for this region:
    # They are named like 'R0_cheb_p0, R0_cheb_p1, R1_cheb_p0, etc...'
    for parname in pars.keys():
        if 'R%i_cheb' % reg_num in parname:
            cheb_parnames.append(parname)
    # This should be calculated at the point of generating
    # the parameters, since this is a fixed data structure
    # Sort the names, to arange the coefficients right:
    cheb_parnames = sorted(cheb_parnames)
    for parname in cheb_parnames:
        p_cont.append(pars[parname].value)

    # Calculate Chebyshev polynomium in x-range:
    cont_model = np.polynomial.Chebyshev(p_cont, domain=(x.min(), x.max()))

    return cont_model(x)


def evaluate_profile(x, pars, z_sys, lines, components, res, dv=0.1):
    """
    Function to evaluate Voigt profile for a fitting `Region'.

    Parameters
    ----------
    x : np.array
        Wavelength array to evaluate the profile on

    pars : dictionary
        Dictionary containing fit parameters from `lmfit'

    lines : list
        List of lines in the region to evaluate.
        Should be a list of `Line' instances.

    components : dictionary
        Dictionary containing component data for the defined ions.

    res : float
        Spectral resolution of the region in km/s.

    dv : float  [default=0.1]
        Desired pixel size of profile evaluation in km/s.

    Returns
    -------
    profile_obs : np.array
        Total observed line profile of all lines in the region,
        convolved with the instrument Line Spread Function.
    """

    dx = np.mean(np.diff(x))
    xmin = np.log10(x.min() - 50*dx)
    xmax = np.log10(x.max() + 50*dx)
    # N = int((x.max() - x.min())/(0.5*x.max() + 0.5*x.min())*299792.458 / dv)
    N = 3*len(x)

    # Calculate logarithmically binned wavelength grid:
    profile_wl = np.logspace(xmin, xmax, N)
    tau = np.zeros_like(profile_wl)

    # Calculate actual pixel size in km/s:
    pxs = np.diff(profile_wl)[0] / profile_wl[0] * 299792.458

    # Determine range in which to evaluate the profile:
    max_logN = max([val.value for key, val in pars.items() if 'logN' in key])
    if max_logN > 19.0:
        velspan = 20000.*(1. + 1.0*(max_logN-19.))
    else:
        velspan = 20000.

    for line in lines:
        if line.active:
            l0, f, gam = line.get_properties()
            ion = line.ion
            n_comp = len(components[ion])
            l_center = l0*(z_sys + 1.)
            vel = (profile_wl - l_center)/l_center*299792.
            span = (vel >= -velspan)*(vel <= velspan)
            ion = ion.replace('*', 'x')
            for n in range(n_comp):
                z = pars['z%i_%s' % (n, ion)].value
                if x.min() < l0*(z+1) < x.max():
                    pass
                else:
                    continue
                b = pars['b%i_%s' % (n, ion)].value
                logN = pars['logN%i_%s' % (n, ion)].value
                tau[span] += Voigt(profile_wl[span], l0, f, 10**logN, 1.e5*b, gam, z=z)

    profile = np.exp(-tau)
    # Calculate Line Spread Function, i.e., instrumental broadening:
    # Since the wavelength grid is logarithmically binned,
    # the kernel is constant in velocity-space:
    fwhm_instrumental = res                                   # in units of km/s
    sigma_instrumental = fwhm_instrumental / 2.35482 / pxs    # in units of pixels
    LSF = gaussian(len(profile_wl)/2, sigma_instrumental)
    LSF = LSF/LSF.sum()
    profile_broad = fftconvolve(profile, LSF, 'same')

    # Interpolate onto the data grid:
    profile_obs = np.interp(x, profile_wl, profile_broad)

    return profile_obs
