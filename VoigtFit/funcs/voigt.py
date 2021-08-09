# -*- coding: utf-8 -*-
"""
The module contains functions to evaluate the optical depth,
to convert this to observed transmission and to convolve the
observed spectrum with the instrumental profile.
"""

__author__ = 'Jens-Kristian Krogager'

import numpy as np
from scipy.signal import fftconvolve, gaussian
from numba import jit

import re

# Regular Expression to match redshift parameter names:
# ex: z0_FeII, z0_H2J0, z3_HI, z15_TiII
z_matcher = re.compile('z[0-9]+_[A-Z][A-Z]?[0-9]?[a-z]?[I-Z]+[0-9]?[a-z]?')


# ==== VOIGT PROFILE ===============
def H(a, x):
    """Voigt Profile Approximation from T. Tepper-Garcia (2006, 2007)."""
    P = x**2
    H0 = np.exp(-x**2)
    Q = 1.5/x**2
    return H0 - a/np.sqrt(np.pi)/P * (H0*H0*(4.*P*P + 7.*P + 4. + Q) - Q - 1)


def Voigt(wl, l0, f, N, b, gam, z=0):
    """
    Calculate the optical depth Voigt profile.

    Parameters
    ----------
    wl : array_like, shape (N)
        Wavelength grid in Angstroms at which to evaluate the optical depth.

    l0 : float
        Rest frame transition wavelength in Angstroms.

    f : float
        Oscillator strength.

    N : float
        Column density in units of cm^-2.

    b : float
        Velocity width of the Voigt profile in cm/s.

    gam : float
        Radiation damping constant, or Einstein constant (A_ul)

    z : float
        The redshift of the observed wavelength grid `l`.

    Returns
    -------
    tau : array_like, shape (N)
        Optical depth array evaluated at the input grid wavelengths `l`.
    """
    # ==== PARAMETERS ==================

    c = 2.99792e10        # cm/s
    m_e = 9.1094e-28       # g
    e = 4.8032e-10        # cgs units

    # ==================================
    # Calculate Profile

    C_a = np.sqrt(np.pi)*e**2*f*l0*1.e-8/m_e/c/b
    a = l0*1.e-8*gam/(4.*np.pi*b)

    dl_D = b/c*l0
    wl = wl/(z+1.)
    x = (wl - l0)/dl_D + 0.00001

    tau = np.float64(C_a) * N * H(a, x)
    return tau


@jit(nopython=True)
def convolve_numba(P, kernel):
    """
    Define convolution function for a wavelength dependent kernel.

    Parameters
    ----------
    P : array_like, shape (N)
        Intrinsic line profile.

    kernel : np.array, shape (N, M)
        Each row of the `kernel` corresponds to the wavelength dependent
        line-spread function (LSF) evaluated at each pixel of the input
        profile `P`. Each LSF must be normalized!

    Returns
    -------
    P_con : np.array, shape (N)
        Resulting profile after performing convolution with `kernel`.

    Notes
    -----
    This function is decorated by the `jit` decorator from `numba`_ in order
    to speed up the calculation.

    .. _numba: http://numba.pydata.org/
    """
    N = kernel.shape[1]//2
    pad = np.ones(N)
    P_pad = np.concatenate((pad, P, pad))
    P_con = np.zeros_like(P)
    for i, lsf_i in enumerate(kernel):
        P_con[i] = np.sum(P_pad[i:i+2*N+1] * lsf_i)

    return P_con


def evaluate_continuum(x, pars, reg_num):
    """
    Evaluate the continuum model using Chebyshev polynomials.
    All regions are fitted with the same order of polynomials.

    Parameters
    ----------
    x : array_like, shape (N)
        Input wavelength grid in Ångstrøm.

    pars : dict(lmfit.Parameters_)
        An instance of lmfit.Parameters_ containing the Chebyshev
        coefficients for each region.

    reg_num : int
        The region number, i.e., the index of the region in the list
        :attr:`VoigtFit.DataSet.regions`.

    Returns
    -------
    cont_model : array_like, shape (N)
        The continuum Chebyshev polynomial evaluated at the input wavelengths `x`.
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
    cheb_parnames = sorted(cheb_parnames, key=lambda x: int(x.split('_')[-1].replace('p', '')))
    for parname in cheb_parnames:
        p_cont.append(pars[parname].value)

    # Calculate Chebyshev polynomium in x-range:
    cont_model = np.polynomial.Chebyshev(p_cont, domain=(x.min(), x.max()))

    return cont_model(x)


def evaluate_profile(x, pars, lines, kernel, z_sys=None, sampling=3, kernel_nsub=1):
    """
    Evaluate the observed Voigt profile. The calculated optical depth, `tau`,
    is converted to observed transmission, `f`:

    .. math:: f = e^{-\\tau}

    The observed transmission is subsequently convolved with the instrumental
    broadening profile assumed to be Gaussian with a full-width at half maximum
    of res. The resolving power is assumed to be constant in velocity space.

    Parameters
    ----------
    x : array_like, shape (N)
        Wavelength array in Ångstrøm on which to evaluate the profile.

    pars : dict(lmfit.Parameters_)
        An instance of lmfit.Parameters_ containing the line parameters.

    lines : list(:class:`Line <dataset.Line>`)
        List of lines to evaluate. Should be a list of
        :class:`Line <dataset.Line>` objects.

    kernel : np.array, shape (N, M)  or float
        The convolution kernel for each wavelength pixel.
        If an array is given, each row of the array must specify
        the line-spread function (LSF) at the given wavelength pixel.
        The LSF must be normalized!
        If a float is given, the resolution FWHM is given in km/s (c/R).
        In this case the spectral resolution is assumed
        to be constant in velocity space.

    z_sys : float or None
        The systemic redshift, used to determine an effective wavelength range
        within which to evaluate the profile. This is handy when fitting very large
        regions, such as HI and metal lines together.

    sampling : integer  [default = 3]
        The subsampling factor used for defining the input logarithmically
        space wavelength grid. The number of pixels in the evaluation will
        be sampling * N, where N is the number of input pixels.
        The final profile will be interpolated back onto the original
        wavelength grid defined by `x`.

    kernel_nsub : integer
        Kernel subsampling factor relative to the data.
        This is only used if the resolution is given as a LSF file.

    Returns
    -------
    profile_obs : array_like, shape (N)
        Observed line profile convolved with the instrument profile.
    """

    if isinstance(kernel, float):
        # Create logarithmically binned grid:
        dx = np.mean(np.diff(x))
        xmin = np.log10(x.min() - 50*dx)
        xmax = np.log10(x.max() + 50*dx)
        N = int(sampling * len(x))
        profile_wl = np.logspace(xmin, xmax, N)
        # Calculate actual pixel size in km/s:
        pxs = np.diff(profile_wl)[0] / profile_wl[0] * 299792.458
        # Set Gaussian kernel width:
        kernel = kernel / pxs / 2.35482

    elif isinstance(kernel, np.ndarray):
        N = int(kernel_nsub * len(x))
        assert kernel.shape[0] == N
        # evaluate on the input grid subsampled by `nsub`:
        if kernel_nsub > 1:
            profile_wl = np.linspace(x.min(), x.max(), N)
        else:
            profile_wl = x.copy()

    else:
        err_msg = "Invalid type of `kernel`: %r" % type(kernel)
        raise TypeError(err_msg)

    tau = evaluate_optical_depth(profile_wl, pars, lines, z_sys=z_sys)
    profile = np.exp(-tau)

    if isinstance(kernel, float):
        LSF = gaussian(10*int(kernel) + 1, kernel)
        LSF = LSF/LSF.sum()
        profile_broad = fftconvolve(profile, LSF, 'same')
        # Interpolate onto the data grid:
        profile_obs = np.interp(x, profile_wl, profile_broad)

    else:
        profile_broad = convolve_numba(profile, kernel)
        if kernel_nsub > 1:
            # Interpolate onto the data grid:
            profile_obs = np.interp(x, profile_wl, profile_broad)
        else:
            profile_obs = profile_broad

    return profile_obs


def evaluate_optical_depth(profile_wl, pars, lines, z_sys=None):
    """
    Evaluate optical depth based on the component parameters in `pars`.

    Parameters
    ----------
    profile_wl : array_like, shape (N)
        Wavelength array in Ångstrøm on which to evaluate the profile.

    pars : dict(lmfit.Parameters_)
        An instance of lmfit.Parameters_ containing the line parameters.

    lines : list(:class:`Line <dataset.Line>`)
        List of lines to evaluate. Should be a list of
        :class:`Line <dataset.Line>` objects.

    z_sys : float or None
        The systemic redshift, used to determine an effective wavelength range
        within which to evaluate the profile. This is handy when fitting very large
        regions, such as HI and metal lines together.

    Returns
    -------
    tau : array_like, shape (N)
        The resulting optical depth of all `lines` in the wavelength region.
    """
    tau = np.zeros_like(profile_wl)

    if z_sys is not None:
        # Determine range in which to evaluate the profile:
        max_logN = max([val.value for key, val in pars.items() if 'logN' in key])
        if max_logN > 19.0:
            velspan = 20000.*(1. + 1.0*(max_logN-19.))
        else:
            velspan = 20000.

    # Determine number of components for each ion:
    components_per_ion = {}
    for line in lines:
        if line.active:
            l0, f, gam = line.get_properties()
            ion = line.ion
            z_pars = []
            for parname in pars.keys():
                parts = parname.split('_')
                if len(parts) == 2:
                    pid, p_ion = parts
                    if 'z' in pid and p_ion == ion:
                        z_pars.append(parname)
            components_per_ion[ion] = len(z_pars)

    for line in lines:
        if line.active:
            l0, f, gam = line.get_properties()
            ion = line.ion
            n_comp = components_per_ion[ion]
            if z_sys is not None:
                l_center = l0*(z_sys + 1.)
                vel = (profile_wl - l_center)/l_center*299792.458
                span = (vel >= -velspan)*(vel <= velspan)
            else:
                span = np.ones_like(profile_wl, dtype=bool)
            ion = ion.replace('*', 'x')
            for n in range(n_comp):
                z = pars['z%i_%s' % (n, ion)].value
                if profile_wl.min() < l0*(z+1) < profile_wl.max():
                    b = pars['b%i_%s' % (n, ion)].value
                    logN = pars['logN%i_%s' % (n, ion)].value
                    tau[span] += Voigt(profile_wl[span], l0, f, 10**logN, 1.e5*b, gam, z=z)
                elif ion == 'HI':
                    b = pars['b%i_%s' % (n, ion)].value
                    logN = pars['logN%i_%s' % (n, ion)].value
                    tau[span] += Voigt(profile_wl[span], l0, f, 10**logN, 1.e5*b, gam, z=z)
                else:
                    continue
    return tau


def resvel_to_pixels(wl, res):
    """
    Convert spectral resolution in km/s to pixels

    Parameters
    ----------
    wl : array, shape (N)
        Input array of wavelength to evaluate

    res : float
        The input spectral velocity resolution in **km/s**.

    Returns
    -------
    w : float
        Kernel width converted to velocity pixels (km/s).
        (to be used with convolution of logarithmically-spaced data/models).
    """
    dl = np.diff(wl)
    pix_size_vel = (dl[-1] - dl[0]) / (wl[-1] - wl[0]) * 299792.458
    assert pix_size_vel > 0.0001, "Wavelength array seems to be linearly spaced. Must be logarithmic!"
    w = res / pix_size_vel / 2.35482
    return w


def fwhm_to_pixels(wl, res_wl):
    """
    Convert spectral resolution in wavelength to pixels
    R = lambda / dlambda, where dlambda is the FWHM

    Parameters
    ----------
    wl : array, shape (N)
        Input array of wavelength to evaluate

    res_wl : float
        The input spectral resolution element (FWHM) in Ångström.

    Returns
    -------
    w : float
        Kernel width converted to pixels (to be used with :func:`convolve_profile <voigt.convolve_profile>`).
    """
    dl = np.diff(wl)
    pix_size_vel = (dl[-1] - dl[0]) / (wl[-1] - wl[0]) * 299792.458
    assert pix_size_vel < 0.0001, "Wavelength array seems not to be linearly spaced. Must be linear!"
    w = res_wl / dl[0] / 2.35482
    return w


def convolve_profile(profile, width):
    """
    Convolve with Gaussian kernel using constant width in pixels!

    Parameters
    ----------
    profile : array, shape (N)
        Input profile to convolve with a Gaussian kernel.

    width : float
        Kernel width (or sigma) of the Gaussian profile in **units of pixels**.
        Note -- this should *not* be the FWHM, as is usually used to denote
        the resolution element: `width` = FWHM / 2.35482

    Returns
    -------
    profile_obs : array, shape(N)
        The convolved version of `profile`.
    """
    LSF = gaussian(10*int(width)+1, width)
    LSF = LSF/LSF.sum()
    # Add padding to avoid edge effects of the convolution:
    pad = np.ones(5*int(width))
    P_padded = np.concatenate((pad, profile, pad))
    profile_broad = fftconvolve(P_padded, LSF, 'same')
    # Remove padding:
    profile_obs = profile_broad[5*int(width):-5*int(width)]
    return profile_obs
