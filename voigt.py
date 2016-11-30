# -*- coding: utf-8 -*-
#   Voigt Line Fit
#
#   module to evaluate line profile

import numpy as np, matplotlib.pyplot as plt
from scipy.signal import fftconvolve, gaussian
import pickle

#==== VOIGT PROFILE ===============
def H(a, x):
	P  = x**2
	H0 = np.exp(-x**2)
	Q  = 1.5/x**2
	return H0 - a/np.sqrt(np.pi)/P * (H0*H0*(4.*P*P + 7.*P + 4. + Q) - Q - 1)

def Voigt(l, l0, f, N, b, gam, z=0):
	"""Calculate the Voigt profile of transition with
	rest frame transition wavelength: 'l0'
	oscillator strength: 'f'
	column density: N  cm^-2
	velocity width: b  cm/s
	"""
	#==== PARAMETERS ==================

	c  = 2.99792e10		#cm/s
	m_e= 9.1095e-28		#g
	e  = 4.8032e-10		#cgs units
	
	#==================================
	#Calculate Profile
	
	C_a  = np.sqrt(np.pi)*e**2*f*l0*1.e-8/m_e/c/b
	a    = l0*1.e-8*gam/(4.*np.pi*b)
	
	dl_D = b/c*l0
	l = l/(z+1.)
	x = (l - l0)/dl_D+0.0001
	
	tau  = np.float64(C_a)*N*H(a,x)
	#return np.exp(-tau)
	return tau


def evaluate_profile(x, pars, z_sys, lines, components, res, npad, nsamp=1):
	"""
	Function to evaluate Voigt profile for a fitting `Region'.

	Parameters
	----------
	x : np.array
		Wavelength array to evaluate the profile on

	pars : dictionary
		Dictionary containing fit parameters from `lmfit'
	
	z_sys : float
		Systemic redshift, as defined in DataSet.redshift

	lines : list
		List of lines in the region to evaluate.
		Should be a list of `Line' instances.

	components : dictionary
		Dictionary containing component data for the defined ions.
	
	res : float
		Spectral resolution of the region in km/s.
	
	npad : integer
		number of pixels added before and after the wavelength array

	>>nsamp : integer<< not active
		Resampling factor. If different from 1 the input array `x'
		will be resampled by this factor.

	Returns
	-------
	profile_obs : np.array
		Total observed line profile of all lines in the region,
		convolved with the instrument Line Spread Function.
	"""

	profile_wl = x
	pxs = np.diff(x)[0]

	### Add padding on each side of the evaluated profile
	### to avoid boundary artefacts during convolution.
	front_padding = np.linspace(x.min()-npad*pxs, x.min(), npad, endpoint=False)
	end_padding = np.linspace(x.max()+pxs, x.max()+npad*pxs, npad)
	profile_wl = np.concatenate([front_padding, profile_wl, end_padding])
	tau = np.zeros_like(profile_wl)

	for line in lines:
		if line.active:
			l0, f, gam = line.get_properties()
			l_center = l0*(z_sys + 1.)
			ion = line.ion
			n_comp = len(components[ion])
			
			ion = ion.replace('*','x')
			for n in range(n_comp):
				z = pars['z%i_%s'%(n, ion)].value
				b = pars['b%i_%s'%(n, ion)].value
				logN = pars['logN%i_%s'%(n, ion)].value
				tau += Voigt(profile_wl, l0, f, 10**logN, 1.e5*b, gam, z=z)

	profile = np.exp(-tau)
	### Calculate Line Spread Function, i.e., instrumental broadening:
	### The resolution FWHM is converted into gaussian sigma in units
	### of number of pixels
	fwhm_instrumental = res/299792.*l_center		# in units of wavelength
	sigma_instrumental = fwhm_instrumental/2.35482/pxs	# in units of pixels
	LSF = gaussian(len(profile_wl), sigma_instrumental)
	LSF = LSF/LSF.sum()
	profile_broad = fftconvolve(profile, LSF, 'same')

	### Rebin back to spectral pixel size:
	#if nsamp>1:
	#	profile_obs = np.interp(x, profile_wl, profile_broad)

	### Remove padding which includes the boundary effects of convolution:
	profile_obs = profile_broad[npad:-npad]

	return profile_obs
