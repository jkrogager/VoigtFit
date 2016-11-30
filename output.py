import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, gaussian
import numpy as np
import itertools
import voigt
import Asplund

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


### ===================================================================================
###
###   Graphics output functions:
### ------------------------------

def plot_fit(dataset, params, active_only=True, linestyles=[':'], colors=['b']):
	"""
	Plot best fit absorption profiles.
	
	  INPUT:
	dataset:  VoigtFit.DataSet instance containing the line regions
	params: Output parameter dictionary from VoigtFit.DataSet.fit()
	
	linestyles:  a list of linestyles to show velocity components
	colors:      a lost of colors to show the velocity components
	
	The colors and linestyles are combined to form an `iterator'
	which cycles through a set of (linestyle, color).
	"""

	plt.close('all')
	for region in dataset.regions:
		if (active_only and region.has_active_lines()) or (not active_only):
			x, y, err, mask = region.unpack()
			cont_err = region.cont_err
			res = region.res

			plt.figure()
			pxs = 0.1
			wl_line = np.arange(x.min(), x.max(), pxs)
			ref_line = region.lines[-1]
			l0, f, gam = ref_line.get_properties()
			l_ref = l0*(dataset.redshift + 1)
			npad = 20
			nsamp = 1

			front_padding = np.linspace(wl_line.min()-npad*pxs, wl_line.min(), npad, endpoint=False)
			end_padding = np.linspace(wl_line.max()+pxs, wl_line.max()+npad*pxs, npad)
			wl_line = np.concatenate([front_padding, wl_line, end_padding])
			tau = np.zeros_like(wl_line)

			for line in region.lines:
				if (active_only and line.active) or (not active_only):
					# Reset line properties for each element
					component_prop = itertools.product(linestyles, colors)
					component_prop = itertools.cycle(component_prop)

					# Load line properties
					l0, f, gam = line.get_properties()
					ion = line.ion
					if ion in dataset.components.keys():
						n_comp = len(dataset.components[ion])

						ion = ion.replace('*','x')
						for n in range(n_comp):
							z = params['z%i_%s'%(n, ion)].value
							b = params['b%i_%s'%(n, ion)].value
							logN = params['logN%i_%s'%(n, ion)].value
							tau += voigt.Voigt(wl_line, l0, f, 10**logN, 1.e5*b, gam, z=z)
							
							ls, color = component_prop.next()
							plt.axvline((l0*(z+1)-l_ref)/l_ref*299792., ls=ls, color=color)

			profile_int = np.exp(-tau)
			fwhm_instrumental = res/299792.*l_ref
			sigma_instrumental = fwhm_instrumental/2.35482/pxs
			LSF = gaussian(len(wl_line), sigma_instrumental)
			LSF = LSF/LSF.sum()
			profile_broad = fftconvolve(profile_int, LSF, 'same')
			profile = profile_broad[npad:-npad]
			wl_line = wl_line[npad:-npad]
			vel_profile = (wl_line-l_ref)/l_ref*299792.

			plt.ylim(profile.min()-0.1, 1.15)
			plt.xlim(-region.velocity_span, region.velocity_span)
			
			vel  = (x-l_ref)/l_ref*299792.
	
			# Expand mask by 1 pixel around each masked range
			# to draw the lines correctly
			mask_idx = np.where(mask==0)[0]
			big_mask_idx = np.union1d(mask_idx+1, mask_idx-1)
			big_mask = np.ones_like(mask, dtype=bool)
			big_mask[big_mask_idx]=False
			masked_range = np.ma.masked_where(big_mask, y)
			plt.plot(vel, masked_range, color='0.7', drawstyle='steps-mid', lw=0.9)

			spectrum = np.ma.masked_where(~mask, y)
			error = np.ma.masked_where(~mask, err)
			plt.errorbar(vel, spectrum, err, ls='', color='gray')
			plt.plot(vel, spectrum, color='k', drawstyle='steps-mid')

			plt.plot(vel_profile, profile, color='r', lw=1.5)

			plt.xlabel("Velocity  [${\\rm km\,s^{-1}}$]")
			plt.ylabel("Normalized flux")
			plt.axhline(1., ls='--', color='k')
			plt.axhline(1.+cont_err, ls=':', color='gray')
			plt.axhline(1.-cont_err, ls=':', color='gray')
			
			title_string = ", ".join([line.tag for line in region.lines])
			plt.title(title_string)


def plot_line(dataset, line_tag, plot_fit=True, linestyles=[':'], colors=['b']):
	"""
	Plot absorption line.
	
	  INPUT:
	dataset:  VoigtFit.DataSet instance containing the line regions
	line_tag: The line tag of the line to show, e.g., 'FeII_2374'

	plot_fit:    if True, the best-fit profile will be shown
	linestyles:  a list of linestyles to show velocity components
	colors:      a lost of colors to show the velocity components
	
	The colors and linestyles are combined to form an `iterator'
	which cycles through a set of (linestyle, color).
	"""

	if line_tag not in dataset.all_lines:
		plt.figure()
		dataset.add_line(line_tag, active=False)
		dataset.prepare_dataset()
	
	region = dataset.find_line(line_tag)

	x, y, err, mask = region.unpack()
	cont_err = region.cont_err
	res = region.res

	plt.figure(figsize=(6,3.75))
	plt.subplots_adjust(bottom=0.15, right=0.98)

	if plot_fit and isinstance(dataset.best_fit, dict):
		pxs = 0.1
		wl_line = np.arange(x.min(), x.max(), pxs)
		ref_line = region.lines[-1]
		l0, f, gam = ref_line.get_properties()
		l_ref = l0*(dataset.redshift + 1)
		npad = 20
		nsamp = 1

		front_padding = np.linspace(wl_line.min()-npad*pxs, wl_line.min(), npad, endpoint=False)
		end_padding = np.linspace(wl_line.max()+pxs, wl_line.max()+npad*pxs, npad)
		wl_line = np.concatenate([front_padding, wl_line, end_padding])
		tau = np.zeros_like(wl_line)

		params = dataset.best_fit
		for line in region.lines:
			# Reset line properties for each element
			component_prop = itertools.product(linestyles, colors)
			component_prop = itertools.cycle(component_prop)

			# Load line properties
			l0, f, gam = line.get_properties()
			ion = line.ion
			n_comp = len(dataset.components[ion])

			ion = ion.replace('*','x')
			for n in range(n_comp):
				z = params['z%i_%s'%(n, ion)].value
				b = params['b%i_%s'%(n, ion)].value
				logN = params['logN%i_%s'%(n, ion)].value
				tau += voigt.Voigt(wl_line, l0, f, 10**logN, 1.e5*b, gam, z=z)
				
				ls, color = component_prop.next()
				plt.axvline((l0*(z+1)-l_ref)/l_ref*299792., ls=ls, color=color)

		profile_int = np.exp(-tau)
		fwhm_instrumental = res/299792.*l_ref
		sigma_instrumental = fwhm_instrumental/2.35482/pxs
		LSF = gaussian(len(wl_line), sigma_instrumental)
		LSF = LSF/LSF.sum()
		profile_broad = fftconvolve(profile_int, LSF, 'same')
		profile = profile_broad[npad:-npad]
		wl_line = wl_line[npad:-npad]
		vel_profile = (wl_line-l_ref)/l_ref*299792.
	
	plt.ylim(y[mask].min()-0.1)
	plt.xlim(-region.velocity_span, region.velocity_span)
	
	vel  = (x-l_ref)/l_ref*299792.
	
	# Expand mask by 1 pixel around each masked range
	# to draw the lines correctly
	mask_idx = np.where(mask==0)[0]
	big_mask_idx = np.union1d(mask_idx+1, mask_idx-1)
	big_mask = np.ones_like(mask, dtype=bool)
	big_mask[big_mask_idx]=False
	masked_range = np.ma.masked_where(big_mask, y)
	plt.plot(vel, masked_range, color='0.7', drawstyle='steps-mid', lw=0.9)

	spectrum = np.ma.masked_where(~mask, y)
	error = np.ma.masked_where(~mask, err)
	plt.errorbar(vel, spectrum, err, ls='', color='gray')
	plt.plot(vel, spectrum, color='k', drawstyle='steps-mid')
	plt.axhline(0., ls='--', color='0.7', lw=0.7)

	if plot_fit:
		plt.plot(vel_profile, profile, color='r', lw=1.5)
		plt.ylim(profile.min()-0.1, 1.15)

	plt.xlabel("Velocity  [${\\rm km\,s^{-1}}$]")
	plt.ylabel("Normalized flux")
	plt.axhline(1., ls='--', color='k')
	plt.axhline(1.+cont_err, ls=':', color='gray')
	plt.axhline(1.-cont_err, ls=':', color='gray')
	
	title_string = ", ".join([line.tag for line in region.lines])
	plt.title(title_string)
	plt.show()


### ===================================================================================
###
###   Text output functions:
### --------------------------

def print_results(dataset, params, elements='all', velocity=True, systemic=0):
	"""
	Plot best fit absorption profiles.
	
	  INPUT:
	dataset:  VoigtFit.DataSet instance containing the line regions
	params: Output parameter dictionary from VoigtFit.DataSet.fit()

	if velocity is set to 'False' the components redshift is shown instead
	of the velocity relative to the systemic redshift.

	if systemic is set the velocities will be relative to this redshift;
	default behaviour is to use the systemic redshift defined in the dataset.
	
	"""

	if systemic:
		z_sys = systemic
	else:
		z_sys = dataset.redshift

	print "\n  Best fit parameters\n"
	print "\t\t\t\tlog(N)\t\t\tb"
	if elements=='all':
		for ion in dataset.components.keys():
			lines_for_this_ion = []
			for line_tag, line in dataset.lines.items():
				if line.ion == ion and line.active:
					lines_for_this_ion.append(line_tag)

			all_transitions = [trans.split('_')[1] for trans in sorted(lines_for_this_ion)]
			### Split list of transitions into chunks of length=4
			### join the transitions in each chunks
			### and join each chunk with 'newline'
			trans_chunks = [", ".join(sublist) for sublist in list(chunks(all_transitions, 4))]
			indent = '\n'+(len(ion)+2)*' '
			trans_string = indent.join(trans_chunks)

			print ion + "  "+trans_string
			n_comp = len(dataset.components[ion])
			for n in range(n_comp):
				ion = ion.replace('*','x')
				z = params['z%i_%s'%(n, ion)].value
				b = params['b%i_%s'%(n, ion)].value
				logN = params['logN%i_%s'%(n, ion)].value
				z_err = params['z%i_%s'%(n, ion)].stderr
				b_err = params['b%i_%s'%(n, ion)].stderr
				logN_err = params['logN%i_%s'%(n, ion)].stderr

				if velocity:
					z_std = z_err/(z_sys+1)*299792.
					z_val = (z-z_sys)/(z_sys+1)*299792.
					z_format = "v = %5.1f +/- %.1f\t"
				else:
					z_std = z_err
					z_val = z
					z_format = "z = %.6f +/- %.6f"

				output_string = z_format%(z_val,z_std)+"\t\t"
				output_string += "%.2f +/- %.2f\t\t"%(logN,logN_err)
				output_string += "%.1f +/- %.1f"%(b,b_err)

				print output_string

			print ""

	else:
		for ion in elements:
			lines_for_this_ion = []
			for line_tag, line in dataset.lines.items():
				if line.ion == ion and line.active:
					lines_for_this_ion.append(line_tag)

			all_transitions = ", ".join([trans.split('_')[1] for trans in sorted(lines_for_this_ion)])
			print ion + "  "+all_transitions
			n_comp = len(dataset.components[ion])
			for n in range(n_comp):
				ion = ion.replace('*','x')
				z = params['z%i_%s'%(n, ion)].value
				b = params['b%i_%s'%(n, ion)].value
				logN = params['logN%i_%s'%(n, ion)].value
				b_err = params['b%i_%s'%(n, ion)].stderr
				logN_err = params['logN%i_%s'%(n, ion)].stderr

				if velocity:
					z_val = (z-z_sys)/(z_sys+1)*299792.
					z_format = "v = %5.1f\t"
				else:
					z_val = z
					z_format = "z = %.6f"

				output_string = z_format%(z_val,z_std)+"\t\t"
				output_string += "%.2f +/- %.2f\t\t"%(logN,logN_err)
				output_string += "%.1f +/- %.1f"%(b,b_err)

				print output_string

			print ""


def print_metallicity(dataset, params, logNHI, err=0.1):
	"""
	Plot best fit absorption profiles.
	
	  INPUT:
	dataset:  VoigtFit.DataSet instance containing the line regions
	params: Output parameter dictionary from VoigtFit.DataSet.fit()
	logNHI:   Column density of neutral hydrogen
	err:      Uncertainty on logNHI
	
	"""

	print "\n  Metallicities\n"
	print "  log(NHI) = %.2f +/- %.2f\n" % (logNHI, err)
	logNHI = np.random.normal(logNHI, err, 10000)
	for ion in dataset.components.keys():
		element = ion[:2] if ion[1].islower() else ion[0]
		logN = []
		logN_err = []
		N_tot = []
		for par in params.keys():
		    if par.find('logN')>=0 and par.find(ion)>=0:
		        N_tot.append(params[par].value)
		        if params[par].stderr < 0.9:
		            logN.append(params[par].value)
		            if params[par].stderr < 0.01:
		                logN_err.append(0.01)
		            else:
		                logN_err.append(params[par].stderr)
		    
		ION = [np.random.normal(n,e,10000) for n,e in zip(logN, logN_err)]
		l68, abundance, u68 = np.percentile(np.log10(np.sum(10**np.array(ION),0)),[16, 50, 84])
		std_err = np.std(np.log10(np.sum(10**np.array(ION),0)))
		
		logN_tot = np.random.normal(abundance, std_err, 10000)
		N_solar, N_solar_err = Asplund.photosphere[element]
		solar_abundance = np.random.normal(N_solar, N_solar_err, 10000)
		
		metal_array = abundance - logNHI - (solar_abundance - 12.)
		metal = np.mean(metal_array)
		metal_err = np.std(metal_array)
		print "  [%s/H] = %.2f +/- %.2f" %(element, metal, metal_err)

def print_abundance(dataset):
	"""
	Plot best fit absorption profiles.
	"""

	if isinstance(dataset.best_fit, dict):
		params = dataset.best_fit
		print "\n  Total Abundances\n"
		for ion in dataset.components.keys():
			element = ion[:2] if ion[1].islower() else ion[0]
			logN = []
			logN_err = []
			N_tot = []
			for par in params.keys():
			    if par.find('logN')>=0 and par.find(ion)>=0:
			        N_tot.append(params[par].value)
			        if params[par].stderr < 0.9:
			            logN.append(params[par].value)
			            if params[par].stderr < 0.01:
			                logN_err.append(0.01)
			            else:
			                logN_err.append(params[par].stderr)
			    
			ION = [np.random.normal(n,e,10000) for n,e in zip(logN, logN_err)]
			l68, abundance, u68 = np.percentile(np.log10(np.sum(10**np.array(ION),0)),[16, 50, 84])
			std_err = np.std(np.log10(np.sum(10**np.array(ION),0)))
			
			print "  logN(%s) = %.2f +/- %.2f" %(ion, abundance, std_err)

	else:
		print "\n [ERROR] - The dataset has not yet been fitted. No parameters found!"
