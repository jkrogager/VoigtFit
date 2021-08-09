import numpy as np
import VoigtFit

### Fit DLA towards quasar Q1313+1441
### Observed in X-shooter P089.A-0068

z_DLA = 1.7941
logNHI = 21.3, 0.1		# value, uncertainty

# If log(NHI) is not known use:
#logNHI = None

#### Load UVB and VIS data:
UVB_fname = 'data/test_UVB_1d.spec'
res_UVB = 8000
VIS_fname = 'data/test_VIS_1d.spec'
res_VIS = 11800

wl_uvb, spec_uvb, err_uvb = np.loadtxt(UVB_fname, unpack=True)
wl_vis, spec_vis, err_vis = np.loadtxt(VIS_fname, unpack=True)

# Alternatively, load a FITS spectrum (either a FITS table or array):
# wl, flux, err, mask, header = VoigtFit.io.load_fits_spectrum(fname)


dataset = VoigtFit.DataSet(z_DLA)
dataset.add_data(wl_uvb, spec_uvb, 299792./res_UVB, err=err_uvb, normalized=False)
dataset.add_data(wl_vis, spec_vis, 299792./res_VIS, err=err_vis, normalized=False)

### Define absorption lines:
dataset.add_line('FeII_2374')
dataset.add_line('FeII_2260')
dataset.add_line('CrII_2056')
dataset.add_line('CrII_2066')
dataset.add_line('CrII_2026')
dataset.add_line('ZnII_2026')
dataset.add_line('MgI_2026')
dataset.add_line('MgI_2852')



### If a line has been defined, and you don't want to fit it
### it can either be removed from the dataset completely:
#dataset.remove_line('CrII_2056')

### or deactivated:
#dataset.deactivate_line('FeII_2374')

### A deactivated line is still present in the dataset,
### but not included in the fit. The line may still show up in the final figure.

### Define components to fit:
# dataset.reset_components()

### Add velocity components for each ion:
#                      ion    z         b   logN
dataset.add_component('FeII', 1.793532, 20, 14.3, var_z=1)
dataset.add_component('FeII', 1.794060, 20, 15.0, var_z=1)
dataset.add_component('FeII', 1.794282, 20, 14.3, var_z=1)
dataset.add_component('FeII', 1.794722, 20, 14.3, var_z=1)
dataset.add_component('FeII', 1.795121, 15, 14.5, var_z=1, var_b=1)
#
# Options for the components:
# var_z=1/0 vary redshift for this component
# var_b=1/0 vary b-parameter for this component
# var_N=1/0 vary column density for this component
#
# Redshift and b-parameters can be tied.
# passing the option 'tie_z=z0_FeII' ties the redshift to the first component of FeII
# passing the option 'tie_b=b2_SiII' ties the b-parameter to the third component of SiII
#
# NOTE - the ion must be defined and the component index starts with 0
#
# The entire velocity structure can be copied from one ion to another:
dataset.copy_components(from_ion='FeII', to_ion='ZnII', logN=12.9, ref_comp=1)
# This copies the five components defined for FeII to ZnII and keeps
# the same pattern of initial guesses for column density.
# By giving ref_comp and logN, this intial guess pattern is scaled such
# that the second component has logN=12.9
#
# Individual components which are not observed for weaker lines can be removed:
#dataset.delete_component('ZnII', 4)	# the index '4' refers to the fifth component
#dataset.delete_component('ZnII', 3)
#dataset.delete_component('ZnII', 2)
#dataset.delete_component('ZnII', 1)
#dataset.delete_component('ZnII', 0)
# NOTE - components should be deleted from last component to first component
#        not the other way around as that messes up the component numbering.

dataset.copy_components(to_ion='CrII', from_ion='FeII', logN=13.6, ref_comp=1)
dataset.copy_components(to_ion='MgI', from_ion='FeII', logN=12.4, ref_comp=1)

# Crucial step:
dataset.prepare_dataset()

# Run the fit:
popt, chi2 = dataset.fit()

# Output best-fit parameters, total column densities and make plot:
dataset.plot_fit()
if logNHI:
    dataset.print_metallicity(*logNHI)
dataset.print_total()

### The best-fit parameters can be accessed from the .best_fit attribute:
#logN0 = dataset.best_fit['logN0_FeII'].value
#logN0_err = dataset.best_fit['logN0_FeII'].stderr
#b1 = dataset.best_fit['b1_FeII'].value
#b1_err = dataset.best_fit['b1_FeII'].stderr

# Or you can create a list of all values:
#logN_FeII = [dataset.best_fit['logN%i_FeII' % num].value for num in range(len(dataset.components['FeII']))]
#logN_err_FeII = [dataset.best_fit['logN%i_FeII' % num].stderr for num in range(len(dataset.components['FeII']))]

dataset.save('example_fit.hdf5')

### The dataset which was defined above can be loaded like this:
# dataset = VoigtFit.load_dataset('example_fit.hdf5')
