import numpy as np
import matplotlib.pyplot as plt
import VoigtFit
import pickle

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

### This command prepares the line regions:
# First the data are interactively normalized
# Then regions which should not be fitted are masked interactively too
dataset.prepare_dataset()

# Save the dataset so you don't have to normalize and mask every time:
VoigtFit.SaveDataSet('test.dataset', dataset)

### The dataset which was defined above can be loaded like this:
# In this case, comment out lines 18-41
#dataset = VoigtFit.LoadDataSet('test.dataset')


### If a line has been defined, and you don't want to fit it
### it can either be removed from the dataset completely:
#dataset.remove_line('CrII_2056')

### or deactivated:
#dataset.deactivate_line('FeII_2374')

dataset.reset_components()

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
dataset.copy_components('ZnII', 'FeII', logN=12.9, ref_comp=1)
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

dataset.copy_components('CrII', 'FeII', logN=13.6, ref_comp=1)
dataset.copy_components('MgI', 'FeII', logN=12.4, ref_comp=1)

dataset.prepare_dataset()

popt, chi2 = dataset.fit(verbose=True)

dataset.plot_fit()
if logNHI:
	dataset.print_metallicity(*logNHI)
dataset.print_abundance()


#### Remove parameter links
#### The links may result in error when loadning the parameters later.

for par in popt.params.values():
	par.expr = None
for par in dataset.pars.values():
	par.expr = None

pickle.dump(popt.params, open('example_best_fit.pars','w'))
VoigtFit.SaveDataSet('example_fit.dataset', dataset)
