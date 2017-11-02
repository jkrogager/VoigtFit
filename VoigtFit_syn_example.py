import numpy as np
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import VoigtFit
import pickle

### Fit DLA towards quasar Q1313+1441
### Observed in X-shooter P089.A-0068

z_DLA = 2.3538
# logNHI = 21.3, 0.1		# value, uncertainty

# If log(NHI) is not known use:
logNHI = None

#### Load UVB and VIS data:
fname = 'data/synthetic_testdata/synspec.dat'
res = 10000

wl, spec = np.loadtxt(fname, unpack=True)

dataset = VoigtFit.DataSet(z_DLA)
dataset.add_data(wl, spec, 299792./res, err=spec/10, normalized=True)

### Define absorption lines:
# dataset.add_line('SiII_1808', velspan=500)
# dataset.add_line('SiII_1526', velspan=500)
# dataset.add_line('SiII_1304', velspan=500)
# dataset.add_line('SiII_1260')
dataset.add_line('FeII_2344', velspan=500)
dataset.add_line('FeII_2374', velspan=500)
dataset.add_line('FeII_2382', velspan=500)
dataset.add_line('FeII_2249')
dataset.add_line('FeII_2260')
# dataset.add_line('AlII_1670')
# dataset.add_line('AlIII_1854')
# dataset.add_line('AlIII_1862')
# dataset.add_line('SII_1259')
# dataset.add_line('SII_1253')
# dataset.add_line('SII_1250')
# dataset.add_line('OI_1302')
# dataset.add_line('ZnII_2026')
# dataset.add_line('ZnII_2062')
# dataset.add_line('CrII_2056')
# dataset.add_line('CrII_2062')
# dataset.add_line('CrII_2066')
# dataset.add_line('MgI_1668')
# dataset.add_line('MgI_2026')
# dataset.add_line('MgI_2852')
# dataset.add_line('MgI_1827')
# dataset.add_line('CII_1334')
# dataset.add_line('CI_1560')
# dataset.add_line('CI_1656')
# dataset.add_line('HI_1215', velspan=30000)
# dataset.add_line('HI_1025', velspan=2000)

### This command prepares the line regions:
# First the data are interactively normalized
# Then regions which should not be fitted are masked interactively too
# dataset.prepare_dataset(mask = False)

# Save the dataset so you don't have to normalize and mask every time:
VoigtFit.SaveDataSet('synthetic.dataset', dataset)

### The dataset which was defined above can be loaded like this:
# In this case, comment out lines 18-41
# dataset = VoigtFit.LoadDataSet('test.dataset')


### If a line has been defined, and you don't want to fit it
### it can either be removed from the dataset completely:
#dataset.remove_line('CrII_2056')

### or deactivated:
#dataset.deactivate_line('FeII_2374')

# dataset.reset_components()

### Add velocity components for each ion:
#                      ion    z         b   logN



dataset.add_component('FeII', 2.352279, 25, 14.39)
dataset.add_component('FeII', 2.352719, 25, 14.17)
dataset.add_component('FeII', 2.353354, 25, 14.34)
dataset.add_component('FeII', 2.353672, 25, 14.96)
dataset.add_component('FeII', 2.354357, 25, 14.89)
dataset.add_component('FeII', 2.355017, 25, 14.35)

# dataset.add_component('HI', 2.3535, 20, 22.0, var_b=False)
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
# dataset.copy_components('ZnII', 'FeII', logN=12.9, ref_comp=1)

# dataset.copy_components('SiII', 'FeII', logN=12.9, ref_comp=1)
# dataset.copy_components('AlII', 'FeII', logN=12.9, ref_comp=1)
# dataset.copy_components('AlIII', 'FeII', logN=12.9, ref_comp=1)
# dataset.copy_components('SII', 'FeII', logN=12.9, ref_comp=1)
# dataset.copy_components('OI', 'FeII', logN=12.9, ref_comp=1)
# dataset.copy_components('ZnII', 'FeII', logN=12.9, ref_comp=1)
# dataset.copy_components('CrII', 'FeII', logN=12.9, ref_comp=1)
# dataset.copy_components('MgI', 'FeII', logN=12.9, ref_comp=1)
# dataset.copy_components('CII', 'FeII', logN=12.9, ref_comp=1)
# dataset.copy_components('CI', 'FeII', logN=12.9, ref_comp=1)

# Missing N = {'MnII':13.2,'NV':14.8, 'CIV':12.8, 'NiII': 14.2, 'SiIV': 12.3, 'MgII':18.0}



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

# dataset.copy_components('CrII', 'FeII', logN=13.6, ref_comp=1)
# dataset.copy_components('MgI', 'FeII', logN=12.4, ref_comp=1)

dataset.prepare_dataset(mask=False)

popt, chi2 = dataset.fit(verbose=True, cheb_order=-1)

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
