import numpy as np
import VoigtFit

# -- Fit DLA towards quasar Q1313+1441
# Observed with X-shooter in P089.A-0068

z_DLA = 1.7941
logNHI = 21.3, 0.1        # value, uncertainty

# -- If log(NHI) is not known use:
# logNHI = None

# -- Load X-SHOOTER UVB and VIS data in ASCII format:
UVB_fname = '../data/test_UVB_1d.spec'
res_UVB = 8000.
VIS_fname = '../data/test_VIS_1d.spec'
res_VIS = 11800.

wl_uvb, spec_uvb, err_uvb = np.loadtxt(UVB_fname, unpack=True)
wl_vis, spec_vis, err_vis = np.loadtxt(VIS_fname, unpack=True)

# -- Here you can load your data in any way you wish
# -- Only requirement is that wl, spec, and err have the same dimensions.

# -- A dataset which has already been defined can be loaded like this:
# dataset = VoigtFit.LoadDataSet('test_data.hdf5')

dataset = VoigtFit.DataSet(z_DLA)
dataset.set_name('Q1313+1441')
dataset.verbose = True
dataset.cheb_order = 2

# -- Add the data loaded from the ASCII table
dataset.add_data(wl_uvb, spec_uvb, 299792.458/res_UVB, err=err_uvb, normalized=False)
dataset.add_data(wl_vis, spec_vis, 299792.458/res_VIS, err=err_vis, normalized=False)

# -- Define absorption lines to fit:
dataset.add_line('FeII_2374')
dataset.add_line('FeII_2260')
dataset.add_line('CrII_2056')
dataset.add_line('CrII_2066')
dataset.add_line('CrII_2026')
dataset.add_line('ZnII_2026')
dataset.add_lines(['MgI_2026', 'MgI_2852'])


# -- If a line has been defined, and you don't want to fit it
#    it can either be removed from the dataset completely:
# dataset.remove_line('CrII_2056')

# -- or deactivated:
# dataset.deactivate_line('FeII_2374')

# -- Deactivated lines will not be included in the fit, but their line
#    definitions and components remain in the dataset for future reference.

# -- If a dataset is loaded from file, it is a good idea to clear the component
#    definitions before adding new components:
dataset.reset_components()

# -- Add components for each ion:
#                      ion    z         b   logN
dataset.add_component('FeII', 1.793532, 20, 14.3, var_z=1)
dataset.add_component('FeII', 1.794060, 20, 15.0, var_z=1)
dataset.add_component('FeII', 1.794282, 20, 14.3, var_z=1)
dataset.add_component('FeII', 1.794722, 20, 14.3, var_z=1)
dataset.add_component('FeII', 1.795121, 15, 14.5, var_z=1, var_b=1)

# -- Options for the components:
# var_z=1/0 vary redshift for this component
# var_b=1/0 vary b-parameter for this component
# var_N=1/0 vary column density for this component
#
# Redshift and b-parameters can be tied.
# passing the option tie_z='z0_FeII' ties the redshift to the first component of FeII
# passing the option tie_b='b2_SiII' ties the b-parameter to the third component of SiII
#
# NOTE - the ion must be defined and the component index starts with 0


# -- The entire velocity structure can be copied from one ion to another:
dataset.copy_components('ZnII', 'FeII', logN=12.9, ref_comp=1)

# -- This copies the five components defined for FeII to ZnII and keeps
#    the same pattern of initial guesses for column density.
#    By giving ref_comp and logN, this intial guess pattern is scaled such
#    that the second component has logN=12.9

# -- Individual components which are not observed for weaker lines can be removed:
# dataset.delete_component('ZnII', 4)   # this index refers to the fifth component
# dataset.delete_component('ZnII', 3)
# dataset.delete_component('ZnII', 2)
# dataset.delete_component('ZnII', 1)
# dataset.delete_component('ZnII', 0)
#
# NOTE - components should be deleted from last component to first component
#        not the other way around as that messes up the component numbering.

dataset.copy_components('CrII', 'FeII', logN=13.6, ref_comp=1)
dataset.copy_components('MgI', 'FeII', logN=12.4, ref_comp=1)

# -- Prepare the dataset: This will prompt the user for interactive
#    masking and normalization, as well as initiating the Parameters:
dataset.prepare_dataset(norm=False, mask=False)

# -- Define masks for individual lines:
dataset.mask_line('ZnII_2026')

# -- Fit the dataset:
popt, chi2 = dataset.fit()

dataset.plot_fit()

# -- Print total column densities
dataset.print_total()

if logNHI:
    dataset.print_metallicity(*logNHI)

# -- Save the dataset to file: taken from the dataset.name
dataset.save()
