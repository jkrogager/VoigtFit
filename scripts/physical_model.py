# -*- coding: UTF-8 -*-
__author__ = 'Jens-Kristian Krogager'

import numpy as np
import VoigtFit


def print_T_model_pars(dataset, filename=None):
    """Print the turbulence and T parameters for physical model."""
    N_comp = len(dataset.components.values()[0])
    print("")
    print(u"  No:     Temperature [K]       Turbulence [km/s]")
    if filename:
        out_file = open(filename, 'w')
        out_file.write(u"# No:     Temperature [K]       Turbulence [km/s] \n")

    for comp_num in range(N_comp):
        T_name = 'T_%i' % comp_num
        turb_name = 'turb_%i' % comp_num
        T_fit = dataset.best_fit[T_name]
        turb_fit = dataset.best_fit[turb_name]
        par_tuple = (comp_num, T_fit.value, T_fit.stderr,
                     turb_fit.value, turb_fit.stderr)
        print(u"  %-3i   %.2e ± %.2e    %.2e ± %.2e" % par_tuple)
        if filename:
            out_file.write(u"  %-3i   %.2e ± %.2e    %.2e ± %.2e \n" % par_tuple)

    print("")
    if filename:
        out_file.close()


# -- Fit DLA towards quasar Q1313+1441
#    Observed in X-shooter P089.A-0068

z_DLA = 0.00345

# If log(NHI) is not known use:
logNHI = None

# -- Load X-SHOOTER UVB and VIS data in ASCII format:
fname = 'thermal_model_2comp.dat'
res = 6.6

wl, spec, err = np.loadtxt(fname, unpack=True)

# -- Here you can load your data in any way you wish
#    Only requirement is that wl, spec, and err have the same dimensions.

# -- A dataset which has already been defined can be loaded like this:
# dataset = VoigtFit.LoadDataSet('test_data.hdf5')

dataset = VoigtFit.DataSet(z_DLA)
dataset.set_name('test_2comp')
dataset.verbose = True
dataset.velspan = 150.
dataset.cheb_order = -1

# -- Add the data loaded from the
dataset.add_data(wl, spec, res, err=err, normalized=True)

# -- Define absorption lines:
dataset.add_many_lines(['FeII_2344', 'FeII_2374', 'FeII_2382'])
dataset.add_many_lines(['FeII_1608', 'FeII_1611'])
dataset.add_line('FeII_2260')
dataset.add_line('FeII_2249')
dataset.add_line('CrII_2056')
dataset.add_line('CrII_2066')
dataset.add_line('CrII_2026')
dataset.add_line('ZnII_2026')
dataset.add_line('CrII_2062')
dataset.add_line('ZnII_2062')
# -- dataset.add_many_lines is equivalent to dataset.add_lines:
dataset.add_many_lines(['CII_1036', 'CII_1334'])
dataset.add_many_lines(['OI_1302', 'OI_1039', 'OI_1355'])
dataset.add_many_lines(['SiII_1526', 'SiII_1808', 'SiII_1304'])
dataset.add_many_lines(['SiII_1260', 'FeII_1260', 'SII_1259'])
dataset.add_many_lines(['SII_1250', 'SII_1253'])


# -- If a line has been defined, and you don't want to fit it
#    it can either be removed from the dataset completely:
# dataset.remove_line('CrII_2056')

# -- or deactivated:
# dataset.deactivate_line('FeII_2374')

# -- Deactivated lines will not be included in the fit, but their line definitions
#    and components remain in the dataset for future reference.

# -- To use the physical model, make sure that all components are cleared:
dataset.reset_components()

# -- Add components for each ion:
#                      ion    z         b   logN
dataset.add_component('FeII', 0.003290, 5., 15.0, var_z=1, var_b=1)
dataset.add_component('FeII', 0.003620, 5., 14.5, var_z=1, var_b=1)


# -- The physical model requires that all ions have the same velocity structure:
#    The default order is 'to' , 'from' :
# dataset.copy_components('CrII', 'FeII')
# -- But the ions can be specified using keywords to ease the call:
dataset.copy_components(from_ion='FeII', to_ion='CrII', tie_b=False)
dataset.copy_components(from_ion='FeII', to_ion='ZnII', tie_b=False)
dataset.copy_components(from_ion='FeII', to_ion='SiII', tie_b=False)
dataset.copy_components(from_ion='FeII', to_ion='SII', tie_b=False)
dataset.copy_components(from_ion='FeII', to_ion='CII', tie_b=False)
dataset.copy_components(from_ion='FeII', to_ion='OI', tie_b=False)

# -- This copies the two components defined for FeII to the other ions and
#    keeps the same pattern of initial guesses for column density scaled
#    to the Solar abundance ratio.

# -- Individual components which are not observed for weaker lines can be removed:
# dataset.delete_component('ZnII', 1)
# dataset.delete_component('ZnII', 0)
#
# NOTE - components should be deleted from last component to first component
#        not the other way around as that messes up the component numbering.
#        Components are zero-indexed!

# -- Prepare the dataset: This will prompt the user for interactive
#    masking and normalization, as well as initiating the Parameters:
dataset.prepare_dataset(norm=False, mask=False)

# -- Define masks for individual lines:
# dataset.mask_line('ZnII_2026')

# --- This is where the magic happens ----------------------------------------
# Set up the thermal and turbulence parameters for each component:
dataset.pars.add('turb_0', value=5., vary=True, min=0.)
dataset.pars.add('turb_1', value=5., vary=True, min=0.)

dataset.pars.add('T_0', value=5000., vary=1, min=0.)
dataset.pars.add('T_1', value=5000., vary=1, min=0.)
# -- This can be defined in a for loop assuming the same intial guess for T:
# T_init = 1.e4
# for comp_num in range(len(dataset.components.values()[0])):
#     dataset.pars.add('T_%i'%comp_num, value=T_init, vary=True, min=0.)

# -- Now set up the links for the 'b'-parameter of each component of each ion:
# 2k_B/m_u in (km/s)^2 units:
K = 0.0166287
for ion, comp in dataset.components.items():
    N_comp = len(comp)
    for comp_num in range(N_comp):
        par_name = 'b%i_%s' % (comp_num, ion)
        lines_for_ion = dataset.get_lines_for_ion(ion)
        m_ion = lines_for_ion[0].mass
        const = K/m_ion
        T_num = dataset.pars['T_%i' % comp_num].value
        b_eff = np.sqrt(5.**2 + K*T_num/m_ion)
        model_constraint = 'sqrt((turb_%i)**2 + %.6f*T_%i)' % (comp_num,
                                                               const,
                                                               comp_num)
        dataset.pars[par_name].set(expr=model_constraint, value=b_eff)

# ---------------------------------------------------------------------------

# -- Fit the dataset:
popt, chi2 = dataset.fit(verbose=True, plot=False, factor=10.)

dataset.plot_fit(filename=dataset.name)

# -- Print total column densities
dataset.print_total()

if logNHI:
    dataset.print_metallicity(*logNHI)

print_T_model_pars(dataset)

# -- Save the dataset to file: taken from the dataset.name
dataset.save()
