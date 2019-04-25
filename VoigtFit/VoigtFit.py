# -*- coding: UTF-8 -*-

__author__ = 'Jens-Kristian Krogager'

import numpy as np
import matplotlib
import warnings
import os
# The native MacOSX backend doesn't work for all:
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from astropy.io import fits as pf
from argparse import ArgumentParser

import output
from parse_input import parse_parameters
from dataset import DataSet, lineList
import hdf5_save
from line_complexes import fine_structure_complexes


warnings.filterwarnings("ignore", category=matplotlib.mplDeprecation)
warnings.filterwarnings("ignore", category=UserWarning)

plt.interactive(True)


def show_transitions(ion='', lower=0., upper=None, fine_lines=False):
    """
    Show the transitions defined in the atomic database.

    Parameters
    ----------
    ion : str   [default = '']
        Which ion to search for in the atomic database.

    lower : float   [default = 0.]
        The lower limit on the rest-frame wavelength of the transition.

    upper : float   [default = 0.]
        The upper limit on the rest-frame wavelength of the transition.

    fine_lines : bool   [default = False]
        If `True`, then fine-structure transistions for the given ion is included.

    Returns
    -------
    all_lines : list(trans)
        A list of transitions. Each `transition` is taken from the atomic database,
        and contains the following indices: `l0`, `trans`, `ion`, `f`, `gam`, `mass`.
    """
    all_lines = list()
    if upper is None:
        upper = max(lineList['l0'])
        if len(ion) == 0:
            print " [WARNING] - No element nor upper limit on wavelength is given!"
            print "             This will return %i lines." % len(lineList)
            proceed = raw_input("Continue? (yes/NO)  > ")
            if proceed.lower() in ['y', 'yes']:
                return lineList
            else:
                return None

    if len(ion) > 0:
        for trans in lineList:
            if trans['ion'] == ion:
                if trans['l0'] > lower and trans['l0'] < upper:
                    all_lines.append(trans)
            elif trans['ion'][:-1] == ion and fine_lines is True:
                if trans['l0'] > lower and trans['l0'] < upper:
                    all_lines.append(trans)
    return all_lines


def air2vac(air):
    """
    Air to vacuum conversion from Bengt Edlén 1953,
    Journal of the Optical Society of America, Vol. 43, Issue 5, pp. 339-344.
    """
    if np.min(air) < 2000.:
        raise ValueError("Input wavelength below valid range!")
    air = np.array(air)
    ij = (np.array(air) >= 2000)
    out = np.array(air).copy()
    s2 = (1.e4/air)**2
    fact = 1.0 + 6.4328e-5 + 2.94981e-2/(146.0 - s2) + 2.5540e-4/(41.0 - s2)
    # Alternative solution from VALD:
    # fact = 1 + 8.336624e-5 + 0.02408927 / (130.106592 - s2) + 1.599740895e-4 / (38.925688 - s2),
    out[ij] = air[ij]*fact[ij]
    return out


def SaveDataSet(filename, dataset):
    """Save dataset to HDF5 file."""
    hdf5_save.save_hdf_dataset(dataset, filename)


def LoadDataSet(filename):
    """Load a dataset from a HDF5 file."""
    dataset = hdf5_save.load_dataset_from_hdf(filename)
    return dataset


def main():

    print "\n"
    print "       VoigtFit                        "
    print ""
    print "    by Jens-Kristian Krogager          "
    print ""
    print "    Institut d'Astrophysique de Paris  "
    print "    November 2017                      "
    print ""
    print "  ____  _           ___________________"
    print "      \/ \  _/\    /                   "
    print "          \/   \  / oigtFit            "
    print "                \/                     "
    print ""
    print ""
    print " Loaded Solar abundances from Asplund et al. 2009  (photospheric)"
    print ""

    descr = """VoigtFit Absorption Line Fitting.
    Please give an input parameter file.
    """
    parser = ArgumentParser(description=descr)
    parser.add_argument("input", type=str, nargs='?', default=None,
                        help="VoigtFit input parameter file.")
    parser.add_argument("-f", action="store_true",
                        help="Force new dataset to be created. This will overwrite existing data.")

    args = parser.parse_args()
    parfile = args.input
    if parfile is None:
        print("")
        print("  No input file was given.")
        print("  I have created a blank template for you to get started: 'vfit.pars'.")
        print("  Please edit this file and run VoigtFit again with this file as input.")
        print("")
        output.create_blank_input()
        return

    parameters = parse_parameters(parfile)
    print " Reading Parameters from file: " + parfile

    name = parameters['name']
    # -- Load DataSet if the file already exists
    if os.path.exists(name+'.hdf5') and not args.f:
        dataset = LoadDataSet(name+'.hdf5')

        # if len(dataset.data) != len(parameters['data']):
        dataset.data = list()
        # Setup data:

        for fname, res, norm, airORvac, nsub in parameters['data']:
            if fname[-5:] == '.fits':
                hdu = pf.open(fname)
                spec = pf.getdata(fname, 0)
                hdr = pf.getheader(fname)
                wl = hdr['CRVAL1'] + np.arange(len(spec))*hdr['CD1_1']

                if len(hdu) > 1:
                    err = pf.getdata(fname, 1)
                elif parameters['snr'] is not None:
                    # err = spec/parameters['snr']
                    err = np.ones_like(spec)*np.median(spec)/parameters['snr']
                else:
                    err = spec/10.
                err[err <= 0.] = np.abs(np.mean(err))
                mask = np.ones_like(wl, dtype=bool)

            else:
                data = np.loadtxt(fname)
                if data.shape[1] == 2:
                    wl, spec = data.T
                    if parameters['snr'] is not None:
                        # err = spec/parameters['snr']
                        err = np.ones_like(spec)*np.median(spec)/parameters['snr']
                    else:
                        err = spec/10.
                    err[err <= 0.] = np.abs(np.mean(err))
                    mask = np.ones_like(wl, dtype=bool)
                elif data.shape[1] == 3:
                    wl, spec, err = data.T
                    mask = np.ones_like(wl, dtype=bool)
                elif data.shape[1] == 4:
                    wl, spec, err, mask = data.T

            if airORvac == 'air':
                wl = air2vac(wl)

            dataset.add_data(wl, spec, res,
                             err=err, normalized=norm, mask=mask, nsub=nsub)

        # -- Handle `lines`:
        # Add new lines that were not defined before:
        new_lines = list()
        for tag, velspan in parameters['lines']:
            if tag not in dataset.all_lines:
                new_lines.append([tag, velspan])
            else:
                # Check if velocity span has changed:
                regions_of_line = dataset.find_line(tag)
                for reg in regions_of_line:
                    if velspan is None:
                        velspan = dataset.velspan

                    if reg.velspan != velspan:
                        dataset.remove_line(tag)
                        new_lines.append([tag, velspan])

                # Check if line is active:
                this_line = dataset.lines[tag]
                if not this_line.active:
                    dataset.activate_line(tag)

        for tag, velspan in new_lines:
            dataset.add_line(tag, velspan=velspan)

        # Remove old lines which should not be fitted:
        defined_tags = [tag for (tag, velspan) in parameters['lines']]
        for tag, line in dataset.lines.items():
            if line.ion[-1].islower():
                # skip this line, cause it's a fine-structure line:
                continue

            elif any([m in tag for m in dataset.molecules.keys()]):
                # skip this line, cause it's a molecular line:
                continue

            elif tag not in defined_tags:
                dataset.deactivate_line(tag)
        # --------------------------------------------------------------------

        # -- Handle `fine-structure lines`:
        # Add new fine-structure lines that were not defined before:
        new_fine_lines = list()
        if len(parameters['fine-lines']) > 0:
            for ground_state, levels, velspan in parameters['fine-lines']:
                if ground_state not in dataset.all_lines:
                    new_fine_lines.append([ground_state, levels, velspan])
                else:
                    # Check if velocity span has changed:
                    regions_of_line = dataset.find_line(ground_state)
                    for reg in regions_of_line:
                        if reg.velspan != velspan:
                            dataset.remove_fine_lines(ground_state)
                            new_fine_lines.append([ground_state, levels, velspan])

                    # Check if this line is active:
                    this_line = dataset.lines[ground_state]
                    if not this_line.active:
                        dataset.activate_fine_lines(ground_state, levels)

        for ground_state, levels, velspan in new_fine_lines:
            dataset.add_fine_lines(ground_state, levels=levels, velspan=velspan)

        # Remove old fine-structure lines which should not be fitted:
        input_tags = [item[0] for item in parameters['fine-lines']]
        for tag, line in dataset.lines.items():
            # Only consider fine-structure lines:
            fine_line_states = fine_structure_complexes.keys()
            if tag in fine_line_states and tag not in input_tags:
                dataset.deactivate_fine_lines(tag)

        # --------------------------------------------------------------------

        # -- Handle `molecules`:
        # Add new molecules that were not defined before:
        new_molecules = dict()
        if len(parameters['molecules'].items()) > 0:
            for molecule, bands in parameters['molecules'].items():
                if molecule not in new_molecules.keys():
                    new_molecules[molecule] = list()

                if molecule in dataset.molecules.keys():
                    for band, Jmax, velspan in bands:
                        if band not in dataset.molecules[molecule]:
                            new_molecules[molecule].append([band, Jmax, velspan])

        if len(new_molecules.items()) > 0:
            for molecule, bands in new_molecules.items():
                for band, Jmax, velspan in bands:
                    dataset.add_molecule(molecule, Jmax=Jmax, velspan=velspan)

        # Remove old molecules which should not be fitted:
        defined_molecular_bands = list()
        for molecule, bands in parameters['molecules']:
            for band, Jmax, velspan in bands:
                defined_molecular_bands.append(band)

        for molecule, bands in dataset.molecules.items():
            for band, Jmax in bands:
                if band not in defined_molecular_bands:
                    dataset.deactivate_molecule(molecule, band)

    # -- Otherwise create a new DataSet
    else:
        # ================================================================================
        # Generate New Dataset:
        #
        dataset = DataSet(parameters['z_sys'], parameters['name'])

        if 'velspan' in parameters.keys():
            dataset.velspan = parameters['velspan']

        # Setup data:
        for fname, res, norm, airORvac, nsub in parameters['data']:
            if fname[-5:] == '.fits':
                hdu = pf.open(fname)
                spec = pf.getdata(fname)
                hdr = pf.getheader(fname)
                wl = hdr['CRVAL1'] + np.arange(len(spec))*hdr['CD1_1']
                if len(hdu) > 1:
                    err = hdu[1].data
                elif parameters['snr'] is not None:
                    # err = spec/parameters['snr']
                    err = np.ones_like(spec)*np.median(spec)/parameters['snr']
                else:
                    err = spec/10.
                err[err <= 0.] = np.abs(np.mean(err))
                mask = np.ones_like(wl, dtype=bool)

            else:
                data = np.loadtxt(fname)
                if data.shape[1] == 2:
                    wl, spec = data.T
                    if parameters['snr'] is not None:
                        # err = spec/parameters['snr']
                        err = np.ones_like(spec)*np.median(spec)/parameters['snr']
                    else:
                        err = spec/10.
                    err[err <= 0.] = np.abs(np.mean(err))
                    mask = np.ones_like(wl, dtype=bool)
                elif data.shape[1] == 3:
                    wl, spec, err = data.T
                    mask = np.ones_like(wl, dtype=bool)
                elif data.shape[1] == 4:
                    wl, spec, err, mask = data.T

            if airORvac == 'air':
                wl = air2vac(wl)

            dataset.add_data(wl, spec, res, err=err, normalized=norm, mask=mask, nsub=nsub)

        # Define lines:
        for tag, velspan in parameters['lines']:
            dataset.add_line(tag, velspan=velspan)

        # Define fine-structure lines:
        for ground_state, levels, velspan in parameters['fine-lines']:
            dataset.add_fine_lines(ground_state, levels=levels, velspan=velspan)

        # Define molecules:
        if len(parameters['molecules'].items()) > 0:
            for molecule, bands in parameters['molecules'].items():
                for band, Jmax, velspan in bands:
                    dataset.add_molecule(molecule, Jmax=Jmax, velspan=velspan)

    # =========================================================================
    # Back to Common Work Flow for all datasets:

    # HERE masking is correct!

    # Load components from file:
    if 'load' in parameters.keys():
        dataset.reset_components()
        for fname in parameters['load']:
            print "\n Loading parameters from file: %s \n" % fname
            dataset.load_components_from_file(fname)
    else:
        dataset.reset_components()

    # Fix the velocity structure of the loaded lines:
    if parameters['fix_velocity']:
        dataset.fix_structure()

    # Prepare thermal model infrastructure:
    if len(parameters['thermal_model']) > 0:
        thermal_model = {ion: [] for ion in parameters['thermal_model'][0]}
        ions_in_model = ', '.join(parameters['thermal_model'][0])
        print ""
        print "  Fitting Thermal Model for ions: " + ions_in_model
    else:
        thermal_model = dict()

    # Define Components:
    for component in parameters['components']:
        (ion, z, b, logN,
         var_z, var_b, var_N,
         tie_z, tie_b, tie_N,
         vel, thermal) = component

        if vel:
            dataset.add_component_velocity(ion, z, b, logN,
                                           var_z=var_z, var_b=var_b,
                                           var_N=var_N, tie_z=tie_z,
                                           tie_b=tie_b, tie_N=tie_N)
        else:
            dataset.add_component(ion, z, b, logN,
                                  var_z=var_z, var_b=var_b, var_N=var_N,
                                  tie_z=tie_z, tie_b=tie_b, tie_N=tie_N)

        if ion in thermal_model.keys():
            thermal_model[ion].append(thermal)

    # Convert boolean indices to component indcides:
    # Ex: [True, False, True, False] -> [0, 2]
    for ion, values in thermal_model.items():
        if np.any(values):
            pass
        else:
            # If no components have been explicitly defined
            # as part of the thermal model, assign all components
            values = [True for _ in values]
        thermal_model[ion] = list(np.nonzero(values)[0])

    if 'interactive' in parameters.keys():
        for line_tag in parameters['interactive']:
            dataset.interactive_components(line_tag, velocity=parameters['interactive_view'])

    for component in parameters['components_to_copy']:
        ion, anchor, logN, ref_comp, tie_z, tie_b = component
        if anchor in thermal_model.keys():
            dataset.copy_components(ion, anchor, logN=logN, ref_comp=ref_comp,
                                    tie_z=tie_z, tie_b=False)
            thermal_model[ion] = thermal_model[anchor]
        else:
            dataset.copy_components(ion, anchor, logN=logN, ref_comp=ref_comp,
                                    tie_z=tie_z, tie_b=tie_b)

    for component in parameters['components_to_delete']:
        dataset.delete_component(*component)

        # Also remove component from therma_model
        ion, num = component
        if ion in thermal_model.keys():
            if num in thermal_model[ion]:
                thermal_model[ion].remove(num)

    # Set default value of norm:
    norm = False
    if 'cheb_order' in parameters.keys():
        dataset.cheb_order = parameters['cheb_order']
        if parameters['cheb_order'] >= 0:
            norm = False
        else:
            norm = True

    if norm is True:
        if parameters['norm_method'].lower() in ['linear', 'spline']:
            dataset.norm_method = parameters['norm_method'].lower()
        else:
            warn_msg = "\n [WARNING] - Unexpected value for norm_method: %r"
            print warn_msg % parameters['norm_method']
            print "             Using default normalization method : linear\n"
        print "\n Continuum Fitting : manual  [%s]\n" % (dataset.norm_method)

    else:
        if dataset.cheb_order == 1:
            order_str = "%ist" % (dataset.cheb_order)
        elif dataset.cheb_order == 2:
            order_str = "%ind" % (dataset.cheb_order)
        elif dataset.cheb_order == 3:
            order_str = "%ird" % (dataset.cheb_order)
        else:
            order_str = "%ith" % (dataset.cheb_order)
        stat_msg = " Continuum Fitting : Chebyshev Polynomial up to %s order"
        print ""
        print stat_msg % (order_str)
        print ""

    # Parse show_vel_norm from parameter file:
    # Ketyword 'norm_view' either vel or wave.
    if 'vel' in parameters['norm_view'].lower():
        show_vel_norm = True
    elif 'wave' in parameters['norm_view'].lower():
        show_vel_norm = False
    else:
        show_vel_norm = False

    # Reset data in regions:
    if 'reset' in parameters.keys():
        if len(parameters['reset']) > 0:
            for line_tag in parameters['reset']:
                regions_of_line = dataset.find_line(line_tag)
                for reg in regions_of_line:
                    dataset.reset_region(reg)
        else:
            dataset.reset_all_regions()

    # prepare_dataset
    dataset.prepare_dataset(mask=False, norm=norm, velocity=show_vel_norm)

    # Define thermal model
    if len(thermal_model.keys()) > 0:
        # Get all the indices of components that have thermal components
        thermal_components = list(set(sum(thermal_model.values(), [])))
        (thermal_ions, T_init, turb_init,
         fix_T, fix_turb) = parameters['thermal_model']

        var_T = not fix_T
        var_turb = not fix_turb
        for num in thermal_components:
            dataset.pars.add('T_%i' % num, value=T_init, min=0.,
                             vary=var_T)
            dataset.pars.add('turb_%i' % num, value=turb_init, min=0.,
                             vary=var_turb)

        # 2k_B/m_u in (km/s)^2 units
        K = 0.0166287
        for ion in thermal_ions:
            for comp_num in thermal_model[ion]:
                par_name = 'b%i_%s' % (comp_num, ion)
                lines_for_ion = dataset.get_lines_for_ion(ion)
                m_ion = lines_for_ion[0].mass
                T_num = dataset.pars['T_%i' % comp_num].value
                turb_num = dataset.pars['turb_%i' % comp_num].value
                b_eff = np.sqrt(turb_num**2 + K*T_num/m_ion)
                mod_pars = (comp_num, K/m_ion, comp_num)
                model_constraint = 'sqrt((turb_%i)**2 + %.6f*T_%i)' % mod_pars
                dataset.pars[par_name].set(expr=model_constraint, value=b_eff)

    # Reset all masks:
    if parameters['clear_mask']:
        for region in dataset.regions:
            region.clear_mask()

    # Parse show_vel_mask from parameter file:
    # Ketyword 'mask_view' either vel or wave.
    if 'vel' in parameters['mask_view'].lower():
        show_vel_mask = True
    elif 'wave' in parameters['mask_view'].lower():
        show_vel_mask = False
    else:
        show_vel_mask = False

    # Mask invidiual lines
    if 'mask' in parameters.keys():
        if len(parameters['mask']) > 0:
            for line_tag in parameters['mask']:
                dataset.mask_line(line_tag, reset=False,
                                  velocity=show_vel_mask)
        else:
            if show_vel_mask:
                z_sys = dataset.redshift
            else:
                z_sys = None
            for region in dataset.regions:
                if region.new_mask and region.has_active_lines():
                    region.define_mask(z=dataset.redshift,
                                       dataset=dataset,
                                       z_sys=z_sys)

    # update resolution:
    if len(parameters['resolution']) > 0:
        for item in parameters['resolution']:
            dataset.set_resolution(item[0], item[1])

    # Run the fit:
    popt, chi2 = dataset.fit(verbose=False, plot=False,
                             **parameters['fit_options'])

    print ""
    print popt.message
    print ""

    # Fix for when the code cannot estimate uncertainties:
    for parname in dataset.best_fit.keys():
        err = dataset.best_fit[parname].stderr
        if err is None:
            dataset.best_fit[parname].stderr = 0.
    SaveDataSet(name + '.hdf5', dataset)

    # Update systemic redshift
    if parameters['systemic'][1] == 'none':
        # do not update the systemic redshift
        pass

    elif isinstance(parameters['systemic'][0], int):
        num, ion = parameters['systemic']
        if num == -1:
            num = len(dataset.components[ion]) - 1
        new_z_sys = dataset.best_fit['z%i_%s' % (num, ion)].value
        dataset.set_systemic_redshift(new_z_sys)

    elif parameters['systemic'][1] == 'auto':
        # find ion to search for strongest component:
        if 'FeII' in dataset.components.keys():
            ion = 'FeII'
        elif 'SiII' in dataset.components.keys():
            ion = 'SiII'
        else:
            ion = dataset.components.keys()[0]

        # find strongest component:
        n_comp = len(dataset.components[ion])
        logN_list = list()
        for n in range(n_comp):
            this_logN = dataset.best_fit['logN%i_%s' % (n, ion)].value
            logN_list.append(this_logN)
        num = np.argmax(logN_list)
        new_z_sys = dataset.best_fit['z%i_%s' % (num, ion)].value
        dataset.set_systemic_redshift(new_z_sys)
    else:
        systemic_err_msg = "Invalid mode to set systemic redshift: %r" % parameters['systemic']
        raise ValueError(systemic_err_msg)

    if 'velocity' in parameters['output_pars']:
        dataset.print_results(velocity=True)
    else:
        dataset.print_results(velocity=False)

    if len(thermal_model.keys()) > 0:
        # print Thermal Model Parameters
        output.print_T_model_pars(dataset, thermal_model)

    if dataset.cheb_order >= 0:
        dataset.print_cont_parameters()

    if 'individual-regions' in parameters['output_pars']:
        individual_regions = True
    else:
        individual_regions = False

    # print metallicity
    logNHI = parameters['logNHI']
    if logNHI:
        dataset.print_metallicity(*logNHI)

    # print abundance
    if parameters['show_total']:
        dataset.print_total()

    # save
    if parameters['save']:
        filename = parameters['filename']
        if not filename:
            filename = name
        if filename.split('.')[-1] in ['pdf']:
            filename = filename[:-4]
        # plot and save
        if 'rebin' in parameters['fit_options'].keys():
            rebin = parameters['fit_options']['rebin']
        else:
            rebin = 1
        dataset.plot_fit(filename=filename, rebin=rebin)
        output.save_parameters_to_file(dataset, filename+'.fit')
        output.save_cont_parameters_to_file(dataset, filename+'.cont')
        output.save_fit_regions(dataset, filename+'.reg',
                                individual=individual_regions)
        plt.show(block=True)

    else:
        if 'rebin' in parameters['fit_options'].keys():
            rebin = parameters['fit_options']['rebin']
        else:
            rebin = 1
        dataset.plot_fit(rebin=rebin)
        plt.show(block=True)


if __name__ == '__main__':
    main()
