# -*- coding: UTF-8 -*-

__author__ = 'Jens-Kristian Krogager'

import numpy as np
import matplotlib
import warnings
import os
from sys import version_info
from matplotlib import pyplot as plt

from argparse import ArgumentParser

from VoigtFit import container
from VoigtFit import io


warnings.filterwarnings("ignore", category=matplotlib.mplDeprecation)
warnings.filterwarnings("ignore", category=UserWarning)

plt.interactive(True)

code_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(code_dir, 'VERSION')) as version_file:
    version = version_file.read().strip()
    if version_info[0] >= 3:
        v_items = version.split('.')
        v_items[0] = '3'
        version = '.'.join(v_items)
    __version__ = version



def main():

    print(r"")
    print(r"       VoigtFit %s                     " % __version__)
    print(r"")
    print(r"    by Jens-Kristian Krogager          ")
    print(r"")
    print(r"    Institut d'Astrophysique de Paris  ")
    print(r"    November 2017                      ")
    print(r"")
    print(r"  ____  _           ___________________")
    print(r"      \/ \  _/\    /                   ")
    print(r"          \/   \  / oigtFit            ")
    print(r"                \/                     ")
    print(r"")
    print(r"")
    print(r" Loaded Solar abundances from Asplund et al. (2009)")
    print(r" Using recommendations from Lodders et al. (2009)")
    print(r" regarding the source: photospheric, meteoritic or their average.")
    print(r"")

    descr = """VoigtFit Absorption Line Fitting.
    Please give an input parameter file.
    """
    parser = ArgumentParser(description=descr)
    parser.add_argument("input", type=str, nargs='?', default=None,
                        help="VoigtFit input parameter file.")
    parser.add_argument("-f", action="store_true",
                        help="Force new dataset to be created. This will overwrite existing data.")
    parser.add_argument("-v", action="store_true",
                        help="Verbose")

    args = parser.parse_args()
    parfile = args.input
    verbose = args.v
    if parfile is None:
        print("")
        print("  No input file was given.")
        print("  I have created a blank template for you to get started: 'vfit.pars'.")
        print("  Please edit this file and run VoigtFit again with this file as input.")
        print("")
        io.output.create_blank_input()
        return

    parameters = io.parse_input.parse_parameters(parfile)
    print(" Reading Parameters from file: " + parfile)

    name = parameters['name']
    # -- Load DataSet if the file already exists
    if os.path.exists(name+'.hdf5') and not args.f:
        dataset = io.hdf5_save.load_dataset(name+'.hdf5')
        if verbose:
            print("Loaded dataset: %s.hdf5" % name)

        all_spectra_defined = all([data_item[0] in dataset.data_filenames for data_item in parameters['data']])
        match_number_of_spectra = len(dataset.data) == len(parameters['data'])
        if match_number_of_spectra and all_spectra_defined:
            # all data is already defined in the dataset
            # do nothing and just move on to setting up lines and components
            pass
        else:
            dataset.data = list()
            for fname, res, norm, airORvac, nsub, ext, use_mask in parameters['data']:
                if verbose:
                    print(" Loading data: " + fname)

                dataset.add_spectrum(fname, res, airORvac, verbose=verbose, ext=ext, normalized=norm, nsub=nsub, use_mask=use_mask)
                if verbose:
                    print(" Successfully added spectrum to dataset.\n")
            # Reset all the regions in the dataset to force-reload the data:
            dataset.reset_all_regions()

        # -- Handle `lines`:
        # Add new lines that were not defined before:
        new_lines = list()
        if verbose:
            print("\n - Lines in dataset:")
            print(list(dataset.lines.keys()))
            print(" - Lines in parameter file:")
            print(parameters['lines'])
        for tag, velspan in parameters['lines']:
            if tag not in dataset.all_lines:
                new_lines.append([tag, velspan])
                if verbose:
                    print(" %s  -  new line! Adding to dataset..." % tag)
            else:
                # Check if line is active:
                this_line = dataset.lines[tag]
                if not this_line.active:
                    if verbose:
                        print(" %s  -  line was inactive! Activating line..." % tag)
                    dataset.activate_line(tag)

                # Check if velocity span has changed:
                regions_of_line = dataset.find_line(tag)
                for reg in regions_of_line:
                    if velspan is None:
                        velspan = dataset.velspan

                    if reg.velspan != velspan:
                        dataset.remove_line(tag)
                        new_lines.append([tag, velspan])
                        if verbose:
                            print(" %s  -  velspan has changed! Updating dataset..." % tag)

        for tag, velspan in new_lines:
            dataset.add_line(tag, velspan=velspan)

        # Remove old lines which should not be fitted:
        defined_tags = [tag for (tag, velspan) in parameters['lines']]
        for tag, line in dataset.lines.items():
            if tag in dataset.fine_lines.keys():
                # skip this line, cause it's a fine-structure complex of CI:
                continue

            elif line.ion[-1].islower() and line.ion[:-1] == 'CI':
                continue

            elif any([m in tag for m in dataset.molecules.keys()]):
                # skip this line, cause it's a molecular line:
                continue

            elif tag not in defined_tags:
                if verbose:
                    print(" %s - line was defined in dataset but not in parameter file" % tag)
                dataset.deactivate_line(tag)
        # --------------------------------------------------------------------

        # -- Handle `fine-structure lines`:
        # Add new fine-structure lines that were not defined before:
        new_fine_lines = list()
        if verbose:
            print("\n - Fine-structure lines in dataset:")
            print(dataset.fine_lines)
            print("\n - Fine-structure lines in parameter file:")
            print(parameters['fine-lines'])
        if len(parameters['fine-lines']) > 0:
            for ground_state, levels, velspan in parameters['fine-lines']:
                if ground_state not in dataset.fine_lines.keys():
                    if verbose:
                        print(" %s  -  new fine-structure complex" % ground_state)
                    new_fine_lines.append([ground_state, levels, velspan])
                else:
                    # Check if this line is active:
                    this_line = dataset.lines[ground_state]
                    if not this_line.active:
                        dataset.activate_fine_lines(ground_state, levels)

                    # Check if velocity span has changed:
                    if verbose:
                        print(" Checking if Velocity Span is unchaged...")
                    regions_of_line = dataset.find_line(ground_state)
                    if velspan is None:
                        velspan = dataset.velspan
                    for reg in regions_of_line:
                        if np.abs(reg.velspan - velspan) < 0.1:
                            if verbose:
                                print(" Detected difference in velocity span: %s" % ground_state)
                            dataset.verbose = False
                            dataset.remove_fine_lines(ground_state)
                            new_fine_lines.append([ground_state, levels, velspan])

        for ground_state, levels, velspan in new_fine_lines:
            dataset.add_fine_lines(ground_state, levels=levels, velspan=velspan)

        # Remove old fine-structure lines which should not be fitted:
        input_tags = [item[0] for item in parameters['fine-lines']]
        if verbose:
            print(" Any fine-structure lines in dataset that should not be fitted?")
        for ground_state, line in dataset.fine_lines.items():
            if ground_state not in input_tags:
                if verbose:
                    print(" %s  -  deactivating fine-lines" % ground_state)
                dataset.deactivate_fine_lines(ground_state, verbose=verbose)

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


    else:
        # --- Create a new DataSet
        dataset = container.dataset.DataSet(parameters['z_sys'], parameters['name'])

        if 'velspan' in parameters.keys():
            dataset.velspan = parameters['velspan']

        # Load data:
        for fname, res, norm, airORvac, nsub, ext, use_mask in parameters['data']:
            if verbose:
                print(" Loading data: " + fname)

            dataset.add_spectrum(fname, res, airORvac, verbose=verbose, ext=ext, normalized=norm, nsub=nsub, use_mask=use_mask)

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

    dataset.verbose = verbose

    if len(parameters['limits']) > 0:
        for limit_lines, limit_options in parameters['limits']:
            dataset.add_lines(limit_lines, velspan=dataset.velspan)
            for line_tag in limit_lines:
                dataset.deactivate_line(line_tag)

    for var_name, var_options in parameters['variables'].items():
        dataset.add_variable(var_name, **var_options)

    # Load components from file:
    dataset.reset_components()
    if 'load' in parameters.keys():
        for fname in parameters['load']:
            print("\n Loading components from file: %s \n" % fname)
            dataset.load_components_from_file(fname)

    # Prepare thermal model infrastructure:
    if len(parameters['thermal_model']) > 0:
        thermal_model = {ion: [] for ion in parameters['thermal_model'][0]}
        ions_in_model = ', '.join(parameters['thermal_model'][0])
        print("")
        print("  Fitting Thermal Model for ions: " + ions_in_model)
    else:
        thermal_model = dict()

    # Define Components:
    component_dict = dict()
    for component in parameters['components']:
        (ion, z, b, logN,
         var_z, var_b, var_N,
         tie_z, tie_b, tie_N,
         vel, thermal) = component

        comp_options = dict(var_z=var_z, tie_z=tie_z,
                            var_b=var_b, tie_b=tie_b,
                            var_N=var_N, tie_N=tie_N)
        if ion not in component_dict.keys():
            component_dict[ion] = list()
        component_dict[ion].append([z, b, logN, comp_options, vel])

        if vel:
            dataset.add_component_velocity(ion, z, b, logN, **comp_options)
        else:
            dataset.add_component(ion, z, b, logN, **comp_options)

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
        dataset.copy_components(ion, anchor, logN=logN, ref_comp=ref_comp,
                                tie_z=tie_z, tie_b=tie_b)
        if anchor in thermal_model.keys():
            thermal_model[ion] = thermal_model[anchor]

        # Check if separate components are defined for the ion:
        if ion in component_dict.keys():
            for component in component_dict[ion]:
                z, b, logN, comp_options, vel = component
                if vel:
                    dataset.add_component_velocity(ion, z, b, logN, **comp_options)
                else:
                    dataset.add_component(ion, z, b, logN, **comp_options)

    # Format component list to dictionary:
    components_to_delete = dict()
    for component in parameters['components_to_delete']:
        ion, comp_id = component
        if ion not in components_to_delete.keys():
            components_to_delete[ion] = list()
        components_to_delete[ion].append(comp_id)

    # Sort the component IDs high to low:
    components_to_delete = {ion: sorted(ctd, reverse=True) for ion, ctd in components_to_delete.items()}

    # Delete components from dataset:
    for ion, comps_to_del in components_to_delete.items():
        for num in comps_to_del:
            dataset.delete_component(ion, num)

            # Also remove component from thermal_model
            if ion in thermal_model.keys():
                if num in thermal_model[ion]:
                    thermal_model[ion].remove(num)

    # Fix the velocity structure of the loaded lines:
    if parameters['fix_velocity']:
        dataset.fix_structure()

    # Set default value of norm:
    norm = False
    if 'cheb_order' in parameters.keys():
        dataset.cheb_order = parameters['cheb_order']
        if parameters['cheb_order'] >= 0:
            norm = False
            dataset.reset_all_regions()
        else:
            norm = True

    if norm is True:
        if parameters['norm_method'].lower() in ['linear', 'spline']:
            dataset.norm_method = parameters['norm_method'].lower()
        else:
            warn_msg = "\n [WARNING] - Unexpected value for norm_method: %r"
            print(warn_msg % parameters['norm_method'])
            print("             Using default normalization method : linear\n")
        print("\n Continuum Fitting : manual  [%s]\n" % (dataset.norm_method))

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
        print("")
        print(stat_msg % (order_str))
        print("")

        if len(parameters['limits']) > 0:
            print("\n\n* In order to determine the equivalent width, you must normalize the line(s)")
            for limit_lines, _ in parameters['limits']:
                for line_tag in limit_lines:
                    dataset.normalize_line(line_tag, norm_method=dataset.norm_method)

    # Parse show_vel_norm from parameter file:
    # Ketyword 'norm_view' either vel or wave.
    if 'vel' in parameters['norm_view'].lower():
        show_vel_norm = True
    elif 'wave' in parameters['norm_view'].lower():
        show_vel_norm = False
    else:
        show_vel_norm = False


    # prepare_dataset
    if verbose:
        print(" - Preparing dataset:")
    prep_msg = dataset.prepare_dataset(mask=False, norm=norm, velocity=show_vel_norm,
                                       **parameters['check_lines'])
    if prep_msg and not dataset.ready2fit:
        print(prep_msg)
        return False

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
        if verbose:
            print(" Masking parameters:", parameters['mask'])
        if len(parameters['mask']) > 0:
            for line_tag, reset in zip(parameters['mask'], parameters['forced_mask']):
                dataset.mask_line(line_tag, reset=reset,
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
    print("\n  Fit is running... Please be patient.\n")
    popt, chi2 = dataset.fit(verbose=False, **parameters['fit_options'])

    print(" The fit has finished with the following exit message:")
    print("  " + popt.message)
    print("")


    # Fix for when the code cannot estimate uncertainties:
    for parname in dataset.best_fit.keys():
        err = dataset.best_fit[parname].stderr
        if err is None:
            dataset.best_fit[parname].stderr = 0.
    dataset.save(name + '.hdf5', verbose=verbose)

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
        if 'SiII' in dataset.components.keys():
            ion = 'SiII'
        elif 'FeII' in dataset.components.keys():
            ion = 'FeII'
        else:
            ion = list(dataset.components.keys())[0]

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
        io.output.print_T_model_pars(dataset, thermal_model)

    # print metallicity
    logNHI = parameters['logNHI']
    if 'HI' in dataset.components.keys():
        dataset.print_metallicity(*dataset.get_NHI())
        print("")
    elif logNHI:
        dataset.print_metallicity(*logNHI)
        print("")

    # print abundance
    if parameters['show_total']:
        dataset.print_total()
        print("")

    filename = name
    # determine limits, if any
    if len(parameters['limits']) > 0:
        print("\n\n---------------------------")
        print("  Determining Upper Limits:")
        print("---------------------------")
        EW_limits = list()
        for limit_lines, limit_options in parameters['limits']:
            for line_tag in limit_lines:
                EW = dataset.equivalent_width_limit(line_tag, verbose=True, **limit_options)
                if EW is not None:
                    EW_limits.append(EW)
                    print(io.output.format_EW(EW))
        print("")
        # Save to file:
        io.output.save_EW(EW_limits, filename+'.limits')

    # Output:
    if 'individual-regions' in parameters['output_pars']:
        individual_regions = True
    else:
        individual_regions = False

    if 'individual-components' in parameters['output_pars']:
        individual_components = True
    else:
        individual_components = False

    # plot and save
    if 'rebin' in parameters['fit_options'].keys():
        rebin = parameters['fit_options']['rebin']
    else:
        rebin = 1
    dataset.plot_fit(filename=filename, rebin=rebin)
    io.output.save_parameters_to_file(dataset, filename+'.fit')
    if dataset.cheb_order >= 0:
        io.output.save_cont_parameters_to_file(dataset, filename+'.cont')
    io.output.save_fit_regions(dataset, filename+'.reg',
                               individual=individual_regions)
    if individual_components:
        io.output.save_individual_components(dataset, filename+'.components')
    print(" - Done...\n")
    plt.show(block=True)


if __name__ == '__main__':
    main()
