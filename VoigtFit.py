# -*- coding: UTF-8 -*-
#    Written by:
#    Jens-Kristian Krogager
#    PhD Student, Dark Cosmology Centre, Niels Bohr Institute
#    University of Copenhagen
#

import numpy as np
import matplotlib
# The native MacOSX backend doesn't work for all:
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
try:
    import pyfits as pf
except:
    from astropy.io import fits as pf
import os
from argparse import ArgumentParser

import output
from parse_input import parse_parameters
from dataset import DataSet, lineList
import hdf5_save

import warnings
warnings.filterwarnings("ignore", category=matplotlib.mplDeprecation)
warnings.filterwarnings("ignore", category=UserWarning)

plt.interactive(True)


def show_transitions(ion='', lower=0., upper=None, fine_lines=False):
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
    # From Donald Morton 1991, ApJS 77,119
    if type(air) == float or type(air) == int:
        air = np.array(air)
    air = np.array(air)
    ij = (np.array(air) >= 2000)
    out = np.array(air).copy()
    sigma2 = (1.e4/air)**2
    # fact = 1.0 + 6.4328e-5 + 2.94981e-2/(146.0 - sigma2) + 2.5540e-4/( 41.0 - sigma2)
    fact = 1.0 + 6.4328e-5 + 2.94981e-2/(146.0 - sigma2) + 2.5540e-4/(41.0 - sigma2)
    out[ij] = air[ij]*fact[ij]
    return out


def SaveDataSet(filename, dataset):
    """Rewritten function to save dataset using HDF5"""
    hdf5_save.save_hdf_dataset(dataset, filename)

# def SaveDataSet(pickle_file, dataset):
#     f = open(pickle_file, 'wb')
#     # Strip parameter ties before saving.
#     # They often cause problems when loading datasets.
#     try:
#         for par in dataset.best_fit.values():
#             par.expr = None
#     except:
#         pass
#
#     try:
#         for par in dataset.pars.values():
#             par.expr = None
#     except:
#         pass
#
#     pickle.dump(dataset, f)
#     f.close()


def LoadDataSet(filename):
    """Rewritten functino to load HDF5 file"""
    dataset = hdf5_save.load_dataset_from_hdf(filename)
    return dataset

# def LoadDataSet(pickle_file):
#     f = open(pickle_file, 'rb')
#     dataset = pickle.load(f)
#     f.close()
#     return dataset


# defined here and in dataset.py for backwards compatibility
class Line(object):
    def __init__(self, tag, active=True):
        self.tag = tag
        index = lineList['trans'].tolist().index(tag)
        tag, ion, l0, f, gam = lineList[index]

        self.tag = tag
        self.ion = ion
        self.element = ion
        self.l0 = l0
        self.f = f
        self.gam = gam
        self.active = active

    def get_properties(self):
        return (self.l0, self.f, self.gam)

    def set_inactive(self):
        self.active = False

    def set_active(self):
        self.active = True


def main():
    parser = ArgumentParser()
    parser.add_argument("input", type=str,
                        help="VoigtFit input parameter file.")
    parser.add_argument("-f", action="store_true",
                        help="Force new dataset to be created. This will overwrite existing data.")

    args = parser.parse_args()
    parfile = args.input
    parameters = parse_parameters(parfile)
    print " Reading Parameters from file: " + parfile

    # Define dataset:
    name = parameters['name']
    if os.path.exists(name+'.hdf5') and not args.f:
        dataset = LoadDataSet(name+'.hdf5')

        # if len(dataset.data) != len(parameters['data']):
        dataset.data = list()
        # Setup data:
        for fname, res, norm, airORvac in parameters['data']:
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
                elif data.shape[1] == 3:
                    wl, spec, err = data.T
                elif data.shape[1] == 4:
                    wl, spec, err, mask = data.T

            if airORvac == 'air':
                wl = air2vac(wl)

            dataset.add_data(wl, spec, res, err=err, normalized=norm)

        # Add new lines that were not defined before:
        new_lines = list()
        for tag, velspan in parameters['lines']:
            if tag not in dataset.all_lines:
                new_lines.append([tag, velspan])
            else:
                reg = dataset.find_line(tag)
                if reg.velspan != velspan:
                    dataset.remove_line(tag)
                    new_lines.append([tag, velspan])

        for tag, velspan in new_lines:
                dataset.add_line(tag, velspan)

        # Remove old lines which should not be fitted:
        defined_tags = [tag for (tag, velspan) in parameters['lines']]
        for tag in dataset.all_lines:
            if tag not in defined_tags:
                dataset.deactivate_line(tag)

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
                    dataset.add_molecule(molecule, J=Jmax, velspan=velspan)

        # Remove old molecules which should not be fitted:
        defined_molecular_bands = list()
        for molecule, bands in parameters['molecules']:
            for band, Jmax, velspan in bands:
                defined_molecular_bands.append(band)

        for molecule, bands in dataset.molecules.items():
            for band in bands:
                if band not in defined_tags:
                    dataset.deactivate_molecule(molecule, band)

        # Define Components:
        dataset.reset_components()
        for component in parameters['components']:
            # ion, z, b, logN, var_z, var_b, var_N, tie_z, tie_b, tie_N = component
            # dataset.add_component(ion, z, b, logN, var_z=var_z, var_b=var_b, var_N=var_N,
            #                       tie_z=tie_z, tie_b=tie_b, tie_N=tie_N)
            ion, z, b, logN, var_z, var_b, var_N, tie_z, tie_b, tie_N, vel = component
            if vel:
                dataset.add_component_velocity(ion, z, b, logN, var_z=var_z, var_b=var_b, var_N=var_N,
                                               tie_z=tie_z, tie_b=tie_b, tie_N=tie_N)
            else:
                dataset.add_component(ion, z, b, logN, var_z=var_z, var_b=var_b, var_N=var_N,
                                      tie_z=tie_z, tie_b=tie_b, tie_N=tie_N)

        if 'interactive' in parameters.keys():
            for line_tag in parameters['interactive']:
                dataset.interactive_components(line_tag)

        for component in parameters['components_to_copy']:
            ion, anchor, logN, ref_comp, tie_z, tie_b = component
            dataset.copy_components(ion, anchor, logN=logN, ref_comp=ref_comp,
                                    tie_z=tie_z, tie_b=tie_b)

        for component in parameters['components_to_delete']:
            dataset.delete_component(*component)

    # ================================================================================
    # Generate New Dataset:
    #
    else:
        dataset = DataSet(parameters['z_sys'], parameters['name'])

        if 'velspan' in parameters.keys():
            dataset.velspan = parameters['velspan']

        # Setup data:
        for fname, res, norm, airORvac in parameters['data']:
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
                elif data.shape[1] == 3:
                    wl, spec, err = data.T
                elif data.shape[1] == 4:
                    wl, spec, err, mask = data.T

            if airORvac == 'air':
                wl = air2vac(wl)

            dataset.add_data(wl, spec, res, err=err, normalized=norm)

        # Define lines:
        for tag, velspan in parameters['lines']:
            dataset.add_line(tag, velspan)

        # Define molecules:
        if len(parameters['molecules'].items()) > 0:
            for molecule, bands in parameters['molecules'].items():
                for band, Jmax, velspan in bands:
                    dataset.add_molecule(molecule, J=Jmax, velspan=velspan)

        # Load components from file:
        if 'load' in parameters.keys():
            for fname in parameters['load']:
                print "\nLoading parameters from file: %s \n" % fname
                dataset.load_components_from_file(fname)
        else:
            dataset.reset_components()

        # Define Components:
        for component in parameters['components']:
            ion, z, b, logN, var_z, var_b, var_N, tie_z, tie_b, tie_N, vel = component
            if vel:
                print "Defining component in velocity"
                dataset.add_component_velocity(ion, z, b, logN, var_z=var_z, var_b=var_b, var_N=var_N,
                                               tie_z=tie_z, tie_b=tie_b, tie_N=tie_N)
            else:
                dataset.add_component(ion, z, b, logN, var_z=var_z, var_b=var_b, var_N=var_N,
                                      tie_z=tie_z, tie_b=tie_b, tie_N=tie_N)

        if 'interactive' in parameters.keys():
            for line_tag in parameters['interactive']:
                dataset.interactive_components(line_tag)

        for component in parameters['components_to_copy']:
            ion, anchor, logN, ref_comp, tie_z, tie_b = component
            dataset.copy_components(ion, anchor, logN=logN, ref_comp=ref_comp,
                                    tie_z=tie_z, tie_b=tie_b)

        for component in parameters['components_to_delete']:
            dataset.delete_component(*component)

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
            print "\n [WARNING] - Unexpected value for norm_method: %r" % parameters['norm_method']
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
        print "\n Continuum Fitting : Chebyshev Polynomial up to %s order\n" % (order_str)

    # Reset data in regions:
    if 'reset' in parameters.keys():
        if len(parameters['reset']) > 0:
            for line_tag in parameters['reset']:
                reg = dataset.find_line(line_tag)
                dataset.reset_region(reg)
        else:
            dataset.reset_all_regions()

    # prepare_dataset
    dataset.prepare_dataset(mask=False, norm=norm)

    # Reset all masks:
    if 'clear_mask' in parameters.keys():
        for region in dataset.regions:
            region.clear_mask()

    # Mask invidiual lines
    if 'mask' in parameters.keys():
        if len(parameters['mask']) > 0:
            for line_tag in parameters['mask']:
                dataset.mask_line(line_tag)
        else:
            for region in dataset.regions:
                region.define_mask(z=dataset.redshift, dataset=dataset)

    # update resolution:
    if len(parameters['resolution']) > 0:
        for item in parameters['resolution']:
            dataset.set_resolution(item[0], item[1])

    # fit
    popt, chi2 = dataset.fit(verbose=False, plot=False)
    print ""
    print popt.message
    print ""

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

    if 'individual-regions' in parameters['output_pars']:
        individual_regions = True
    else:
        individual_regions = False

    # print metallicity
    logNHI = parameters['logNHI']
    if logNHI:
        dataset.print_metallicity(*logNHI)

    # print abundance
    if parameters['show_abundance']:
        dataset.print_abundance()

    # save
    SaveDataSet(name + '.hdf5', dataset)
    if parameters['save']:
        filename = parameters['filename']
        if not filename:
            filename = name
        if filename.split('.')[-1] in ['pdf']:
            filename = filename[:-4]
        # plot and save
        dataset.plot_fit(filename=filename, show=True)
        output.save_parameters_to_file(dataset, filename+'.fit')
        output.save_cont_parameters_to_file(dataset, filename+'.cont')
        output.save_fit_regions(dataset, filename+'.reg', individual=individual_regions)

    else:
        dataset.plot_fit()
        plt.show()


if __name__ == '__main__':
    main()
