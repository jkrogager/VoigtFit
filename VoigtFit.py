# -*- coding: UTF-8 -*-
#    Written by:
#    Jens-Kristian Krogager
#    PhD Student, Dark Cosmology Centre, Niels Bohr Institute
#    University of Copenhagen
#

import numpy as np
import matplotlib.pyplot as plt
import pyfits as pf
import pickle
import os
from argparse import ArgumentParser

import output
from parse_input import parse_parameters
from dataset import DataSet, lineList


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


def SaveDataSet(pickle_file, dataset):
    f = open(pickle_file, 'wb')
    # Strip parameter ties before saving.
    # They often cause problems when loading datasets.
    try:
        for par in dataset.best_fit.values():
            par.expr = None
    except:
        pass

    try:
        for par in dataset.pars.values():
            par.expr = None
    except:
        pass

    pickle.dump(dataset, f)
    f.close()


def LoadDataSet(pickle_file):
    f = open(pickle_file, 'rb')
    dataset = pickle.load(f)
    f.close()
    return dataset


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

    args = parser.parse_args()
    parfile = args.input
    parameters = parse_parameters(parfile)
    print " Reading Parameters from file: " + parfile

    # Define dataset:
    name = parameters['name']
    if os.path.exists(name+'.dataset'):
        dataset = LoadDataSet(name+'.dataset')

        # Add new lines that were not defined before:
        new_lines = list()
        for tag, velspan in parameters['lines']:
            if tag not in dataset.all_lines:
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
            ion, z, b, logN, var_z, var_b, var_N, tie_z, tie_b, tie_N = component
            dataset.add_component(ion, z, b, logN, var_z=var_z, var_b=var_b, var_N=var_N,
                                  tie_z=tie_z, tie_b=tie_b, tie_N=tie_N)

        for component in parameters['components_to_copy']:
            ion, anchor, logN, ref_comp, tie_z, tie_b = component
            dataset.copy_components(ion, anchor, logN=logN, ref_comp=ref_comp,
                                    tie_z=tie_z, tie_b=tie_b)

        for component in parameters['components_to_delete']:
            dataset.delete_component(*component)

    else:
        dataset = DataSet(parameters['z_sys'])

        # Setup data:
        for fname, res, norm, airORvac in parameters['data']:
            if fname[-5:] == '.fits':
                spec = pf.getdata(fname)
                hdr = pf.getheader(fname)
                wl = hdr['CRVAL1'] + np.arange(len(spec))*hdr['CD1_1']
                N = len(spec)
                err = np.std(spec[N/2-N/20:N/2+N/20])*np.ones_like(spec)

            else:
                data = np.loadtxt(fname)
                if data.shape[1] == 2:
                    wl, spec = data.T
                    N = len(spec)
                    err = np.std(spec[N/2-N/20:N/2+N/20]) * np.ones_like(spec)
                elif data.shape[1] == 3:
                    wl, spec, err = data.T
                elif data.shape[1] == 4:
                    wl, spec, err, mask = data.T

            if airORvac == 'air':
                wl = air2vac(wl)

            dataset.add_data(wl, spec, res, err=err, normalized=norm)

        # Define normalization method:
        # dataset.norm_method = 1

        # Define lines:
        for tag, velspan in parameters['lines']:
            dataset.add_line(tag, velspan)

        # Define molecules:
        if len(parameters['molecules'].items()) > 0:
            for molecule, bands in parameters['molecules'].items():
                for band, Jmax, velspan in bands:
                    dataset.add_molecule(molecule, J=Jmax, velspan=velspan)

        # Define Components:
        dataset.reset_components()
        for component in parameters['components']:
            ion, z, b, logN, var_z, var_b, var_N, tie_z, tie_b, tie_N = component
            dataset.add_component(ion, z, b, logN, var_z=var_z, var_b=var_b, var_N=var_N,
                                  tie_z=tie_z, tie_b=tie_b, tie_N=tie_N)

        for component in parameters['components_to_copy']:
            ion, anchor, logN, ref_comp, tie_z, tie_b = component
            dataset.copy_components(ion, anchor, logN=logN, ref_comp=ref_comp,
                                    tie_z=tie_z, tie_b=tie_b)

        for component in parameters['components_to_delete']:
            dataset.delete_component(*component)

    # prepare_dataset
    if parameters['nomask']:
        dataset.prepare_dataset(mask=False)
    else:
        dataset.prepare_dataset(mask=True)

    # update resolution:
    if len(parameters['resolution']) > 0:
        for item in parameters['resolution']:
            dataset.set_resolution(item[0], item[1])

    # fit
    dataset.fit(verbose=False, plot=False)

    # print metallicity
    dataset.print_results()
    logNHI = parameters['logNHI']
    if logNHI:
        dataset.print_metallicity(*logNHI)

    # print abundance
    if parameters['show_abundance']:
        dataset.print_abundance()

    # save
    SaveDataSet(name+'.dataset', dataset)
    # dataset.save(name + '.dataset')
    if parameters['save']:
        filename = parameters['filename']
        if not filename:
            filename = name
        if filename.split('.')[-1] in ['pdf']:
            filename = filename[:-4]
        # plot and save
        dataset.plot_fit(filename=filename, show=False)

        output.save_parameters_to_file(dataset, filename)

    else:
        dataset.plot_fit()
        plt.show()


if __name__ == '__main__':
    main()
