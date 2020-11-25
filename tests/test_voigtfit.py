# -*- coding: UTF-8 -*-

import numpy as np
from os.path import exists, abspath, dirname, join
from os import remove

import VoigtFit as vfit

code_dir = dirname(abspath(__file__))

def test_column_densities():

    input_data = dict()
    dat_in = np.loadtxt(join(code_dir, 'test_2comp.input'), dtype=str)
    for line in dat_in:
        ion = line[0]
        input_data[ion] = float(line[3])

    ### Load the test data and check output
    z_sys = 2.3538
    test_fname = join(code_dir, 'test_2comp.dat')
    ds = vfit.DataSet(z_sys)
    ds.verbose = False
    ds.cheb_order = -1
    ds.velspan = 200.
    res = 299792. / 10000.
    ds.add_spectrum(test_fname, res, normalized=True)
    ds.add_lines(['SiII_1808', 'SiII_1020', 'SiII_1304', 'FeII_1611', 'FeII_2249', 'FeII_2260', 'FeII_2374'])
    ds.add_lines(['SII_1253', 'SII_1250'])
    ds.add_line('HI_1215', velspan=2000.)
    ds.add_component('HI', 2.3535, 50., 20.3, var_b=False)
    ds.add_component('SiII', 2.3532, 15., 15.4)
    ds.add_component('SiII', 2.3539, 10., 15.8)
    ds.copy_components(from_ion='SiII', to_ion='SII')
    ds.copy_components(from_ion='SiII', to_ion='FeII')
    ds.prepare_dataset(mask=False, f_lower=10.)

    popt, chi2 = ds.fit(verbose=False)

    logN_criteria = list()
    for ion in list(ds.components.keys()):
        if ion == 'HI':
            logN_tot = popt.params['logN0_%s' % ion].value
        else:
            logN1 = popt.params['logN0_%s' % ion].value
            logN2 = popt.params['logN1_%s' % ion].value
            logN_tot = np.log10(10**logN1 + 10**logN2)
        delta = logN_tot - input_data[ion]
        print("%s : %.2f  [input: %.2f]" % (ion, logN_tot, input_data[ion]))
        logN_criteria.append(delta < 0.02)

    assert all(logN_criteria), "Not all column densities were recovered correctly."


def test_dataset():
    z_sys = 2.3538
    test_fname = join(code_dir, 'test_2comp.dat')
    res = 299792. / 10000.
    ds = vfit.DataSet(z_sys)
    ds.add_spectrum(test_fname, res, normalized=True)
    ds.verbose = False
    ds.cheb_order = 2
    ds.velspan = 200.

    ds.add_lines(['FeII_1608', 'FeII_1611', 'SiII_1526', 'SiII_1808'])
    ds.deactivate_line('SiII_1526')
    ds.add_component_velocity('FeII', -50., 10, 15.1)
    ds.add_component_velocity('FeII', +10., 10, 15.4)
    ds.add_component_velocity('SiII', -50., 10, 15.4)
    ds.add_component_velocity('SiII', +10., 10, 15.8)
    ds.prepare_dataset(mask=False, f_lower=10.)

    N_regions = len(ds.regions)
    N_active_regions = len([reg for reg in ds.regions if reg.has_active_lines()])
    N_pars = len(ds.pars.keys())

    assert N_regions == 4, "Incorrect number of fit regions!"
    assert N_active_regions == 3, "Incorrect number of active fit regions!"
    assert N_pars == 21

    ds.delete_component('SiII', 0)
    ds.add_line('FeII_2374')
    ds.add_fine_lines('CI_1656')
    ds.deactivate_fine_lines('CI_1656')
    ds.activate_line('SiII_1526')
    ds.prepare_dataset(mask=False, f_lower=10.)

    assert len(ds.regions) == 6
    N_active_regions = len([reg for reg in ds.regions if reg.has_active_lines()])
    assert N_active_regions == 5
    assert len(ds.pars.keys()) == 24
    N_lines = len(ds.all_lines)
    assert N_lines == 11, "Incorrect number of lines"


def test_output():

    ### Load the test data and save output
    z_sys = 2.3538
    test_fname = join(code_dir, 'test_2comp.dat')
    ds = vfit.DataSet(z_sys)
    ds.verbose = False
    ds.cheb_order = 0
    ds.velspan = 200.
    res = 299792. / 10000.
    ds.add_spectrum(test_fname, res, normalized=True)
    ds.add_lines(['SiII_1808', 'FeII_1611', 'FeII_2249', 'FeII_2260', 'FeII_2374'])
    ds.add_component('SiII', 2.3532, 10., 15.4)
    ds.add_component('SiII', 2.3539, 10., 15.8)
    ds.copy_components(from_ion='SiII', to_ion='FeII')
    ds.prepare_dataset(mask=False, f_lower=10.)

    popt, chi2 = ds.fit(verbose=False)
    dataset_fname = join(code_dir, 'test_dataset.hdf5')
    ds.save(dataset_fname)
    assert exists(dataset_fname)

    # Reload dataset from hdf5 dataset:
    del ds
    ds = vfit.load_dataset(dataset_fname)
    assert len(ds.all_lines) == 5, "Incorrect number of lines defined in dataset."
    assert hasattr(ds, 'best_fit'), "Dataset does not have .best_fit parameters."
    N_parameters = len(list(ds.best_fit.keys()))
    assert N_parameters == 17, "Incorrect number of parameters defined in ds.best_fit."

    # Remove temporary file:
    remove(dataset_fname)


def test_masking():
    z_sys = 2.3538
    test_fname = join(code_dir, 'test_2comp.dat')
    ds = vfit.DataSet(z_sys)
    ds.verbose = False
    ds.cheb_order = -1
    ds.velspan = 200.
    res = 299792. / 10000.
    ds.add_spectrum(test_fname, res, normalized=True)
    ds.add_line('SiII_1808')
    region, = ds.find_line('SiII_1808')
    mask = np.random.randint(0, 2, len(region.mask), dtype=bool)
    N_mask_old = np.sum(mask)
    region.set_mask(mask)

    dataset_fname = join(code_dir, 'test_dataset.hdf5')
    ds.save(dataset_fname)
    assert exists(dataset_fname)

    del ds
    ds = vfit.load_dataset(dataset_fname)
    region, = ds.find_line('SiII_1808')
    N_mask_new = np.sum(region.mask)
    assert N_mask_new == N_mask_old, "Wrong number of masked pixels."


# def test_lsf():
#     # load LSF file with dataset and check format and that the LSF is correctly retained after saving
#     pass
