# -*- coding: UTF-8 -*-
"""
A module to save a VoigtFit dataset to a file.
The files are saved in HDF5 format to allow easy portability.

The module also contains a function to sconvert
the older pickled datasets to the new HDF5 format.
"""
__author__ = 'Jens-Kristian Krogager'

import numpy as np
from os.path import splitext, basename
import pickle
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import h5py
from lmfit import Parameters

import regions
import dataset


def dataset_to_hdf(fname):
    """ Convert a pickled dataset to the HDF5 format"""
    f = open(fname, 'rb')
    ds = pickle.load(f)
    f.close()
    f_base = basename(fname)
    root, ext = splitext(f_base)
    hdf_fname = root + '.vfit.h5'
    save_hdf_dataset(ds, hdf_fname)
    return hdf_fname


def save_hdf_dataset(dataset, fname, verbose=True):
    """
    Save VoigtFit.dataset to a HDF5 file.
    The function maps the internal data to a HDF5 data model.
    """

    if splitext(fname)[1] == '.hdf5':
        pass
    else:
        fname += '.hdf5'

    with h5py.File(fname, 'w') as hdf:

        # set main attributes:
        hdf.attrs.create('redshift', dataset.redshift)
        hdf.attrs.create('velspan', dataset.velspan)
        if hasattr(dataset, 'name'):
            hdf.attrs.create('name', dataset.name)
        else:
            hdf.attrs.create('name', '')
        if hasattr(dataset, 'verbose'):
            hdf.attrs.create('verbose', dataset.verbose)
        else:
            hdf.attrs.create('verbose', True)

        # .data:
        data = hdf.create_group('data')
        for num, chunk in enumerate(dataset.data):
            spec = data.create_group('spec%i' % (num+1))
            spec.attrs.create('res', chunk['res'])
            spec.attrs.create('norm', chunk['norm'])
            spec.create_dataset('wl', data=chunk['wl'])
            spec.create_dataset('flux', data=chunk['flux'])
            spec.create_dataset('error', data=chunk['error'])

        # .regions:
        hdf_regions = hdf.create_group('regions')
        for num, reg in enumerate(dataset.regions):
            reg_group = hdf_regions.create_group('region%i' % (num+1))
            reg_group.attrs.create('velspan', reg.velspan)
            reg_group.attrs.create('res', reg.res)
            reg_group.attrs.create('normalized', reg.normalized)
            reg_group.attrs.create('cont_err', reg.cont_err)
            reg_group.attrs.create('new_mask', reg.new_mask)
            reg_group.attrs.create('specID', reg.specID)
            reg_group.attrs.create('kernel_fwhm', reg.kernel_fwhm)
            reg_group.attrs.create('kernel_nsub', reg.kernel_nsub)
            reg_group.attrs.create('label', reg.label)
            reg_group.create_dataset('kernel', data=reg.kernel)
            reg_group.create_dataset('wl', data=reg.wl)
            reg_group.create_dataset('flux', data=reg.flux)
            reg_group.create_dataset('mask', data=reg.mask)
            reg_group.create_dataset('error', data=reg.err)
            lines = reg_group.create_group('lines')
            for line in reg.lines:
                lines.create_group(line.tag)
                lines[line.tag].attrs.create('active', line.active)

        # .molecules:
        molecules = hdf.create_group('molecules')
        if hasattr(dataset, 'molecules'):
            for molecule, items in dataset.molecules.items():
                pre_array = [tuple(item) for item in items]
                band_data = np.array(pre_array,
                                     dtype=[('band', 'S8'), ('Jmax', 'i4')])
                molecules.create_dataset(molecule, data=band_data)

        # .components:
        components = hdf.create_group('components')
        for ion, comps in dataset.components.items():
            ion_group = components.create_group(ion)
            if len(comps) > 0:
                for cnum, comp in enumerate(comps):
                    comp_group = ion_group.create_group("comp%i" % (cnum+1))
                    comp_group.create_dataset('z', data=comp[0])
                    comp_group.create_dataset('b', data=comp[1])
                    comp_group.create_dataset('logN', data=comp[2])
                    for varname in ['z', 'b', 'N']:
                        if varname == 'N':
                            tie_constraint = comp[3]['tie_%s' % varname]
                            tie_constraint = 'None' if tie_constraint is None else tie_constraint
                            comp_group['logN'].attrs.create('tie_%s' % varname, tie_constraint)
                            comp_group['logN'].attrs.create('var_%s' % varname, comp[3]['var_%s' % varname])
                        else:
                            tie_constraint = comp[3]['tie_%s' % varname]
                            tie_constraint = 'None' if tie_constraint is None else tie_constraint
                            comp_group[varname].attrs.create('tie_%s' % varname, tie_constraint)
                            comp_group[varname].attrs.create('var_%s' % varname, comp[3]['var_%s' % varname])

        # .best_fit:
        if dataset.best_fit is not None:
            p_opt = dataset.best_fit
            best_fit = hdf.create_group('best_fit')
            for ion, comps in dataset.components.items():
                params = best_fit.create_group(ion)
                for n in range(len(comps)):
                    param_group = params.create_group("comp%i" % (n+1))
                    param_group.create_dataset('z', data=p_opt['z%i_%s' % (n, ion)].value)
                    param_group.create_dataset('b', data=p_opt['b%i_%s' % (n, ion)].value)
                    param_group.create_dataset('logN', data=p_opt['logN%i_%s' % (n, ion)].value)

                    param_group['z'].attrs.create('error', p_opt['z%i_%s' % (n, ion)].stderr)
                    param_group['b'].attrs.create('error', p_opt['b%i_%s' % (n, ion)].stderr)
                    param_group['logN'].attrs.create('error', p_opt['logN%i_%s' % (n, ion)].stderr)

    if verbose:
        print "Successfully saved the dataset to file: " + fname


def load_dataset_from_hdf(fname):
    """Load dataset from HDF5 file and instantiate a `VoigtFit.Dataset' class."""
    with h5py.File(fname, 'r') as hdf:
        z_sys = hdf.attrs['redshift']
        ds = dataset.DataSet(z_sys)
        ds.velspan = hdf.attrs['velspan']
        ds.verbose = hdf.attrs['verbose']
        if 'name' in hdf.attrs.keys():
            ds.set_name(hdf.attrs['name'])
        else:
            ds.set_name('')

        # Load .data:
        data = hdf['data']
        for chunk in data.values():
            res = chunk.attrs['res']
            norm = chunk.attrs['norm']
            ds.add_data(chunk['wl'].value, chunk['flux'].value, res,
                        err=chunk['error'].value, normalized=norm)

        # Load .regions:
        # --- this will be deprecated in later versions
        hdf_regions = hdf['regions']
        for reg in hdf_regions.values():
            region_lines = list()
            for line_tag, line_group in reg['lines'].items():
                act = line_group.attrs['active']
                # Add check for backward compatibility:
                if line_tag in dataset.lineList['trans']:
                    line_instance = dataset.Line(line_tag, active=act)
                    region_lines.append(line_instance)
                    ds.all_lines.append(line_tag)
                    ds.lines[line_tag] = line_instance
                else:
                    print(" [WARNING] - Anomaly detected for line:")
                    print("             %s" % line_tag)
                    print(" I suspect that the atomic linelist has changed...")
                    print("")

            # Instantiate the Region Class with the first Line:
            line_init = region_lines[0]
            v = reg.attrs['velspan']
            specID = reg.attrs['specID']
            Region = regions.Region(v, specID, line_init)
            if len(region_lines) == 1:
                # The first and only line has already been loaded
                pass

            elif len(region_lines) > 1:
                # Load the rest of the lines:
                for line in region_lines[1:]:
                    Region.lines.append(line)
            else:
                err_msg = "Something went wrong in this region: %s. No lines are defined!" % str(reg.name)
                raise ValueError(err_msg)

            # Set region data and attributes:
            Region.res = reg.attrs['res']
            Region.normalized = reg.attrs['normalized']
            Region.cont_err = reg.attrs['cont_err']
            Region.new_mask = reg.attrs['new_mask']
            Region.kernel_fwhm = reg.attrs['kernel_fwhm']
            try:
                Region.label = reg.attrs['label']
            except KeyError:
                Region.label = ''
            try:
                Region.kernel_nsub = reg.attrs['kernel_nsub']
            except KeyError:
                Region.kernel_nsub = 1

            Region.kernel = reg['kernel'].value
            Region.wl = reg['wl'].value
            Region.flux = reg['flux'].value
            Region.mask = reg['mask'].value
            Region.err = reg['error'].value

            ds.regions.append(Region)

        # Load .molecules:
        molecules = hdf['molecules']
        if len(molecules) > 0:
            for molecule, band_data in molecules.items():
                bands = [[b, J] for b, J in band_data]
                ds.molecules[molecule] = bands
                # No need to call ds.add_molecule
                # lines are added above when defining the regions.

        # Load .components:
        components = hdf['components']
        if 'best_fit' in hdf:
            # --- Prepare fit parameters  [class: lmfit.Parameters]
            ds.best_fit = Parameters()

        for ion, comps in components.items():
            ds.components[ion] = list()
            if len(comps) > 0:
                for n, comp in enumerate(comps.values()):
                    if 'best_fit' in hdf:
                        # If 'best_fit' exists, use the best-fit values.
                        # The naming for 'best_fit' and 'components' is parallel
                        # so one variable in components can easily be identified
                        # in the best_fit data group by replacing the path:
                        pointer = comp.name
                        fit_pointer = pointer.replace('components', 'best_fit')
                        z = hdf[fit_pointer+'/z'].value
                        z_err = hdf[fit_pointer+'/z'].attrs['error']
                        b = hdf[fit_pointer+'/b'].value
                        b_err = hdf[fit_pointer+'/b'].attrs['error']
                        logN = hdf[fit_pointer+'/logN'].value
                        logN_err = hdf[fit_pointer+'/logN'].attrs['error']

                    else:
                        z = comp['z'].value
                        z_err = None
                        b = comp['b'].value
                        b_err = None
                        logN = comp['logN'].value
                        logN_err = None

                    # Extract component options:
                    opts = dict()
                    for varname in ['z', 'b', 'N']:
                        if varname == 'N':
                            hdf_name = 'logN'
                        else:
                            hdf_name = varname

                        tie = comp[hdf_name].attrs['tie_%s' % varname]
                        tie = None if tie == 'None' else tie
                        vary = comp[hdf_name].attrs['var_%s' % varname]
                        opts['tie_%s' % varname] = tie
                        opts['var_%s' % varname] = vary

                    # Add component to DataSet class:
                    ds.add_component(ion, z, b, logN, **opts)

                    if 'best_fit' in hdf:
                        # Add Parameters to DataSet.best_fit:
                        z_name = 'z%i_%s' % (n, ion)
                        b_name = 'b%i_%s' % (n, ion)
                        N_name = 'logN%i_%s' % (n, ion)
                        ds.best_fit.add(z_name, value=z, vary=opts['var_z'])
                        ds.best_fit[z_name].stderr = z_err
                        ds.best_fit.add(b_name, value=b, vary=opts['var_b'],
                                        min=0., max=500.)
                        ds.best_fit[b_name].stderr = b_err
                        ds.best_fit.add(N_name, value=logN, vary=opts['var_N'],
                                        min=0., max=40.)
                        ds.best_fit[N_name].stderr = logN_err

        if 'best_fit' in hdf:
            # Now the components have been defined in ds, so I can use them for the loop
            # to set the parameter ties:
            for ion, comps in ds.components.items():
                for n, comp in enumerate(comps):
                    z, b, logN, opts = comp
                    z_name = 'z%i_%s' % (n, ion)
                    b_name = 'b%i_%s' % (n, ion)
                    N_name = 'logN%i_%s' % (n, ion)

                    if opts['tie_z']:
                        ds.best_fit[z_name].expr = opts['tie_z']
                    if opts['tie_b']:
                        ds.best_fit[b_name].expr = opts['tie_b']
                    if opts['tie_N']:
                        ds.best_fit[N_name].expr = opts['tie_N']

        return ds
