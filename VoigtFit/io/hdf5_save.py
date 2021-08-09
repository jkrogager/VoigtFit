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

from ..container import regions


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


def save_hdf_dataset(ds, fname, verbose=True):
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
        hdf.attrs['redshift'] = ds.redshift
        hdf.attrs['velspan'] = ds.velspan
        if hasattr(ds, 'name'):
            hdf.attrs['name'] = ds.name
        else:
            hdf.attrs['name'] = ''
        if hasattr(ds, 'verbose'):
            hdf.attrs['verbose'] = ds.verbose
        else:
            hdf.attrs['verbose'] = True

        # .data:
        data = hdf.create_group('data')
        for num, chunk in enumerate(ds.data):
            spec = data.create_group('spec%i' % (num+1))
            spec.attrs['filename'] = ds.data_filenames[num]
            spec.attrs['res'] = chunk['res']
            spec.attrs['norm'] = chunk['norm']
            spec.attrs['nsub'] = chunk['nsub']
            spec.attrs['specID'] = chunk['specID']
            spec.create_dataset('wl', data=chunk['wl'])
            spec.create_dataset('flux', data=chunk['flux'])
            spec.create_dataset('mask', data=chunk['mask'])
            spec.create_dataset('error', data=chunk['error'])

        # .regions:
        hdf_regions = hdf.create_group('regions')
        for num, reg in enumerate(ds.regions):
            reg_group = hdf_regions.create_group('region%i' % (num+1))
            reg_group.attrs['velspan'] = reg.velspan
            reg_group.attrs['res'] = reg.res
            reg_group.attrs['normalized'] = reg.normalized
            reg_group.attrs['cont_err'] = reg.cont_err
            reg_group.attrs['new_mask'] = reg.new_mask
            reg_group.attrs['specID'] = reg.specID
            reg_group.attrs['kernel_fwhm'] = reg.kernel_fwhm
            reg_group.attrs['kernel_nsub'] = reg.kernel_nsub
            reg_group.attrs['label'] = reg.label
            reg_group.create_dataset('kernel', data=reg.kernel)
            reg_group.create_dataset('wl', data=reg.wl)
            reg_group.create_dataset('flux', data=reg.flux)
            reg_group.create_dataset('mask', data=reg.mask)
            reg_group.create_dataset('error', data=reg.err)
            lines = reg_group.create_group('lines')
            for line in reg.lines:
                lines.create_group(line.tag)
                lines[line.tag].attrs['active'] = line.active

        # .molecules:
        molecules = hdf.create_group('molecules')
        if hasattr(ds, 'molecules'):
            for molecule, items in ds.molecules.items():
                pre_array = [tuple(item) for item in items]
                band_data = np.array(pre_array,
                                     dtype=[('band', 'S8'), ('Jmax', 'i4')])
                molecules.create_dataset(molecule, data=band_data)

        fine_lines = hdf.create_group('fine_lines')
        if hasattr(ds, 'fine_lines'):
            for ground_state, lines in ds.fine_lines.items():
                # line_array = np.array(lines, dtype='str')
                line_array = [s.encode("ascii", "ignore") for s in lines]
                fine_lines.create_dataset(str(ground_state), data=line_array)

        # .components:
        components = hdf.create_group('components')
        for ion, ds_comps in ds.components.items():
            ion_group = components.create_group(ion)
            for cnum, comp in enumerate(ds_comps):
                comp_group = ion_group.create_group("comp%i" % (cnum+1))
                comp_group.attrs['z'] = comp.z
                comp_group.attrs['b'] = comp.b
                comp_group.attrs['logN'] = comp.logN
                for key, val in comp.options.items():
                    val = 'None' if val is None else val
                    comp_group.attrs[key] = val

        # .best_fit:
        if ds.best_fit is not None:
            p_opt = ds.best_fit
            best_fit = hdf.create_group('best_fit')
            for ion, comps in ds.components.items():
                params = best_fit.create_group(ion)
                for n in range(len(comps)):
                    param_group = params.create_group("comp%i" % (n+1))
                    # Save best-fit values:
                    param_group.attrs['z'] = p_opt['z%i_%s' % (n, ion)].value
                    param_group.attrs['b'] = p_opt['b%i_%s' % (n, ion)].value
                    param_group.attrs['logN'] = p_opt['logN%i_%s' % (n, ion)].value
                    # and uncertainties:
                    param_group.attrs['z_err'] = p_opt['z%i_%s' % (n, ion)].stderr
                    param_group.attrs['b_err'] = p_opt['b%i_%s' % (n, ion)].stderr
                    param_group.attrs['logN_err'] = p_opt['logN%i_%s' % (n, ion)].stderr

            # Save Chebyshev parameters:
            cheb_group = best_fit.create_group('cheb_params')
            for parname in list(ds.best_fit.keys()):
                if 'cheb_p' in parname:
                    coeff = ds.best_fit[parname]
                    cheb_par = cheb_group.create_group(parname)
                    cheb_par.attrs['value'] = coeff.value
                    cheb_par.attrs['error'] = coeff.stderr

    if verbose:
        print("Successfully saved the dataset to file: " + fname)


def load_dataset_from_hdf(fname):
    from ..container.lines import Line, lineList
    from ..container.dataset import DataSet
    """Load dataset from HDF5 file and instantiate a `VoigtFit.Dataset' class."""
    with h5py.File(fname, 'r') as hdf:
        z_sys = hdf.attrs['redshift']
        ds = DataSet(z_sys)
        ds.velspan = hdf.attrs['velspan']
        ds.verbose = hdf.attrs['verbose']
        if 'name' in hdf.attrs.keys():
            ds.set_name(hdf.attrs['name'])
        else:
            ds.set_name('')

        # Load .data:
        data = hdf['data']
        for chunk in data.values():
            # For backward compatibility:
            if 'filename' in chunk.attrs.keys():
                filename = chunk.attrs['filename']
            else:
                filename = ''
            res = chunk.attrs['res']
            norm = chunk.attrs['norm']
            if 'nsub' in chunk.attrs.keys():
                nsub = chunk.attrs['nsub']
            else:
                nsub = 1
            wl = np.array(chunk['wl'])
            flux = np.array(chunk['flux'])
            error = np.array(chunk['error'])
            if 'mask' in chunk.keys():
                mask = np.array(chunk['mask'])
            else:
                mask = np.ones_like(wl, dtype=bool)

            ds.add_data(wl, flux, res,
                        err=error, normalized=norm, nsub=nsub,
                        mask=mask, filename=filename)

        # Load .regions:
        # --- this will be deprecated in later versions
        hdf_regions = hdf['regions']
        for reg in hdf_regions.values():
            region_lines = list()
            for line_tag, line_group in reg['lines'].items():
                act = line_group.attrs['active']
                # Add check for backward compatibility:
                if line_tag in lineList['trans']:
                    line_instance = Line(line_tag, active=act)
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
            if 'specID' in reg.attrs.keys():
                specID = reg.attrs['specID']
            else:
                specID = 'sid_tmp'
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
            if 'kernel_fwhm' in reg.attrs.keys():
                Region.kernel_fwhm = reg.attrs['kernel_fwhm']
            else:
                Region.kernel_fwhm = reg.attrs['res']

            try:
                Region.label = reg.attrs['label']
            except KeyError:
                Region.label = ''

            try:
                Region.kernel_nsub = reg.attrs['kernel_nsub']
            except KeyError:
                Region.kernel_nsub = 1

            if 'kernel' in reg.keys():
                if len(reg['kernel'].shape) == 2:
                    Region.kernel = np.array(reg['kernel'])
                else:
                    Region.kernel = float(reg['kernel'][()])
            else:
                Region.kernel = reg.attrs['res']
            Region.wl = np.array(reg['wl'])
            Region.flux = np.array(reg['flux'])
            Region.mask = np.array(reg['mask'])
            Region.err = np.array(reg['error'])

            ds.regions.append(Region)

        # Load .molecules:
        molecules = hdf['molecules']
        if len(molecules) > 0:
            for molecule, band_data in molecules.items():
                bands = [[b, J] for b, J in band_data]
                ds.molecules[molecule] = bands
                # No need to call ds.add_molecule
                # lines are added above when defining the regions.

        # Load .fine_lines:
        # Older datasets do not have 'fine_lines', so add a check for backwards compatibility:
        if 'fine_lines' in hdf:
            fine_lines = hdf['fine_lines']
            if len(fine_lines) > 0:
                for ground_state, line_tags in fine_lines.items():
                    unicode_list = [s.decode('utf-8') for s in line_tags]
                    ds.fine_lines[ground_state] = unicode_list

        # Load .components:
        components = hdf['components']
        if 'best_fit' in hdf:
            # --- Prepare fit parameters  [class: lmfit.Parameters]
            ds.best_fit = Parameters()

        for ion, comps in components.items():
            ds.components[ion] = list()
            N_comps = len(comps)
            if N_comps > 0:
                for n in range(N_comps):
                    pointer = '/components/%s/comp%i' % (ion, n+1)
                    comp = hdf[pointer]
                    if 'best_fit' in hdf:
                        # If 'best_fit' exists, use the best-fit values.
                        # The naming for 'best_fit' and 'components' is parallel
                        # so one variable in components can easily be identified
                        # in the best_fit data group by replacing the path:
                        fit_pointer = pointer.replace('components', 'best_fit')
                        z = hdf[fit_pointer].attrs['z']
                        z_err = hdf[fit_pointer].attrs['z_err']
                        b = hdf[fit_pointer].attrs['b']
                        b_err = hdf[fit_pointer].attrs['b_err']
                        logN = hdf[fit_pointer].attrs['logN']
                        logN_err = hdf[fit_pointer].attrs['logN_err']

                    else:
                        z = comp.attrs['z']
                        z_err = None
                        b = comp.attrs['b']
                        b_err = None
                        logN = comp.attrs['logN']
                        logN_err = None

                    # Extract component options:
                    opts = dict()
                    for varname in ['z', 'b', 'N']:
                        tie = comp.attrs['tie_%s' % varname]
                        tie = None if tie == 'None' else tie
                        vary = comp.attrs['var_%s' % varname]
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
                                        min=0.)
                        ds.best_fit[b_name].stderr = b_err
                        ds.best_fit.add(N_name, value=logN, vary=opts['var_N'])
                        ds.best_fit[N_name].stderr = logN_err

        if 'best_fit' in hdf:
            # Now the components have been defined in `ds`, so I can use them for the loop
            # to set the parameter ties:
            for ion, comps in ds.components.items():
                for n, comp in enumerate(comps):
                    z, b, logN = comp.get_pars()
                    z_name = 'z%i_%s' % (n, ion)
                    b_name = 'b%i_%s' % (n, ion)
                    N_name = 'logN%i_%s' % (n, ion)

                    if comp.tie_z:
                        ds.best_fit[z_name].expr = comp.tie_z
                    if comp.tie_b:
                        ds.best_fit[b_name].expr = comp.tie_b
                    if comp.tie_N:
                        ds.best_fit[N_name].expr = comp.tie_N

            # Load Chebyshev parameters:
            cheb_group = hdf['best_fit/cheb_params']
            for parname, cheb_par in cheb_group.items():
                ds.best_fit.add(parname, value=cheb_par.attrs['value'])
                ds.best_fit[parname].stderr = cheb_par.attrs['error']

        return ds


def SaveDataSet(filename, ds):
    """Save dataset to HDF5 file."""
    print(" [WARNING] - this function is deprecated. Use save_dataset()")
    save_hdf_dataset(ds, filename)


def LoadDataSet(filename):
    """Load a dataset from a HDF5 file."""
    print(" [WARNING] - this function is deprecated. Use load_dataset()")
    ds = load_dataset_from_hdf(filename)
    return ds


def save_dataset(filename, ds):
    """Save dataset to HDF5 file."""
    save_hdf_dataset(ds, filename)


def load_dataset(filename):
    """Load a dataset from a HDF5 file."""
    ds = load_dataset_from_hdf(filename)
    return ds
