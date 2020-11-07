
from astropy.io import fits
import numpy as np

def get_wavelength_from_header(hdr):
    if ('CRVAL1' and 'CRPIX1' in hdr.keys()) and ('CDELT1' in hdr.keys() or 'CD1_1' in hdr.keys()):
        if 'CD1_1' in hdr.keys():
            cdelt = hdr['CD1_1']
        else:
            cdelt = hdr['CDELT1']
        crval = hdr['CRVAL1']
        crpix = hdr['CRPIX1']

        wavelength = (np.arange(hdr['NAXIS1']) - (crpix-1))*cdelt + crval

        if 'CUNIT1' in hdr.keys() and hdr['CUNIT1'] == 'nm':
            wavelength *= 10.

        return wavelength

    else:
        raise WavelengthError("Not enough information in header to create wavelength array")

class WavelengthError(Exception):
    """Raised if the header doesn't contain the proper wavelength solution: CRVAL, CD etc."""
    pass

class FormatError(Exception):
    """Raised when the FITS format is not understood"""
    pass


def load_fits_spectrum(fname):
    HDU = fits.open(fname)
    primhdr = HDU[0].header
    if primhdr['NAXIS'] == 1:
        if len(HDU) == 1:
            raise FormatError("Could not find Error Array")

        elif len(HDU) == 2:
            data = HDU[0].data
            data_hdr = HDU[0].header
            error = HDU[1].data
            mask = np.ones_like(data, dtype=bool)

        elif len(HDU) > 2:
            data_in_hdu = False
            for extname in ['FLUX', 'SCI', 'FLAM', 'FNU']:
                if extname in HDU:
                    data = HDU[extname].data
                    data_hdr = HDU[extname].header
                    data_in_hdu = True
            if not data_in_hdu:
                raise FormatError("Could not find Flux Array")

            error_in_hdu = False
            for extname in ['ERR', 'SIG', 'SIGMA', 'ERROR', 'IVAR', 'VAR']:
                if extname in HDU:
                    if extname == 'IVAR':
                        error = 1./np.sqrt(HDU[extname].data)
                    elif extname == 'VAR':
                        error = np.sqrt(HDU[extname].data)
                    else:
                        error = HDU[extname].data
                    error_in_hdu = True
            if not error_in_hdu:
                raise FormatError("Could not find Error Array")

            # Does the spectrum contain a pixel mask?
            extname = 'MASK'
            if extname in HDU:
                mask = HDU[extname].data
            else:
                mask = np.ones_like(data, dtype=bool)

        try:
            wavelength = get_wavelength_from_header(primhdr)
            return wavelength, data, error, mask
        except WavelengthError:
            wavelength = get_wavelength_from_header(data_hdr)
            return wavelength, data, error, mask
        else:
            raise FormatError

    elif primhdr['NAXIS'] == 2:
        print(" Not a 1-dimensional spectrum!!")
        raise FormatError("The data seems to be a 2D image of shape: {}".format(HDU[0].data.shape))

    elif primhdr['NAXIS'] == 3:
        # IRAF Format:
        data_array = HDU[0].data
        data = data_array[0][0]
        error = data_array[3][0]
        mask = np.ones_like(data, dtype=bool)
        wavelength = get_wavelength_from_header(primhdr)
        return wavelength, data, error, mask

    else:
        if isinstance(HDU[1], fits.BinTableHDU) or isinstance(HDU[1], fits.TableHDU):
            tbdata = HDU[1].data
            wl_in_table = False
            for colname in ['wl', 'lam', 'lambda', 'loglam', 'wave', 'wavelength']:
                table_names = [name.lower() for name in tbdata.names]
                if colname in table_names:
                    wl_in_table = True
                    if colname == 'loglam':
                        wavelength = 10**tbdata[colname]
                    else:
                        wavelength = tbdata[colname]

            data_in_table = False
            for colname in ['data', 'spec', 'flux', 'flam', 'fnu', 'flux_density']:
                if colname in table_names:
                    data_in_table = True
                    data = tbdata[colname]

            error_in_table = False
            for colname in ['err', 'sig', 'error', 'ivar', 'sigma', 'var']:
                if colname in table_names:
                    error_in_table = True
                    if colname == 'ivar':
                        error = 1./np.sqrt(tbdata[colname])
                    elif colname == 'var':
                        error = np.sqrt(tbdata[colname])
                    else:
                        error = tbdata[colname]

            all_arrays_found = wl_in_table and data_in_table and error_in_table
            if not all_arrays_found:
                raise FormatError("Could not find all data columns in the table")

            mask = np.ones_like(data)
            if 'mask' in tbdata.names:
                mask = tbdata[colname]
            mask = mask.astype('bool')

            return wavelength.flatten(), data.flatten(), error.flatten(), mask

        elif len(HDU) == 2:
            raise FormatError("Could not find Error Array")

        elif len(HDU) > 2:
            data_in_hdu = False
            for extname in ['FLUX', 'SCI', 'FLAM', 'FNU']:
                if extname in HDU:
                    data = HDU[extname].data
                    data_hdr = HDU[extname].header
                    data_in_hdu = True
            if not data_in_hdu:
                raise FormatError("Could not find Flux Array")

            err_in_hdu = False
            for extname in ['ERR', 'SIG']:
                if extname in HDU:
                    error = HDU[extname].data
                    err_in_hdu = True
            if not err_in_hdu:
                raise FormatError("Could not find Error Array")

            if 'MASK' in HDU:
                mask = HDU['MASK'].data
            else:
                mask = np.ones_like(data, dtype=bool)

            try:
                wavelength = get_wavelength_from_header(data_hdr)
            except WavelengthError:
                wavelength = get_wavelength_from_header(primhdr)
            else:
                raise WavelengthError
            return wavelength, data, error, mask
