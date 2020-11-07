# jkrogager/fitsutil/src/fits_input.py
__author__ = "Jens-Kristian Krogager"

import warnings
from astropy.io import fits
import numpy as np


class MultipleSpectraWarning(Warning):
    """Throw warning when several FITS Table extensions or multiple IRAF objects are present"""
    pass

class WavelengthError(Exception):
    """Raised if the header doesn't contain the proper wavelength solution: CRVAL, CD etc."""
    pass

class FormatError(Exception):
    """Raised when the FITS format is not understood"""
    pass


def get_wavelength_from_header(hdr):
    """
    Obtain wavelength solution from Header keywords:

        Wavelength_i = CRVAL1 + (PIXEL_i - (CRPIX1-1)) * CDELT1

    CDELT1 can be CD1_1 as well.

    If all these keywords are not present in the header, raise a WavelengthError

    Returns
    -------
    wavelength : np.array (float)
        Numpy array of wavelengths.
    """
    if ('CRVAL1' and 'CRPIX1' in hdr.keys()) and ('CDELT1' in hdr.keys() or 'CD1_1' in hdr.keys()):
        if 'CD1_1' in hdr.keys():
            cdelt = hdr['CD1_1']
        else:
            cdelt = hdr['CDELT1']
        crval = hdr['CRVAL1']
        crpix = hdr['CRPIX1']

        wavelength = (np.arange(hdr['NAXIS1']) - (crpix-1))*cdelt + crval

        # if 'CUNIT1' in hdr.keys() and hdr['CUNIT1'] == 'nm':
        #     wavelength *= 10.

        return wavelength

    else:
        raise WavelengthError("Not enough information in header to create wavelength array")


# -- These names are used to define proper column names for Wavelength, Flux and Error:
wavelength_column_names = ['wl', 'lam', 'lambda', 'loglam', 'wave', 'wavelength']
flux_column_names = ['data', 'spec', 'flux', 'flam', 'fnu', 'flux_density']
error_column_names = ['err', 'sig', 'error', 'ivar', 'sigma', 'var']

# -- These names are used to define proper ImageHDU names for Flux and Error:
flux_HDU_names = ['FLUX', 'SCI', 'FLAM', 'FNU']
error_HDU_names = ['ERR', 'ERRS', 'SIG', 'SIGMA', 'ERROR', 'ERRORS', 'IVAR', 'VAR']


def get_spectrum_fits_table(tbdata):
    """
    Scan the TableData for columns containing wavelength, flux, error and mask.
    All arrays of {wavelength, flux and error} must be present.

    The columns are identified by matching predefined column names:
        For wavelength arrays: %(WL_COL_NAMES)r

        For flux arrays: %(FLUX_COL_NAMES)r

        For error arrays: %(ERR_COL_NAMES)r

    Returns
    -------
    wavelength : np.array (float)
        Numpy array of wavelengths.
    data : np.array (float)
        Numpy array of flux density.
    error : np.array (float)
        Numpy array of uncertainties on the flux density
    mask : np.array (bool)
        Numpy boolean array of pixel mask. `True` if the pixel is 'good',
        `False` if the pixel is bad and should not be used.
    """
    wl_in_table = False
    for colname in wavelength_column_names:
        table_names = [name.lower() for name in tbdata.names]
        if colname in table_names:
            wl_in_table = True
            if colname == 'loglam':
                wavelength = 10**tbdata[colname]
            else:
                wavelength = tbdata[colname]

    data_in_table = False
    for colname in flux_column_names:
        if colname in table_names:
            data_in_table = True
            data = tbdata[colname]

    error_in_table = False
    for colname in error_column_names:
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

    mask = np.ones_like(data, dtype=bool)
    if 'mask' in tbdata.names:
        mask = tbdata[colname]

    return wavelength.flatten(), data.flatten(), error.flatten(), mask

# Hack the doc-string of the function to input the variable names:
output_column_names = {'WL_COL_NAMES': wavelength_column_names,
                       'FLUX_COL_NAMES': flux_column_names,
                       'ERR_COL_NAMES': error_column_names}
get_spectrum_fits_table.__doc__ = get_spectrum_fits_table.__doc__ % output_column_names


def get_spectrum_hdulist(HDUlist):
    """
    Scan the HDUList for names that match one of the defined names for
    flux and flux error. If one is missing, the code will raise a FormatError.

    The ImageHDUs are identified by matching predefined Extension names:

        For flux arrays: %(FLUX_HDU_NAMES)r

        For error arrays: %(ERR_HDU_NAMES)r

    Returns
    -------
    data : np.array (float)
        Numpy array of flux density.
    error : np.array (float)
        Numpy array of uncertainties on the flux density
    mask : np.array (bool)
        Numpy boolean array of pixel mask. `True` if the pixel is 'good',
        `False` if the pixel is bad and should not be used.
    data_hdr : astropy.io.fits.Header
        The FITS Header of the given data extension.
        The wavelength information should be contained in this header.
    """
    data_in_hdu = False
    for extname in flux_HDU_names:
        if extname in HDUlist:
            data = HDUlist[extname].data
            data_hdr = HDUlist[extname].header
            data_in_hdu = True
    if not data_in_hdu:
        raise FormatError("Could not find Flux Array")

    error_in_hdu = False
    for extname in error_HDU_names:
        if extname in HDUlist:
            if extname == 'IVAR':
                error = 1./np.sqrt(HDUlist[extname].data)
            elif extname == 'VAR':
                error = np.sqrt(HDUlist[extname].data)
            else:
                error = HDUlist[extname].data
            error_in_hdu = True
    if not error_in_hdu:
        raise FormatError("Could not find Error Array")

    # Does the spectrum contain a pixel mask?
    extname = 'MASK'
    if extname in HDUlist:
        mask = HDUlist[extname].data
    else:
        mask = np.ones_like(data, dtype=bool)

    return data, error, mask, data_hdr

# Hack the doc-string of the function to input the variable names:
output_hdu_names = {'FLUX_HDU_NAMES': flux_HDU_names,
                    'ERR_HDU_NAMES': error_HDU_names}
get_spectrum_hdulist.__doc__ = get_spectrum_hdulist.__doc__ % output_hdu_names


def load_fits_spectrum(fname, ext=None, iraf_obj=None):
    """
    Flexible inference of spectral data from FITS files.
    The function allows to read a large number of spectral formats including
    FITS tables, multi extension FITS ImageHDUs, or IRAF like arrays

    Parameters
    ----------
    fname : string
        Filename for the FITS file to open
    ext : int or string
        Extension number (int) or Extension Name (string)
    iraf_obj : int
        Index of the IRAF array, e.g. the flux is found at index: [spectral_pixels, iraf_obj, 0]

    Returns
    -------
    wavelength : np.array (float)
        Numpy array of wavelengths.
    data : np.array (float)
        Numpy array of flux density.
    error : np.array (float)
        Numpy array of uncertainties on the flux density
    mask : np.array (bool)
        Numpy boolean array of pixel mask. `True` if the pixel is 'good',
        `False` if the pixel is bad and should not be used.
    """
    HDUlist = fits.open(fname)
    primhdr = HDUlist[0].header
    primary_has_data = HDUlist[0].data is not None
    if primary_has_data:
        if primhdr['NAXIS'] == 1:
            if len(HDUlist) == 1:
                raise FormatError("Only one extension: Could not find both Flux and Error Arrays")

            elif len(HDUlist) == 2:
                data = HDUlist[0].data
                data_hdr = HDUlist[0].header
                error = HDUlist[1].data
                mask = np.ones_like(data, dtype=bool)

            elif len(HDUlist) > 2:
                data, error, mask, data_hdr = get_spectrum_hdulist(HDUlist)

            try:
                wavelength = get_wavelength_from_header(primhdr)
                return wavelength, data, error, mask
            except WavelengthError:
                wavelength = get_wavelength_from_header(data_hdr)
                return wavelength, data, error, mask
            else:
                raise FormatError("Could not find Wavelength Array")

        elif primhdr['NAXIS'] == 2:
            raise FormatError("The data seems to be a 2D image of shape: {}".format(HDUlist[0].data.shape))

        elif primhdr['NAXIS'] == 3:
            # This could either be a data cube (such as SINFONI / MUSE)
            # or IRAF format:
            IRAF_in_hdr = 'IRAF' in primhdr.__repr__()
            has_CRVAL3 = 'CRVAL3' in primhdr.keys()
            if IRAF_in_hdr and not has_CRVAL3:
                # This is most probably an IRAF spectrum file:
                #  (N_pixels, N_objs, 4)
                #  The 4 axes are [flux, flux_noskysub, sky_flux, error]
                data_array = HDUlist[0].data
                if iraf_obj is None:
                    # Use the first object by default
                    iraf_obj = 0
                    # If other objects are present, throw a warning:
                    if data_array.shape[1] > 1:
                        warnings.warn("More than one object detected in the file", MultipleSpectraWarning)

                data = data_array[0][iraf_obj]
                error = data_array[3][iraf_obj]
                mask = np.ones_like(data, dtype=bool)
                wavelength = get_wavelength_from_header(primhdr)
                return wavelength, data, error, mask
            else:
                raise FormatError("The data seems to be a 3D cube of shape: {}".format(HDUlist[0].data.shape))

    else:
        is_fits_table = isinstance(HDUlist[1], fits.BinTableHDU) or isinstance(HDUlist[1], fits.TableHDU)
        if is_fits_table:
            if ext:
                tbdata = HDUlist[ext].data
            else:
                tbdata = HDUlist[1].data

            has_multi_extensions = len(HDUlist) > 2
            if has_multi_extensions and (ext is None):
                warnings.warn("More than one data extension detected in the file", MultipleSpectraWarning)
            wavelength, data, error, mask = get_spectrum_fits_table(tbdata)
            return wavelength, data, error, mask

        elif len(HDUlist) == 2:
            raise FormatError("Only one data extension: Could not find both Flux and Error Arrays")

        elif len(HDUlist) > 2:
            data, error, mask, data_hdr = get_spectrum_hdulist(HDUlist)
            try:
                wavelength = get_wavelength_from_header(data_hdr)
            except WavelengthError:
                wavelength = get_wavelength_from_header(primhdr)
            else:
                raise FormatError("Could not find Wavelength Array")
            return wavelength, data, error, mask
