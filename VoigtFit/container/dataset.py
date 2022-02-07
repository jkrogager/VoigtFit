# -*- coding: UTF-8 -*-

__author__ = 'Jens-Kristian Krogager'

import numpy as np
import matplotlib.pyplot as plt
import copy
import warnings
import re

from astropy.io import fits
import astropy.units as u
from lmfit import Parameters, Minimizer

from VoigtFit.utils import Asplund
from VoigtFit.container.components import Component
from VoigtFit.io.fits_input import load_fits_spectrum, FormatError, MultipleSpectraWarning
from VoigtFit.io import hdf5_save
from VoigtFit.funcs.limits import match_ion_state, match_ion_state_all, tau_percentile, tau_noise_range, equivalent_width
from VoigtFit.utils import line_complexes
from VoigtFit.utils.line_complexes import fine_structure_complexes
from VoigtFit.container.lines import Line, lineList
from VoigtFit.utils import molecules
from VoigtFit.io import output
from VoigtFit.container.regions import Region, load_lsf
from VoigtFit.utils import terminal_attributes as term
from VoigtFit.funcs.voigt import evaluate_profile, evaluate_continuum

from collections import namedtuple

EquivalentWidth = namedtuple('EquivalentWidth', ['W_rest', 'W_err', 'logN', 'logN_err', 'logN_limit', 'line', 'sigma'])

myfloat = np.float64

def air2vac(wavelength, unit='AA'):
    """
    Implements the air to vacuum wavelength conversion described in eqn 65 of
    Griesen 2006

    wavelength : array or float
        Input wavelength in air.

    unit : string   [default='AA']
        Units of the input wavelengths, default is Angstrom (AA).

    Returns
    -------
    The vacuum converted array of wavelength in the same units
    """
    wl = wavelength*u.Unit(unit)
    wlum = wl.to(u.um).value
    n_a = 1. + 1e-6*(287.6155 + 1.62887/wlum**2 + 0.01360/wlum**4)
    return n_a * wavelength

def vac2air(wavelength, unit='AA'):
    """
    Implements the air to vacuum wavelength conversion described in eqn 65 of
    Griesen 2006

    wavelength : array or float
        Input wavelength in vacuum.

    unit : string   [default='AA']
        Units of the input wavelengths, default is Angstrom (AA).

    Returns
    -------
    The air converted array of wavelength in the same units
    """
    wl = wavelength*u.Unit(unit)
    wlum = wl.to(u.um).value
    n_a = 1. + 1e-6*(287.6155 + 1.62887/wlum**2 + 0.01360/wlum**4)
    return wavelength / n_a


def mask_vel(dataset, line_tag, v1, v2):
    regions_of_line = dataset.find_line(line_tag)
    for reg in regions_of_line:
        l0 = dataset.lines[line_tag].l0
        z = dataset.redshift
        vel = (reg.wl/(l0*(z+1)) - 1.)*299792.458
        mask = (vel > v1)*(vel < v2)
        new_mask = reg.mask.copy()
        new_mask = new_mask * ~mask
        reg.set_mask(new_mask)


def calculate_velocity_bin_size(x):
    """Calculate the bin size of *x* in velocity units."""
    log_x = np.logspace(np.log10(x.min()), np.log10(x.max()), len(x))
    return np.diff(log_x)[0] / log_x[0] * 299792.458


def verify_lsf(res, wl):
    """Check that the LSF file covers the whole spectral range of `wl`"""
    lsf_wl = np.genfromtxt(res, max_rows=1)
    covering = (lsf_wl.min() <= wl.min()) * (lsf_wl.max() >= wl.max())
    if not covering:
        err_msg = "The given LSF file does not cover the wavelength range!"
        raise ValueError(err_msg)



# --- Definition of main class *DataSet*:
class DataSet(object):
    def __init__(self, redshift, name=''):
        """
        Main class of the package ``VoigtFit``. The DataSet handles all the major parts of the fit.
        Spectral data must be added using the :meth:`add_data <VoigtFit.DataSet.add_data>` method.
        Hereafter the absorption lines to be fitted are added to the DataSet using the
        :meth:`add_line <VoigtFit.DataSet.add_line>` or
        :meth:`add_many_lines <VoigtFit.DataSet.add_many_lines>` methods.
        Lastly, the components of each element is defined using the
        :meth:`add_component <VoigtFit.DataSet.add_component>` method.
        When all lines and components have been defined, the DataSet must be prepared by
        calling the :meth:`prepare_dataset <VoigtFit.DataSet.prepare_dataset>`
        method and subsequently, the lines can be fitted using
        the :meth:`fit <VoigtFit.DataSet.fit>` method.

        .. rubric:: Attributes

        redshift : float
            Systemic redshift of the absorption system.

        name : str   [default = '']
            The name of the DataSet, this will be used for saving the dataset to a file structure.

        verbose : bool   [default = True]
            If `False`, the printed information statements will be suppressed.

        data : list(data_chunks)
            A list of *data chunks* defined for the dataset. A *data chunk* is
            a dictionary with keys 'wl', 'flux', 'error', 'res', 'norm'.
            See :meth:`DataSet.add_data <VoigtFit.DataSet.add_data>`.

        lines : dict
            A dictionary holding pairs of defined (*line_tag* : :class:`dataset.Line`)

        all_lines : list(str)
            A list of all the defined *line tags* for easy look-up.

        molecules : dict
            A dictionary holding a list of the defined molecular bands and Jmax
            for each molecule:
            ``{molecule* : [[band1, Jmax1], [band2, Jmax2], etc...]}``

        regions : list(:class:`regions.Region`)
            A list of the fitting regions.

        cheb_order : int   [default = -1]
            The maximum order of Chebyshev polynomials to use for the continuum
            fitting in each region. If negative, the Chebyshev polynomials will
            not be included in the fit.

        norm_method : str   [default = 'linear']
            Default normalization method to use for interactive normalization
            if Chebyshev polynomial fitting should not be used.

        components : dict
            A dictionary of components for each *ion* defined:
            (*ion* : [z, b, logN, options]). See :meth:`DataSet.add_component
            <VoigtFit.DataSet.add_component>`.

        velspan : float, Tuple(float, float)  [default = 400]
            The default velocity range to use for the definition
            of fitting regions.

        ready2fit : bool   [default = False]
            This attribute is checked before fitting the dataset. Only when
            the attribute has been set to `True` can the dataset be fitted.
            This will be toggled after a successful run of
            :meth:`DataSet.prepare_dataset <VoigtFit.DataSet.prepare_dataset>`.

        best_fit : `lmfit.Parameters`_   [default = None]
            Best-fit parameters from lmfit_.
            This attribute will be `None` until the dataset has been fitted.

        pars : `lmfit.Parameters`_   [default = None]
            Placeholder for the fit parameters initiated before the fit.
            The parameters will be defined during the call to :meth:`DataSet.prepare_dataset
            <VoigtFit.DataSet.prepare_dataset>` based on the defined components.

        static_variables : `lmfit.Parameters`_
            Parameter dictionary that holds fit variables other than those related to
            components of absorption lines and continuum parameters.


        .. _lmfit.Parameters: https://lmfit.github.io/lmfit-py/parameters.html
        .. _lmfit: https://lmfit.github.io/lmfit-py/

        """
        # Define the systemic redshift
        self.redshift = redshift

        # container for data chunks to be fitted
        # data should be added by calling method 'add_data'
        self.data = list()
        self.data_filenames = list()

        self.verbose = True

        # container for absorption lines. Each line is defined as a class 'Line'.
        # a dictionary containing a Line class for each line-tag key:
        self.lines = dict()
        # a list containing all the line-tags defined. The same as lines.keys()
        self.all_lines = list()
        # a dictionary conatining a list of transitions for each fine-structure ground state:
        # Ex: self.fine_lines = {'CI_1656': ['CI_1656', 'CIa_1656', 'CIa_1657', ...]}
        self.fine_lines = dict()
        # a dictionary conatining a list of bands defined for each molecule:
        # Ex: self.molecules = {'CO': ['AX(0-0)', 'AX(1-0)']}
        self.molecules = dict()

        # container for the fitting regions containing Lines
        # each region is defined as a class 'Region'
        self.regions = list()
        self.cheb_order = -1
        self.norm_method = 'linear'

        # Number of components in each ion
        self.components = dict()

        # Define default velocity span for fitting region
        self._velspan = (-400., 400.)  # km/s

        self.ready2fit = False
        self.best_fit = None
        self.minimizer = None
        self.pars = None
        self.static_variables = Parameters()
        self.name = name


    def check_velspan(self, velspan):
        if hasattr(velspan, '__iter__'):
            if len(velspan) != 2:
                raise ValueError("argument 'velspan' must have two values! not %i" % len(velspan))
        elif velspan is None:
            velspan = self._velspan
        else:
            velspan = (-1.*np.abs(velspan), np.abs(velspan))
        return velspan

    @property
    def velspan(self):
        return self._velspan

    @velspan.setter
    def velspan(self, value):
        if value is None:
            print("`velspan` must be a number or a tuple of two numbers, not None!")
            return
        self._velspan = self.check_velspan(value)

    def set_velspan(self, value):
        if value is None:
            print("`velspan` must be a number or a tuple of two numbers, not None!")
            return
        self._velspan = self.check_velspan(value)

    def set_name(self, name):
        """Set the name of the DataSet. This parameter is used when saving the dataset."""
        self.name = name

    def get_name(self):
        """Returns the name of the DataSet."""
        return self.name

    def add_spectrum(self, filename, res, airORvac='vac', normalized=False, mask_array=None, mask=None, use_mask=True, continuum=None, nsub=1, verbose=False, ext=None):
        """
        Add spectral data from ASCII file (three or four columns accepted).
        This is a wrapper of the method `add_data`.

        Parameters
        ----------
        filename : string
            Filename of the ASCII spectrum containing at least 3 columns:
            (wavelength, flux, error). A fourth column may be included
            containing a spectral mask.

        res : float or string
            Spectral resolution either given in km/s  (c/R),
            which is assumed to be constant over the whole spectrum,
            or as a string referring to a file containing the detailed
            line-spread function for the given spectrum.
            See details in the documentation.

        airORvac : string  {'vac' or 'air'}
            Defines whether the input wavelengths are in 'air' or 'vacuum' units. If given as 'air'
            the wavelengths will be converted to 'vacuum'.

        normalized : bool   [default = False]
            If the input spectrum is normalized this should be given as True
            in order to skip normalization steps.

        mask_array : array, shape (n)
            Boolean/int array defining the fitting regions.
            Only pixels with mask=True/1 will be included in the fit.

        mask : Deprecated, use `mask_array`

        use_mask : bool   [default = True]
            If False, do not pass the mask array to the dataset. This may be useful if the mask defined in the spectrum
            is not appropriate for the fitting purposes.

        continuum : array, shape (n)
            Continuum spectrum. The input spectrum will be normalized using
            this continuum spectrum.

        nsub : integer   [default = 1]
            Kernel subsampling factor relative to the data.
            This is only used if the resolution is given as a LSF file.

        verbose : bool   [default = False]
            Print status messages.

        ext : int or string
            Index or name of the extension in the HDU List. Only used if the input is a FITS file.
        """
        if mask is not None:
            warnings.warn("The keyword mask is deprecated. Use 'mask_array' instead", DeprecationWarning)
            mask_array = mask

        file_type = filename.split('.')[-1]
        if file_type.lower() in ['fits', 'fit']:
            with warnings.catch_warnings(record=True) as warning_list:
                try:
                    wl, spec, err, mask, _ = load_fits_spectrum(filename, ext=ext)
                    has_multiSpecWarning = any([w.category is MultipleSpectraWarning for w in warning_list])
                    if has_multiSpecWarning:
                        # Show warning that there are multiple spectra
                        print(term.red)
                        print("\n [WARNING] - Several data tables were detected. The first extension was used.")
                        print("")
                        fits.info(filename)
                        print(term.reset)
                        print("")

                except FormatError as error_msg:
                    print(error_msg)
                    print("")
                    fits.info(filename)
                    print("")
                    fits_data = fits.getdata(filename)
                    if isinstance(fits_data, fits.FITS_rec):
                        # Show the Table Column definitions:
                        print(fits_data.columns)
                        raise

        else:
            data = np.loadtxt(filename)
            if data.shape[1] == 3:
                wl, spec, err = data.T
                mask = np.ones_like(wl, dtype=bool)
            elif data.shape[1] == 4:
                wl, spec, err, mask = data.T
                mask = mask.astype(bool)
            elif data.shape[1] > 4:
                wl = data[:, 0]
                spec = data[:, 1]
                err = data[:, 2]
                mask = data[:, 3]
                mask = mask.astype(bool)
            else:
                print(" [ERROR] - Not enough columns to load wavelength, flux and error")
                print("           Check file format, must be three or more columns separated by blank space")
                print("")
                raise FormatError

        if airORvac == 'air':
            if verbose:
                print(" Converting wavelength from air to vacuum.")
            wl = air2vac(wl)

        if continuum is not None and len(continuum) == len(spec):
            spec = spec/continuum
            err = err/continuum
            normalized = True

        if mask_array is not None:
            if len(mask_array) == len(spec):
                mask = mask_array
            else:
                print("Wrong dimensions of the Input Mask: got %i pixels, spectrum has %i pixels" % (len(mask_array), len(spec)))

        if use_mask:
            self.add_data(wl, spec, res, err=err, normalized=normalized, mask=mask, nsub=nsub, filename=filename)
        else:
            self.add_data(wl, spec, res, err=err, normalized=normalized, nsub=nsub, filename=filename)

    def add_data(self, wl, flux, res, err=None, normalized=False, mask=None, nsub=1, filename=''):
        """
        Add spectral data to the DataSet. This will be used to define fitting regions.

        Parameters
        ----------
        wl : ndarray, shape (n)
            Input vacuum wavelength array in Ångstrøm.

        flux : ndarray, shape (n)
            Input flux array, should be same length as wl

        res : float or string
            Spectral resolution either given in km/s  (c/R),
            which is assumed to be constant over the whole spectrum,
            or as a string referring to a file containing the detailed
            line-spread function for the given spectrum.
            See details in the data section of the :ref:`documentation`.

        err : ndarray, shape (n)   [default = None]
            Error array, should be same length as wl
            If `None` is given, an uncertainty of 1 is given to all pixels.

        normalized : bool   [default = False]
            If the input spectrum is normalized this should be given as True
            in order to skip normalization steps.

        mask : array, shape (n)
            Boolean/int array defining the fitting regions.
            Only pixels with mask=True/1 will be included in the fit.

        nsub : integer
            Kernel subsampling factor relative to the data.
            This is only used if the resolution is given as a LSF file.

        filename : string
            The filename from which the data originated. Optional but highly recommended.
            Alternatively, use the method :meth:`DataSet.add_spectrum
            <VoigtFit.DataSet.add_spectrum>`.
        """

        mask_warning = """
        All pixels in the spectrum have been masked out.
        Pixels to *include* in the fit should have value = 1.
        Pixels to *excluded* should have value = 0.

        %sData have not been passed to the DataSet!%s
        """
        bold = '\033[1m'
        reset = '\033[0m'

        # assign specid:
        specid = "sid_%i" % len(self.data)

        if err is None:
            err = np.ones_like(flux)

        if mask is None:
            mask = np.ones_like(flux, dtype=bool)
        else:
            mask = mask.astype(bool)
            if np.sum(mask) == 0:
                print(mask_warning.strip() % (bold, reset))
                return

        # if isinstance(res, str):
        #     verify_lsf(res, wl)

        self.data.append({'wl': wl, 'flux': flux,
                          'error': err, 'res': res,
                          'norm': normalized, 'specID': specid,
                          'mask': mask, 'nsub': nsub})
        self.data_filenames.append(filename)

    def reset_region(self, reg):
        """Reset the data in a given :class:`regions.Region` to use the raw input data."""
        for chunk in self.data:
            if reg.res == chunk['res'] and (chunk['wl'].min() < reg.wl.min() < chunk['wl'].max()):
                raw_data = chunk

        cutout = (raw_data['wl'] >= reg.wl.min()) * (raw_data['wl'] <= reg.wl.max())
        reg.res = raw_data['res']
        reg.err = raw_data['error'][cutout]
        reg.flux = raw_data['flux'][cutout]
        reg.wl = raw_data['wl'][cutout]
        reg.normalized = raw_data['norm']

    def reset_all_regions(self, active_only=True):
        """
        Reset the data in all :class:`Regions <regions.Region>`
        defined in the DataSet to use the raw input data.
        """
        for reg in self.regions:
            if reg.has_active_lines() or not active_only:
                self.reset_region(reg)

    def get_resolution(self, line_tag, verbose=False):
        """Return the spectral resolution for the fitting :class:`Region <regions.Region>`
        where the line with the given `line_tag` is defined.

        Parameters
        ----------
        line_tag : str
            The line-tag for the line to look up: e.g., "FeII_2374"

        verbose : bool   [default = False]
            If `True`, print the returned spectral resolution to std out.

        Returns
        -------
        resolutions : list of float
            A list of the spectral resolution of the fitting regions
            where the given line is defined.
        """
        if line_tag:
            resolutions = list()
            regions_of_line = self.find_line(line_tag)
            for region in regions_of_line:
                if verbose and self.verbose:
                    if isinstance(region.res, str):
                        output_msg = "Spectral resolution in the region around %s is defined in file: %s"
                    else:
                        output_msg = " Spectral resolution in the region around %s is %.1f km/s."
                    print(output_msg % (line_tag, region.res))
                resolutions.append(region.res)
            return resolutions

    def set_resolution(self, res, line_tag=None, verbose=True, nsub=1):
        """
        Set the spectral resolution in km/s for the :class:`Region <regions.Region>`
        containing the line with the given `line_tag`. If multiple spectra are fitted
        simultaneously, this method will set the same resolution for *all* spectra.
        If `line_tag` is not given, the resolution will be set for *all* regions,
        including the raw data chunks defined in :attr:`VoigtFit.DataSet.data`!

        Note -- If not all data chunks have the same resolution, this method
        should be used with caution. It is advised to check the spectral resolution beforehand
        and only update the appropriate regions using a for-loop.
        """
        if line_tag:
            regions_of_line = self.find_line(line_tag)
            for region in regions_of_line:
                region.res = res
                if isinstance(res, str):
                    # verify_lsf(res, region.wl)
                    region.kernel = load_lsf(res, region.wl, nsub=nsub)
                    region.kernel_nsub = nsub
                else:
                    region.kernel = res
                    region.kernel_nsub = nsub

        else:
            if verbose:
                print(" [WARNING] - Setting spectral resolution for all regions, R=%.1f km/s!")
                if isinstance(res, str):
                    warn_msg = "             LSF-file: %s"
                else:
                    warn_msg = "             R = %.1f km/s!"
                print(warn_msg % res)

            for region in self.regions:
                if isinstance(res, str):
                    # verify_lsf(res, region.wl)
                    region.kernel = load_lsf(res, region.wl, nsub=nsub)
                    region.kernel_nsub = nsub
                region.res = res

            for chunk in self.data:
                chunk['res'] = res

    def set_systemic_redshift(self, z_sys):
        """Update the systemic redshift of the dataset"""
        self.redshift = z_sys

    def add_line(self, line_tag, velspan=None, active=True):
        """
        Add an absorption line to the DataSet.

        Parameters
        ----------
        line_tag : str
            The line tag for the transition which should be defined.
            Ex: "FeII_2374"

        velspan : tuple(float, float)   [default = None]
            The velocity span around the line center, which will be included
            in the fit. Can either be given as a single number for a symmetric region,
            or as a range or (lower, upper).
            If `None` is given, use the default `self.velspan` (default = ±400 km/s).

        active : bool   [default = True]
            Set the :class:`Line <VoigtFit.DataSet.Line>` as active
            (i.e., included in the fit).

        Notes
        -----
        This will initiate a :class:`Line <VoigtFit.DataSet.Line>` class
        with the atomic data for the transition, as well as creating a
        fitting :class:`Region <regions.Region>` containing the data cutout
        around the line center.
        """

        self.ready2fit = False
        if line_tag in self.all_lines:
            if self.verbose:
                print(" [WARNING] - %s is already defined." % line_tag)
            return False

        if line_tag in lineList['trans']:
            new_line = Line(line_tag)
        else:
            if self.verbose:
                print("\nThe transition (%s) not found in line list!"
                      % line_tag)
            return False

        velspan = self.check_velspan(velspan)
        vmin, vmax = velspan

        if new_line.element not in self.components.keys():
            # Initiate component list if ion has not been defined before:
            self.components[new_line.ion] = list()

        l_center = new_line.l0*(self.redshift + 1.)

        if self.data:
            success = False
            for chunk in self.data:
                if chunk['wl'].min() < l_center < chunk['wl'].max():
                    wl = chunk['wl']
                    vel = (wl-l_center)/l_center*299792.
                    span = ((vel >= vmin)*(vel <= vmax)).nonzero()[0]
                    new_wavelength = wl[span]

                    # Initiate new Region:
                    new_region = Region(velspan, chunk['specID'])
                    new_region.add_line(new_line)

                    merge = -1
                    if len(self.regions) > 0:
                        for num, region in enumerate(self.regions):
                            # Only allow regions arising from the same
                            # data chunk to be merged:
                            if chunk['specID'] == region.specID:
                                wl_overlap = np.intersect1d(new_wavelength,
                                                            region.wl)
                                if wl_overlap.any():
                                    merge = num

                        # If the region overlaps with another region
                        # merge the two regions:
                        if merge >= 0:
                            new_region.lines += self.regions[merge].lines

                            # merge the wavelength ranges
                            region_wl = np.union1d(new_wavelength,
                                                   self.regions[merge].wl)
                            old_mask = self.regions[merge].mask
                            old_wl = self.regions[merge].wl
                            tmp_mask = chunk['mask'][span]
                            tmp_wl = chunk['wl'][span]

                            # remove the overlapping region from the dataset
                            self.regions.pop(merge)

                        else:
                            region_wl = new_wavelength

                    else:
                        region_wl = new_wavelength

                    # Wavelength has now been defined and merged
                    # Cutout spectral chunks and add them to the new Region
                    cutout = (wl >= region_wl.min()) * (wl <= region_wl.max())
                    new_region.add_data_to_region(chunk, cutout)

                    # Update the mask of the new region to include
                    # new mask definitions from the old region:
                    # In the overlapping region, the mask from the existing
                    # region will overwrite any pixel mask in the data chunk
                    if merge >= 0:
                        old_mask_i = np.interp(new_region.wl, old_wl, old_mask,
                                               left=1, right=1)
                        old_mask_i = old_mask_i.astype(bool)
                        tmp_mask_i = np.interp(new_region.wl, tmp_wl, tmp_mask,
                                               left=1, right=1)
                        region_overlap = np.interp(new_region.wl,
                                                   old_wl, 1+0*old_mask,
                                                   right=0, left=0)
                        region_overlap = region_overlap.astype(bool)
                        tmp_mask_i[region_overlap] = 1
                        tmp_mask_i = tmp_mask_i.astype(bool)
                        new_mask = tmp_mask_i * old_mask_i
                        new_region.set_mask(new_mask)

                    self.regions.append(new_region)
                    if line_tag not in self.all_lines:
                        self.all_lines.append(line_tag)
                        self.lines[line_tag] = new_line
                    success = True

            if not success:
                if self.verbose:
                    err_msg = ("\n [ERROR] - The given line is not covered "
                               "by the spectral data: %s \n")
                    print(err_msg % line_tag)
                return False

        else:
            if self.verbose:
                print(" [ERROR] - No data is loaded. "
                      "Run method `add_data` to add spectral data.")

    def add_many_lines(self, tags, velspan=None):
        """
        Add many lines at once.

        Parameters
        ----------
        tags : list(str)
            A list of line tags for the transitions that should be added.

        velspan : float   [default = None]
            The velocity span around the line center, which will be included
            in the fit. If `None` is given, use the default
            :attr:`velspan <VoigtFit.DataSet.velspan>` (±400 km/s).
        """

        self.ready2fit = False
        velspan = self.check_velspan(velspan)

        for tag in tags:
            self.add_line(tag, velspan)

    def add_lines(self, line_tags, velspan=None):
        """Alias for `self.add_many_lines`."""
        self.add_many_lines(line_tags, velspan=velspan)

    def remove_line(self, line_tag):
        """
        Remove an absorption line from the DataSet. If this is the last line
        in a fitting region the given region will be eliminated, and if this
        is the last line of a given ion, then the components will be eliminated
        for that ion.

        Parameters
        ----------
        line_tag : str
            Line tag of the transition that should be removed.
        """
        if (line_tag in self.all_lines) and (line_tag in list(self.lines.keys())):
            self.all_lines.remove(line_tag)
            self.lines.pop(line_tag)
        else:
            in_all_lines = "" if line_tag in self.all_lines else "not "
            in_lines = "" if line_tag in list(self.lines.keys()) else "not "
            print("")
            print(" [ERROR] - Problem detected in database.")
            print(" The line %s is %sdefined in `self.all_lines`." %
                  (line_tag, in_all_lines))
            print(" The line %s is %sdefined in `self.lines`." %
                  (line_tag, in_lines))
            print("")

        # --- Check if the ion has transitions defined in other regions
        ion = line_tag.split('_')[0]
        ion_defined_elsewhere = False
        all_ions = list()
        for this_line_tag in self.all_lines:
            this_ion = this_line_tag.split('_')[0]
            if this_ion not in all_ions:
                all_ions.append(this_ion)

            if this_line_tag.find(ion) >= 0:
                ion_defined_elsewhere = True

        # --- If it is not defined elsewhere, remove it from components
        if not ion_defined_elsewhere and ion not in all_ions:
            # Check if components have been defined for the ion:
            if ion in self.components.keys():
                self.components.pop(ion)

        remove_this = -1
        for num, region in enumerate(self.regions):
            if region.has_line(line_tag):
                remove_this = num

        if remove_this >= 0:
            if len(self.regions[remove_this].lines) == 1:
                self.regions.pop(remove_this)
            else:
                self.regions[remove_this].remove_line(line_tag)

        else:
            if self.verbose:
                print("")
                print(" The line, %s, is not defined. Nothing to remove." %
                      line_tag)

    def remove_all_lines(self):
        lines_to_remove = self.all_lines.copy()
        for line_tag in lines_to_remove:
            self.remove_line(line_tag)

    def normalize_line(self, line_tag, norm_method='spline', velocity=True):
        """
        Normalize or re-normalize a given line

        Parameters
        ----------
        line_tag : str
            Line tag of the line whose fitting region should be normalized.

        norm_method : str   [default = 'spline']
            Normalization method used for the interactive continuum fit.
            Should be on of: ["spline", "linear"]

        velocity : bool   [default = False]
            If a `True`, the regions are displayed in velocity space
            relative to the systemic redshift instead of in wavelength space
            when masking and defining continuum normalization interactively.
        """

        if velocity:
            z_sys = self.redshift
        else:
            z_sys = None

        regions_of_line = self.find_line(line_tag)
        for region in regions_of_line:
            if not region.normalized:
                go_on = 0
                while go_on == 0:
                    go_on = region.normalize(norm_method=norm_method,
                                             z_sys=z_sys)
                    # region.normalize returns 1 when continuum is fitted
            # region.normalize(norm_method=norm_method, z_sys=z_sys)

    def mask_line(self, line_tag, reset=True, mask=None, telluric=True, velocity=False):
        """
        Define exclusion masks for the fitting region of a given line.
        Note that the masked regions are exclusion regions and will not be used for the fit.
        If components have been defined, these will be shown as vertical lines.

        Parameters
        ----------
        line_tag : str
            Line tag for the :class:`Line <VoigtFit.DataSet.Line>` whose
            :class:`Region <regions.Region>` should be masked.

        reset : bool   [default = True]
            If `True`, clear the mask before defining a new mask.

        mask : array_like, shape (n)   [default = None]
            If the mask is given, it must be a boolean array of the same length
            as the region flux, err, and wl arrays.
            Passing a mask this was supresses the interactive masking process.

        telluric : bool   [default = True]
            If `True`, a telluric absorption template and sky emission template
            is shown for reference.

        velocity : bool   [default = False]
            If a `True`, the regions are displayed in velocity space
            relative to the systemic redshift instead of in wavelength space
            when masking and defining continuum normalization interactively.
        """
        if velocity:
            z_sys = self.redshift
        else:
            z_sys = None

        regions_of_line = self.find_line(line_tag)
        for region in regions_of_line:
            if reset:
                region.clear_mask()
                region.new_mask = True

            if hasattr(mask, '__iter__'):
                region.mask = mask
                region.new_mask = False
            else:
                if region.new_mask:
                    region.define_mask(z=self.redshift, dataset=self,
                                       telluric=telluric, z_sys=z_sys)

    def mask_range(self, line_tag, x1, x2, idx=None):
        """Define mask in a range from `x1` to `x2` in velocity space."""
        regions_of_line = self.find_line(line_tag)
        z = self.redshift
        l0 = self.lines[line_tag].l0
        if idx is None:
            # Loop over all regions of line:
            for reg in regions_of_line:
                vel = (reg.wl/(l0*(z+1)) - 1.)*299792.458
                mask = (vel > x1)*(vel < x2)
                new_mask = reg.mask.copy()
                new_mask = new_mask * ~mask
                reg.set_mask(new_mask)

        elif hasattr(idx, '__iter__'):
            # loop over regions in idx
            for num in idx:
                reg = regions_of_line[num]
                vel = (reg.wl/(l0*(z+1)) - 1.)*299792.458
                mask = (vel > x1)*(vel < x2)
                new_mask = reg.mask.copy()
                new_mask = new_mask * ~mask
                reg.set_mask(new_mask)

        else:
            reg = regions_of_line[idx]
            vel = (reg.wl/(l0*(z+1)) - 1.)*299792.458
            mask = (vel > x1)*(vel < x2)
            new_mask = reg.mask.copy()
            new_mask = new_mask * ~mask
            reg.set_mask(new_mask)

    def clear_mask(self, line_tag, idx=None):
        """
        Clear the mask for the :class:`Region <regions.Region>`
        containing the given `line_tag`.
        If more regions are defined for the same line (when fitting multiple spectra),
        the given region can be specified by passing an index `idx`.
        """
        regions_of_line = self.find_line(line_tag)
        if idx is None:
            for reg in regions_of_line:
                reg.clear_mask()
        else:
            reg = regions_of_line[idx]
            reg.clear_mask()

    def find_line(self, line_tag):
        """
        Look up the fitting :class:`Region <regions.Region>` for a given *line tag*.

        Parameters
        ----------
        line_tag : str
            The line tag of the line whose region will be returned.

        Returns
        -------
        regions_of_line : list of :class:`Region <regions.Region>`
            A list of the fitting regions containing the given line.
            This can be more than one region in case of overlapping or multiple spectra.
        """
        regions_of_line = list()
        if line_tag in self.all_lines:
            for region in self.regions:
                if region.has_line(line_tag):
                    regions_of_line.append(region)

            return regions_of_line

        else:
            if self.verbose:
                print("\n The line (%s) is not defined." % line_tag)

        return None

    def lines_of_ion(self, ion):
        """Return a list of all line tags for a given ion."""
        return [ll.tag for ll in self.lines.values() if ll.ion == ion]

    def has_line(self, line_tag, active_only=False):
        """Return True if the given line is defined."""
        if active_only:
            return line_tag in self.all_active_lines()
        else:
            return line_tag in line_tag in self.all_lines

    def has_ion(self, ion, active_only=False):
        """Return True if the dataset has lines defined for the given ion."""
        if active_only:
            all_ions = list(set([ll.ion for ll in self.lines.values() if ll.active]))
        else:
            all_ions = list(set([ll.ion for ll in self.lines.values()]))
        return ion in all_ions

    def activate_line(self, line_tag):
        """Activate a given line defined by its `line_tag`"""
        if line_tag in self.lines.keys():
            line = self.lines[line_tag]
            line.set_active()
            self.ready2fit = False

        else:
            regions_of_line = self.find_line(line_tag)
            for region in regions_of_line:
                for line in region.lines:
                    if line.tag == line_tag:
                        line.set_active()

    def deactivate_line(self, line_tag):
        """
        Deactivate a given line defined by its `line_tag`.
        This will exclude the line during the fit but will not remove the data.
        """
        if line_tag in self.lines.keys():
            line = self.lines[line_tag]
            line.set_inactive()

        else:
            regions_of_line = self.find_line(line_tag)
            for region in regions_of_line:
                for line in region.lines:
                    if line.tag == line_tag:
                        line.set_inactive()

        # --- Check if the ion has transitions defined in other regions
        ion = line_tag.split('_')[0]
        ion_defined_elsewhere = False
        for this_line_tag in self.all_lines:
            if this_line_tag.find(ion) >= 0:
                ion_defined_elsewhere = True

        # --- If it is not defined elsewhere, remove it from components
        if not ion_defined_elsewhere:
            self.components.pop(ion)
            self.ready2fit = False

    def deactivate_all(self):
        """Deactivate all lines defined in the DataSet. This will not remove the lines."""
        for line_tag in self.all_lines:
            self.deactivate_line(line_tag)
        self.components = dict()

    def activate_all(self):
        """Activate all lines defined in the DataSet."""
        for line_tag in self.all_lines:
            self.activate_line(line_tag)

    def all_active_lines(self):
        """Returns a list of all the active lines defined by their `line_tag`."""
        act_lines = list()
        for line_tag, line in self.lines.items():
            if line.active:
                act_lines.append(line_tag)
        return act_lines

    def get_lines_for_ion(self, ion):
        """
        Return a list of :class:`Line <dataset.Line>` objects
        corresponding to the given *ion*.

        Parameters
        ----------
        ion : str
            The ion for the lines to get.

        Returns
        -------
        lines_for_ion : list(:class:`Line <dataset.Line>`)
            List of Lines defined for the given *ion*.

        """
        lines_for_ion = list()
        for line in self.lines.values():
            if line.ion == ion:
                lines_for_ion.append(line)

        return lines_for_ion

    def reset_static_variables(self):
        self.static_variables = Parameters()

    def reset_components(self, ion=None):
        """
        Reset component structure for a given ion.

        Parameters
        ----------
        ion : str   [default = None]
            The ion for which to reset the components: e.g., 'FeII', 'HI', 'CIa', etc.
            If `None` is given, *all* components for *all* ions will be reset.
        """

        if ion:
            if ion in self.components.keys():
                self.components.pop(ion)
            else:
                if self.verbose:
                    print(" [ERROR] - No components defined for ion: %s" % ion)

        else:
            self.components = dict()

    def add_component(self, ion, z, b, logN,
                      var_z=True, var_b=True, var_N=True, tie_z=None, tie_b=None, tie_N=None):
        """
        Add component for a given ion. Each component defined will be used for all transitions
        defined for a given ion.

        Parameters
        ----------
        ion : str
            The ion for which to define a component: e.g., "FeII", "HI", "CIa", etc.

        z : float
            The redshift of the component.

        b : float
            The effective broadening parameter for the component in km/s.
            This parameter is constrained to be in the interval [0 - 1000] km/s.

        logN : float
            The 10-base logarithm of the column density of the component.
            The column density is expected in cm^-2.

        var_z : bool   [default = True]
            If `False`, the redshift of the component will be kept fixed.

        var_b : bool   [default = True]
            If `False`, the b-parameter of the component will be kept fixed.

        var_N : bool   [default = True]
            If `False`, the column density of the component will be kept fixed.

        tie_z, tie_b, tie_N : str   [default = None]
            Parameter constraints for the different variables.

        Notes
        -----
        The ties are defined relative to the parameter names. The naming is as follows:
        The redshift of the first component of FeII is called "z0_FeII",
        the logN of the second component of SiII is called "logN1_SiII".
        For more information about parameter ties, see the documentation for lmfit_.

        """
        this_comp = Component(z, b, logN, var_z, var_b, var_N, tie_z, tie_b, tie_N)
        if ion in self.components.keys():
            self.components[ion].append(this_comp)
        else:
            self.components[ion] = [this_comp]

    def add_component_velocity(self, ion, v, b, logN,
                               var_z=True, var_b=True, var_N=True, tie_z=None, tie_b=None, tie_N=None):
        """
        Same as for :meth:`add_component <VoigtFit.DataSet.add_component>`
        but input is given as relative velocity instead of redshift.
        """
        z = self.redshift + v/299792.458*(self.redshift + 1.)
        this_comp = Component(z, b, logN, var_z, var_b, var_N, tie_z, tie_b, tie_N)
        if ion in self.components.keys():
            self.components[ion].append(this_comp)
        else:
            self.components[ion] = [this_comp]

    def interactive_components(self, line_tag, velocity=False):
        """
        Define components interactively for a given ion. The components will be defined on the
        basis of the given line for that ion. If the line is defined in several spectra
        then the interactive window will show up for each.
        Running the interactive mode more times for different transitions of the same ion
        will append the components to the structure.
        If no components should be added, then simply click `enter` to terminate the process
        for the given transition.

        Parameters
        ----------
        line_tag : str
            Line tag for the line belonging to the ion for which components should be defined.

        velocity : bool   [default = False]
            If a `True`, the region is displayed in velocity space
            relative to the systemic redshift instead of in wavelength space.

        Notes
        -----
        This will launch an interactive plot showing the fitting region of the given line.
        The user can then click on the positions of the components which. At the end, the
        redshifts and estimated column densities are printed to terminal. The b-parameter
        is assumed to be unresolved, i.e., taken from the resolution.
        """
        regions_of_line = self.find_line(line_tag)
        if len(regions_of_line) > 1:
            print("\n Note -- The given line (%s) is defined in more than one spectral region")
            print("        All regions will be shown one by one for interactive definition...\n")
        for region in regions_of_line:
            wl, flux, err, mask = region.unpack()
            plt.close('all')
            fig = plt.figure()
            ax = fig.add_subplot(111)
            mask_idx = np.where(mask == 0)[0]
            big_mask_idx = np.union1d(mask_idx + 1, mask_idx - 1)
            big_mask = np.ones_like(mask, dtype=bool)
            big_mask[big_mask_idx] = False
            masked_range = np.ma.masked_where(big_mask, flux)
            flux = np.ma.masked_where(~mask, flux)

            line = self.lines[line_tag]

            if velocity:
                l_ref = line.l0 * (self.redshift + 1.)
                x = (wl - l_ref)/l_ref * 299792.458
                x_label = u"Rel. Velocity  [${\\rm km\\ s^{-1}}$]"
            else:
                x = wl
                x_label = u"Wavelength  [Å]"

            ax.plot(x, masked_range, color='0.7', drawstyle='steps-mid', lw=0.9)
            ax.plot(x, flux, 'k', drawstyle='steps-mid')

            if line.element in self.components.keys():
                l0, f, gam = line.get_properties()
                ion = line.ion
                for comp in self.components[ion]:
                    z = comp.z
                    if velocity:
                        ax.axvline((l0*(z+1) - l_ref)/l_ref * 299792.458, ls=':', color='r', lw=0.4)
                    else:
                        ax.axvline(l0*(z+1), ls=':', color='r', lw=0.4)

            if region.normalized:
                c_level = 1.
            else:
                ax.set_title("Click to Mark Average Continuum Level...\n(1 click)")
                cont_level_point = plt.ginput(1, 30.)
                c_level = cont_level_point[0][1]
                ax.axhline(c_level, color='0.3', ls=':')

            ax.set_title("Mark peak absorption for components of %s\nFinish with 'enter'" % line.tag)
            ax.set_xlabel(x_label)
            if region.normalized:
                ax.set_ylabel(u"Normalized Flux")
            else:
                ax.set_ylabel(u"Flux")
            plt.draw()
            plt.tight_layout()

            # Assume that components are unresolved:
            b = region.kernel_fwhm / 2.35482
            comp_lines = list()
            comp_list = list()
            success = False
            while success is False:
                comps = plt.ginput(-1, 60)
                for x0, y0 in comps:
                    if velocity:
                        z0 = self.redshift + x0/299792.458 * (self.redshift + 1.)
                    else:
                        z0 = x0/line.l0 - 1.
                    # Calculate logN from peak depth:
                    y0 = max(y0/c_level, 0.01)
                    logN = np.log10(-b * np.log(y0) / (1.4983e-15 * line.l0 * line.f))

                    vline = ax.axvline(x0, color='darkblue', alpha=0.8)
                    comp_lines.append(vline)
                    comp_list.append([z0, b, logN])
                plt.draw()

                print("Are the shown components correct?")
                ax.set_title("Are the shown components correct?")
                plt.draw()
                answer = input("[YES/no]  ")
                if answer.lower() in ['', 'yes', 'y']:
                    success = True
                else:
                    comp_list = list()
                    for vline in comp_lines:
                        vline.remove()
                    comp_lines = list()
                    ax.set_title("Mark peak absorption for components of %s\nFinish with 'enter'" % line.tag)
                    plt.draw()

            if len(comp_list) > 0:
                print("Defining following components:")
                print("(the lines below can be copied directly to the input file)\n")
                for z, b, logN in comp_list:
                    print("component %s  z=%.6f  b=%.1f  logN=%.2f" % (line.ion, z, b, logN))
                    self.add_component(line.ion, z, b, logN)
            else:
                print("No components were defined. Are you sure you want to continue?")


    def delete_component(self, ion, index):
        """Remove component of the given `ion` with the given `index`."""
        if ion in self.components.keys():
            self.components[ion].pop(index)

        else:
            if self.verbose:
                print(" [ERROR] - No components defined for ion: "+ion)

    def copy_components(self, to_ion='', from_ion='', logN=0, ref_comp=None, tie_z=True, tie_b=True):
        """
        Copy velocity structure to `ion` from another `ion`.

        Parameters
        ----------
        to_ion : str
            The new ion to define.

        from_ion : str
            The base ion which will be used as reference.

        logN : float
            If logN is given, the starting guess is defined from this value
            following the pattern of the components defined for `anchor` relative to the
            `ref_comp` (default: the first component).

        ref_comp : int
            The reference component to which logN will be scaled.

        tie_z : bool   [default = True]
            If `True`, the redshifts for all components of the two ions will be tied together.

        tie_b : bool   [default = True]
            If `True`, the b-parameters for all components of the two ions will be tied together.
        """
        if to_ion == '' and from_ion == '':
            err_msg = " [ERROR] - Must specify both 'to_ion' and 'from_ion'!"
            raise ValueError(err_msg)
        elif from_ion not in self.components.keys():
            err_msg = " [ERROR] - The base ion ('from_ion') is not defined in the dataset!"
            raise KeyError(err_msg)

        reference = self.components[from_ion]
        # overwrite the components already defined for ion if they exist
        self.components[to_ion] = list()

        if ref_comp is not None:
            offset_N = logN - reference[ref_comp].logN
        else:
            # Strip ionization state to get element:
            element = to_ion[:2] if to_ion[1].islower() else to_ion[0]
            element_anchor = from_ion[:2] if from_ion[1].islower() else from_ion[0]
            solar_elements = Asplund.solar.keys()
            if element in solar_elements and element_anchor in solar_elements:
                # Use Solar abundance ratios:
                offset_N = Asplund.solar[element][0] - Asplund.solar[element_anchor][0]
            else:
                offset_N = 0.

        for num, comp in enumerate(reference):
            new_comp = copy.deepcopy(comp)
            has_active_ion = self.has_ion(from_ion, active_only=True)
            if logN:
                new_comp.logN += offset_N
            if tie_z and has_active_ion:
                new_comp.tie_z = 'z%i_%s' % (num, from_ion)
            if tie_b and has_active_ion:
                new_comp.tie_b = 'b%i_%s' % (num, from_ion)

            self.components[to_ion].append(new_comp)

    def load_components_from_file(self, fname, fit_pars=True):
        """Load best-fit parameters from an output file `fname`.
        If `fit_pars` is True, then update the best_fit parameters."""
        components_to_add = list()
        all_ions_in_file = list()
        with open(fname) as parameters:
            for line in parameters.readlines():
                line = line.strip()
                pars = line.split()
                if len(line) == 0:
                    pass
                elif line[0] == '#':
                    pass
                elif len(pars) == 8:
                    num = int(pars[0])
                    ion = pars[1]
                    z = float(pars[2])
                    z_err = float(pars[3])
                    b = float(pars[4])
                    b_err = float(pars[5])
                    logN = float(pars[6])
                    logN_err = float(pars[7])
                    components_to_add.append([num, ion, z, b, logN,
                                              z_err, b_err, logN_err])
                    if ion not in all_ions_in_file:
                        all_ions_in_file.append(ion)

        for ion in all_ions_in_file:
            if ion in self.components.keys():
                self.reset_components(ion)
                # Remove all parameters from self.best_fit
                if isinstance(self.best_fit, dict) and fit_pars:
                    pars_to_delete = list()
                    for parname in self.best_fit.keys():
                        if ion in parname:
                            pars_to_delete.append(parname)
                    for parname in pars_to_delete:
                        self.best_fit.pop(parname)

        for comp_pars in components_to_add:
            (num, ion, z, b, logN, z_err, b_err, logN_err) = comp_pars
            self.add_component(ion, z, b, logN)
            if fit_pars and self.best_fit:
                parlist = [['z', z, z_err],
                           ['b', b, b_err],
                           ['logN', logN, logN_err]]

                for base, val, err in parlist:
                    parname = '%s%i_%s' % (base, num, ion)
                    self.best_fit.add(parname, value=val)
                    self.best_fit[parname].stderr = err

        if self.verbose:
            print("\n  Added components for the given ions:")
            print("  " + ", ".join(all_ions_in_file))


    def fix_structure(self, ion=None):
        """Fix the velocity structure, that is, the redshifts and the b-parameters.

        Parameters
        ----------
        ion : str   [default = None]
            The ion for which the structure should be fixed.
            If `None` is given, the structure is fixed for all ions.
        """
        if ion:
            for comp in self.components[ion]:
                comp.var_b = False
                comp.var_z = False
        else:
            for ion in self.components.keys():
                for comp in self.components[ion]:
                    comp.var_b = False
                    comp.var_z = False

    def free_structure(self, ion=None):
        """Free the velocity structure, that is, the redshifts and the b-parameters.

        Parameters
        ----------
        ion : str   [default = None]
            The ion for which the structure should be released.
            If `None` is given, the structure is released for all ions.
        """
        if ion:
            for comp in self.components[ion]:
                comp.var_b = True
                comp.var_z = True
        else:
            for ion in self.components.keys():
                for comp in self.components[ion]:
                    comp.var_b = True
                    comp.var_z = True

    # Fine-structure Lines:
    def add_fine_lines(self, line_tag, levels=None, full_label=False, velspan=None):
        """
        Add fine-structure line complexes by providing only the main transition.
        This function is mainly useful for the CI complexes, where the many
        lines are closely located and often blended.

        Parameters
        ----------
        line_tag : str
            Line tag for the ground state transition, e.g., "CI_1656"

        levels : str, list(str)    [default = None]
            The levels of the fine-structure complexes to add, starting with "a"
            referring to the first excited level, "b" is the second, etc..
            Several levels can be given at once: ['a', 'b'].
            Note that the ground state transition is always included.
            If `levels` is not given, all levels are included.

        full_label : bool   [default = False]
            If `True`, the label will be translated to the full quantum
            mechanical description of the state.
        """
        self.fine_lines[line_tag] = list()
        velspan = self.check_velspan(velspan)

        if hasattr(levels, '__iter__'):
            for fineline in fine_structure_complexes[line_tag]:
                ion = fineline.split('_')[0]
                if ion[-1] in levels or ion[-1].isupper():
                    self.add_line(fineline, velspan)
                    self.fine_lines[line_tag].append(fineline)

        elif levels is None:
            for fineline in fine_structure_complexes[line_tag]:
                self.add_line(fineline, velspan)
                self.fine_lines[line_tag].append(fineline)

        else:
            for fineline in fine_structure_complexes[line_tag]:
                ion = fineline.split('_')[0]
                if ion[-1] in levels or ion[-1].isupper():
                    self.add_line(fineline, velspan)
                    self.fine_lines[line_tag].append(fineline)

        # Set label:
        regions_of_line = self.find_line(line_tag)
        for reg in regions_of_line:
            if full_label:
                reg.label = line_complexes.full_labels[line_tag]
            else:
                raw_label = line_tag.replace('_', r'\ \lambda')
                reg.set_label(r"${\rm %s}$" % raw_label)

    def remove_fine_lines(self, line_tag, levels=None):
        """
        Remove lines associated to a given fine-structure complex.

        Parameters
        ----------
        line_tag : str
            The line tag of the ground state transition to remove.

        levels : str, list(str)   [default = None]
            The levels of the fine-structure complexes to remove, with "a" referring
            to the first excited level, "b" is the second, etc..
            Several levels can be given at once as a list: ['a', 'b']
            or as a concatenated string: 'abc'.
            By default, all levels are included.
        """
        for fineline in fine_structure_complexes[line_tag]:
            if fineline in self.all_lines:
                line = self.lines[fineline]
                if levels is None:
                    pass
                elif line.ion[-1] in levels:
                    self.fine_lines[line_tag].remove(fineline)
                else:
                    continue
                self.remove_line(fineline)
                if self.verbose:
                    print(" Removing line: %s" % fineline)
        if levels is None:
            self.fine_lines.pop(line_tag)

    def deactivate_fine_lines(self, line_tag, levels=None, verbose=True):
        """
        Deactivate all lines associated to a given fine-structure complex.

        Parameters
        ----------
        line_tag : str
            The line tag of the ground state transition to deactivate.

        levels : str, list(str)   [default = None]
            The levels of the fine-structure complexes to deactivate,
            with the string "a" referring to the first excited level,
            "b" is the second, etc...
            Several levels can be given at once as a list: ['a', 'b']
            or as a concatenated string: 'abc'.
            By default, all levels are included.
        """
        for fineline in fine_structure_complexes[line_tag]:
            if fineline in self.all_lines:
                line = self.lines[fineline]
                if levels is None:
                    pass
                elif line.ion[-1] in levels:
                    pass
                elif line.ion[-1].isupper():
                    pass
                else:
                    continue
                self.deactivate_line(fineline)
                if self.verbose and verbose:
                    print("Deactivated line: %s" % fineline)

    def activate_fine_lines(self, line_tag, levels=None):
        """
        Activate all lines associated to a given fine-structure complex.

        Parameters
        ----------
        line_tag : str
            The line tag of the ground state transition to activate.

        levels : str, list(str)   [default = None]
            The levels of the fine-structure complexes to activate,
            with the string "a" referring to the first excited level,
            "b" is the second, etc...
            Several levels can be given at once as a list: ['a', 'b']
            or as a concatenated string: 'abc'.
            By default, all levels are included.
        """
        for fineline in fine_structure_complexes[line_tag]:
            if fineline in self.all_lines:
                line = self.lines[fineline]
                if levels is None:
                    pass
                elif line.ion[-1] in levels:
                    pass
                elif line.ion[-1].isupper():
                    pass
                else:
                    continue

                self.activate_line(fineline)
                if self.verbose:
                    print("Activated line: %s" % fineline)
    # =========================================================================

    # Molecules:
    def add_molecule(self, molecule, band, Jmax=0, velspan=None, full_label=False):
        """
        Add molecular lines for a given band, e.g., "AX(0-0)" of CO.

        Parameters
        ----------
        molecule : str
            The molecular identifier, e.g., 'CO', 'H2'

        band : str
            The vibrational band of the molecule, e.g., for CO: "AX(0-0)"
            These bands are defined in :mod:`molecules`.

        Jmax : int   [default = 0]
            The maximal rotational level to include. All levels up to and including `J`
            will be included.

        velspan : float   [default = None]
            The velocity span around the line center, which will be included in the fit.
            If `None` is given, use the default :attr:`velspan <VoigtFit.DataSet.velspan>`
            defined (500 km/s).

        full_label : bool   [default = False]
            If `True`, the label will be translated to the full quantum
            mechanical description of the state.
        """
        if molecule == 'CO':
            full_labels = molecules.CO_full_labels
            nu_level = molecules.CO[band]
            ref_J0 = molecules.CO[band][0][0]
        elif molecule == 'H2':
            full_labels = molecules.H2_full_labels
            nu_level = molecules.H2[band]
            ref_J0 = molecules.H2[band][0][0]

        for transitions in nu_level[:Jmax+1]:
            self.add_many_lines(transitions, velspan=velspan)

        regions_of_line = self.find_line(ref_J0)
        for region in regions_of_line:
            if full_label:
                region.label = full_labels[band]
            else:
                region.label = "%s %s" % (molecule, band)

        if molecule not in self.molecules.keys():
            self.molecules[molecule] = list()
        self.molecules[molecule].append([band, Jmax])

    def remove_molecule(self, molecule, band):
        """Remove all lines for the given band of the given molecule."""
        bands_for_molecule = [item[0] for item in self.molecules[molecule]]
        if band not in bands_for_molecule:
            if self.verbose:
                warning_msg = " [ERROR] - The %s band for %s is not defined!"
                print("")
                print(warning_msg % (band, molecule))
            return None

        if molecule == 'CO':
            nu_level = molecules.CO[band]
        elif molecule == 'H2':
            nu_level = molecules.H2[band]

        for transitions in nu_level:
            for line_tag in transitions:
                if line_tag in self.all_lines:
                    self.remove_line(line_tag)

        remove_idx = -1
        for num, this_item in enumerate(self.molecules[molecule]):
            if this_item[0] == band:
                remove_idx = num
        if remove_idx >= 0:
            self.molecules[molecule].pop(remove_idx)
        else:
            print(" [ERROR] - %s was not found in self.molecules['%s']" % (band, molecule))

        if len(self.molecules[molecule]) == 0:
            self.molecules.pop(molecule)

    def deactivate_molecule(self, molecule, band):
        """
        Deactivate all lines for the given band of the given molecule.
        To see the available molecular bands defined, see the manual pdf
        or :mod:`VoigtFit.molecules`.
        """
        bands_for_molecule = [item[0] for item in self.molecules[molecule]]
        if band not in bands_for_molecule:
            if self.verbose:
                warning_msg = " [ERROR] - The %s band for %s is not defined!"
                print("")
                print(warning_msg % (band, molecule))
            return None

        if molecule == 'CO':
            nu_level = molecules.CO[band]
        elif molecule == 'H2':
            nu_level = molecules.H2[band]

        for transitions in nu_level:
            for line_tag in transitions:
                if line_tag in self.all_lines:
                    self.deactivate_line(line_tag)

    def activate_molecule(self, molecule, band):
        """
        Activate all lines for the given band of the given molecule.

            - Ex: ``activate_molecule('CO', 'AX(0-0)')``
        """
        bands_for_molecule = [item[0] for item in self.molecules[molecule]]
        if band not in bands_for_molecule:
            if self.verbose:
                warning_msg = " [ERROR] - The %s band for %s is not defined!"
                print("")
                print(warning_msg % (band, molecule))
            return None

        if molecule == 'CO':
            nu_level = molecules.CO[band]
        elif molecule == 'H2':
            nu_level = molecules.H2[band]

        for transitions in nu_level:
            for line_tag in transitions:
                if line_tag in self.all_lines:
                    self.activate_line(line_tag)

    # =========================================================================

    def add_variable(self, name, **kwargs):
        self.static_variables.add(name, **kwargs)

    def prepare_dataset(self, norm=True, mask=False, verbose=True,
                        active_only=False,
                        force_clean=True,
                        velocity=False,
                        check_lines=True,
                        f_lower=0., f_upper=100.,
                        l_lower=0., l_upper=1.e4):
        """
        Prepare the data for fitting. This function sets up the parameter structure,
        and handles the normalization and masking of fitting regions.

        Parameters
        ----------
        norm : bool   [default = True]
            Opens an interactive window to let the user normalize each region
            using the defined :attr:`norm_method <VoigtFit.DataSet.norm_method>`.

        mask : bool   [default = True]
            Opens an interactive window to let the user mask each fitting region.

        verbose : bool   [default = True]
            If this is set, the code will print small info statements during the run.

        active_only : bool   [default = False]
            If True, only define masks for active lines.

        force_clean : bool   [default = False]
            If this is True, components for inactive elements will be removed.

        velocity : bool   [default = False]
            If `True`, the regions are displayed in velocity space
            relative to the systemic redshift instead of in wavelength space
            when masking and defining continuum normalization interactively.

        check_lines : bool   [default = True]
            If `True`, all available lines covered by the data will be checked.
            The user will be propmted if lines are available for ions that have
            already been defined.

        f_lower : float   [default = 0.]
            Lower limit on oscillator strengths for transitions when verifying
            all transitions for defined ions.

        f_upper : float   [default = 100.]
            Upper limit on oscillator strengths for transitions when verifying
            all transitions for defined ions.

        l_lower : float   [default = 0.]
            Lower limit on rest-frame wavelength for transitions when verifying
            all transitions for defined ions.

        l_upper : float   [default = 1.e4]
            Upper limit on rest-frame wavelength for transitions when verifying
            all transitions for defined ions.

        Returns
        -------
        bool
            The function returns `True` when the dataset has passed all the steps.
            If one step fails, the function returns `False`.
            The :attr:`ready2fit <VoigtFit.DataSet.ready2fit>` attribute of the dataset is also
            updated accordingly.

        """

        if velocity:
            z_sys = self.redshift
        else:
            z_sys = None

        plt.close('all')
        # --- Normalize fitting regions manually, or use polynomial fitting
        if norm:
            for region in self.regions:
                if not region.normalized:
                    go_on = 0
                    while go_on == 0:
                        go_on = region.normalize(norm_method=self.norm_method,
                                                 z_sys=z_sys)
                        # region.normalize returns 1 when continuum is fitted

            if verbose and self.verbose:
                print("")
                print(" [DONE] - Continuum fitting successfully finished.")
                print("")

        # --- Check that no components for inactive elements are defined:
        for this_ion in list(self.components.keys()):
            lines_for_this_ion = [this_line.active for this_line in self.lines.values() if this_line.ion == this_ion]

            if np.any(lines_for_this_ion):
                pass
            else:
                if verbose:
                    warn_msg = "\n [WARNING] - Components defined for inactive or missing element: %s"
                    print(warn_msg % this_ion)

                if force_clean:
                    # Remove components for inactive elements
                    self.components.pop(this_ion)
                    if verbose:
                        print("             The components have been removed.")
                print("")

        # --- Prepare fit parameters  [class: lmfit.Parameters]
        self.pars = Parameters()
        self.pars += self.static_variables
        # - First setup parameters with values only:
        for ion in self.components.keys():
            for n, comp in enumerate(self.components[ion]):
                ion = ion.replace('*', 'x')
                z, b, logN = comp.get_pars()
                z_name = 'z%i_%s' % (n, ion)
                b_name = 'b%i_%s' % (n, ion)
                N_name = 'logN%i_%s' % (n, ion)

                self.pars.add(z_name, value=myfloat(z), vary=comp.var_z)
                self.pars.add(b_name, value=myfloat(b), vary=comp.var_b, min=0.)
                self.pars.add(N_name, value=myfloat(logN), vary=comp.var_N)

        # - Then setup parameter links:
        for ion in self.components.keys():
            for n, comp in enumerate(self.components[ion]):
                ion = ion.replace('*', 'x')
                z_name = 'z%i_%s' % (n, ion)
                b_name = 'b%i_%s' % (n, ion)
                N_name = 'logN%i_%s' % (n, ion)

                if comp.tie_z:
                    self.pars[z_name].expr = comp.tie_z
                if comp.tie_b:
                    self.pars[b_name].expr = comp.tie_b
                if comp.tie_N:
                    self.pars[N_name].expr = comp.tie_N

        # Setup Chebyshev parameters:
        if self.cheb_order >= 0:
            for reg_num, reg in enumerate(self.regions):
                if not reg.has_active_lines():
                    continue
                p0 = np.median(reg.flux)
                var_par = reg.has_active_lines()
                if np.sum(reg.mask) == 0:
                    var_par = False
                for cheb_num in range(self.cheb_order+1):
                    if cheb_num == 0:
                        self.pars.add('R%i_cheb_p%i' % (reg_num, cheb_num), value=p0, vary=var_par)
                    else:
                        self.pars.add('R%i_cheb_p%i' % (reg_num, cheb_num), value=0.0, vary=var_par)

        # Check that all static variables are used:
        all_constraints = "  ".join([p.expr for p in self.pars.values() if p.expr])
        var_names = list(self.static_variables.keys())
        for varname in var_names:
            regex = r'(^|[^a-z^A-Z^0-9_.])[+,-,*,\/]?(%s)[+,-,*,\/]?([^a-z^A-Z^0-9_.]|$)' % varname
            find_var = re.compile(regex)
            if find_var.search(all_constraints) is None:
                self.pars.pop(varname)
                self.static_variables.pop(varname)
                if self.verbose and verbose:
                    print(" [INFO] - unused variable was removed: %s" % varname)

        # --- mask spectral regions that should not be fitted
        if mask:
            for region in self.regions:
                # if region.new_mask:
                if region.new_mask:
                    if active_only and region.has_active_lines():
                        region.define_mask(z=self.redshift, dataset=self,
                                           z_sys=z_sys)
                    elif not active_only:
                        region.define_mask(z=self.redshift, dataset=self,
                                           z_sys=z_sys)

            if verbose and self.verbose:
                print("")
                print(" [DONE] - Spectral masks successfully created.")
                print("")

        self.ready2fit = True
        plt.close('all')

        # --- Check that all active elements have components defined:
        for line_tag in self.all_lines:
            ion = line_tag.split('_')[0]
            line = self.lines[line_tag]
            if ion not in self.components.keys() and line.active:
                if self.verbose:
                    print("")
                    print(" [ERROR] - Components are not defined for element: "+ion)
                    print("")
                self.ready2fit = False
                # TODO:
                # automatically open interactive window if components are not defined.
                error_msg = " [ERROR] - Components are not defined for element: %s \n" % ion
                return error_msg

        # -- Check all transitions of the given ions that are covered by the data:
        if check_lines:
            lines_not_defined = list()
            for this_ion in self.components.keys():
                for chunk in self.data:
                    wl_tot = chunk['wl']
                    lmin = wl_tot.min() / (self.redshift + 1.)
                    lmax = wl_tot.max() / (self.redshift + 1.)
                    cut = (lineList['l0'] > lmin) & (lineList['l0'] < lmax)
                    cut &= (lineList['ion'] == this_ion)
                    cut &= (lineList['l0'] >= l_lower) & (lineList['l0'] <= l_upper)
                    cut &= (lineList['f'] >= f_lower) & (lineList['f'] <= f_upper)
                    for entry in lineList[cut]:
                        if entry['trans'] in self.lines.keys():
                            pass
                        else:
                            lines_not_defined.append(entry)

            if len(lines_not_defined) > 0 and self.verbose:
                print("")
                print(term.red)
                print(" [WARNING]  -  Check-lines is activated")
                print(" The following lines of included ions are also covered by the data:")
                print(term.reset)
                for entry in lines_not_defined:
                    print(" %13s :  f = %.2e" % (entry[0], entry[3]))
                print("\n")

        if self.ready2fit:
            if verbose and self.verbose:
                print("\n  Dataset is ready to be fitted.")
                print("")
            return ""

    def fit(self, verbose=True, **kwargs):
        """
        Fit the absorption lines using chi-square minimization.

        Parameters
        ----------
        verbose : bool   [default = True]
            This will print the fit results to terminal.

        plot : bool   [default = False]
            This will make the best-fit solution show up in a new window.

        **kwargs
            Keyword arguments are derived from the `scipy.optimize`_ minimization methods.
            The default method is leastsq_, used in `lmfit <https://lmfit.github.io/lmfit-py/>`_.
            This can be changed using the `method` keyword.
            See documentation in lmfit_ and scipy.optimize_.

        rebin : int   [default = 1]
            Rebin data by a factor *rebin* before fitting.
            Passed as part of `kwargs`.

        sampling : int  [default = 3]
            Subsampling factor for the evaluation of the line profile.
            This is only used if the kernel is constant in velocity
            along the spectrum.
            Passed as part of `kwargs`.

        Returns
        -------
        popt : lmfit.MinimizerResult_
            The minimzer results from lmfit_ containing best-fit parameters
            and fit details, e.g., exit status and reduced chi squared.
            See documentation for lmfit_.

        chi2 : float
            The chi squared value of the best-fit. Note that this value is **not**
            the reduced chi squared. This value, and the number of degrees of freedom,
            are available under the `popt` object.


        .. _scipy.optimize: https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
        .. _leastsq: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html
        .. _lmfit.MinimizerResult: https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.MinimizerResult

        """

        if not self.ready2fit:
            if self.verbose:
                print(" [Error]  - Dataset is not ready to be fitted.")
                print("            Run `.prepare_dataset()` before fitting.")
            return None, None

        if 'rebin' in kwargs:
            rebin = kwargs.pop('rebin')
        else:
            rebin = 1

        if 'sampling' in kwargs:
            sampling = kwargs.pop('sampling')
        else:
            sampling = 3

        if rebin > 1 and self.verbose:
            print("\n  Rebinning the data by a factor of %i \n" % rebin)
            print(" [WARNING] - rebinning for LSF file kernel is not supported!")

        if self.verbose:
            print("\n  Fit is running... Please be patient.\n")

        def chi(pars):
            model = list()
            data = list()
            error = list()

            for reg_num, region in enumerate(self.regions):
                if region.has_active_lines():
                    x, y, err, mask = region.unpack()
                    if rebin > 1:
                        x, y, err = output.rebin_spectrum(x, y, err, rebin)
                        mask = output.rebin_bool_array(mask, rebin)
                        if isinstance(region.kernel, float):
                            nsub = region.kernel_nsub
                        else:
                            # Multiply the subsampling factor of the kernel by the rebin factor
                            nsub = region.kernel_nsub * rebin
                    else:
                        nsub = region.kernel_nsub

                    # Generate line profile
                    profile_obs = evaluate_profile(x, pars, self.lines.values(),
                                                   region.kernel, z_sys=self.redshift,
                                                   sampling=sampling, kernel_nsub=nsub)

                    if self.cheb_order >= 0:
                        cont_model = evaluate_continuum(x, pars, reg_num)
                    else:
                        cont_model = 1.

                    model.append((profile_obs*cont_model)[mask])
                    data.append(np.array(y[mask], dtype=myfloat))
                    error.append(np.array(err[mask], dtype=myfloat))

            model_spectrum = np.concatenate(model)
            data_spectrum = np.concatenate(data)
            error_spectrum = np.concatenate(error)

            residual = data_spectrum - model_spectrum
            return residual/error_spectrum

        self.minimizer = Minimizer(chi, self.pars, nan_policy='omit')
        # Set default values for `ftol` and `factor` if method is not given:
        if 'method' not in kwargs.keys():
            if 'factor' not in kwargs.keys():
                kwargs['factor'] = 1.
            if 'ftol' not in kwargs.keys():
                kwargs['ftol'] = 0.01
        popt = self.minimizer.minimize(**kwargs)
        self.best_fit = popt.params

        if self.cheb_order >= 0:
            # Normalize region data with best-fit polynomial:
            for reg_num, region in enumerate(self.regions):
                if not region.has_active_lines():
                    continue
                x, y, err, mask = region.unpack()
                cont_model = evaluate_continuum(x, self.best_fit, reg_num)
                region.flux /= cont_model
                region.err /= cont_model
                region.normalized = True

        if self.verbose and verbose:
            print("\n The fit has finished with the following exit message:")
            print("  " + popt.message)
            print("")

            if verbose:
                output.print_results(self, self.best_fit, velocity=True)
                # if self.cheb_order >= 0:
                #     output.print_cont_parameters(self)

        chi2 = popt.chisqr
        return popt, chi2

    def plot_fit(self, rebin=1, fontsize=12, xmin=None, xmax=None, max_rows=4,
                 ymin=None, ymax=None, filename=None,
                 subsample_profile=10, npad=50, loc='left',
                 highlight_props=None, residuals=True, norm_resid=False,
                 default_props={}, element_props={}, legend=True,
                 label_all_ions=False, xunit='vel',
                 line_props=None, hl_line_props=None):
        """
        Plot *all* the absorption lines and the best-fit profiles.
        For details, see :func:`VoigtFit.output.plot_all_lines`.
        """

        output.plot_all_lines(self, plot_fit=True, rebin=rebin, fontsize=fontsize,
                              xmin=xmin, xmax=xmax, max_rows=max_rows,
                              ymin=ymin, ymax=ymax,
                              filename=filename, loc=loc,
                              subsample_profile=subsample_profile, npad=npad,
                              residuals=residuals, norm_resid=norm_resid,
                              legend=legend, label_all_ions=label_all_ions,
                              default_props=default_props, element_props=element_props,
                              highlight_props=highlight_props, xunit=xunit,
                              line_props=line_props, hl_line_props=hl_line_props)
        plt.show()

    def velocity_plot(self, **kwargs):
        """
        Create a velocity plot, showing all the fitting regions defined, in order to compare
        different lines and to identify blends and contamination.
        """
        output.plot_all_lines(self, plot_fit=False, **kwargs)

    def plot_line(self, line_tag, index=0, plot_fit=False, loc='left', rebin=1,
                  nolabels=False, axis=None, fontsize=12,
                  xmin=None, xmax=None, ymin=None, ymax=None,
                  show=True, subsample_profile=1, npad=50,
                  residuals=True, norm_resid=False, legend=True,
                  default_props={}, element_props={}, highlight_props=None,
                  label_all_ions=False, xunit='velocity',
                  line_props=None, hl_line_props=None):
        """
        Plot a single fitting :class:`Region <regions.Region>`
        containing the line corresponding to the given `line_tag`.
        For details, see :func:`output.plot_single_line`.
        """
        output.plot_single_line(self, line_tag, index=index, plot_fit=plot_fit,
                                loc=loc, rebin=rebin, nolabels=nolabels,
                                axis=axis, fontsize=fontsize,
                                xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                                show=show, subsample_profile=subsample_profile,
                                npad=npad, residuals=residuals, norm_resid=norm_resid,
                                legend=legend, label_all_ions=label_all_ions,
                                default_props=default_props, element_props=element_props,
                                highlight_props=highlight_props, xunit=xunit,
                                line_props=line_props, hl_line_props=hl_line_props,
                                sort_f=False)

    def print_results(self, velocity=True, elements='all', systemic=None):
        """
        Print the best fit parameters.

        Parameters
        ----------
        velocity : bool   [default = True]
            If `True`, show the relative velocities of each component instead of redshifts.

        elements : list(str)   [default = 'all']
            A list of elements for which to show parameters.

        systemic : float   [default = None]
            The systemic redshift used as reference for the relative velocities.
        """
        output.print_results(self, self.best_fit, elements, velocity, systemic)

    def print_cont_parameters(self):
        """Print Chebyshev coefficients for the continuum fit."""
        output.print_cont_parameters(self)

    def print_metallicity(self, logNHI, err=0.1):
        """Print the total column densities for each element relative to
        HI in Solar units."""
        output.print_metallicity(self, self.best_fit, logNHI, err)

    def print_total(self):
        """Print the total column densities of all components."""
        output.print_total(self)

    def sum_components(self, ions, components=None):
        """
        Calculate the total column density for the given `components`
        of the given `ion`.

        Parameters
        ----------
        ions : str or list(str)
            List of ions or a single ion for which to calculate the total
            column density.

        components : list(int)
            List of integers corresponding to the indeces of the components
            to sum over.

        Returns
        -------
        total_logN : dict()
            Dictionary containing the log of total column density for each ion.

        total_logN_err : dict()
            Dictionary containing the error on the log of total column density
            for each ion.
        """
        if hasattr(self.best_fit, 'keys'):
            pass
        else:
            print(" [ERROR] - Best fit parameters are not found.")
            print("           Make sure the fit has converged...")
            return {}, {}

        if hasattr(ions, '__iter__'):
            pass
        else:
            ions = [ions]
        total_logN = dict()
        total_logN_err = dict()
        for ion in ions:
            logN = list()
            logN_err = list()
            if not components:
                N_comp = len(self.components[ion])
                comp_nums = range(N_comp)
            else:
                comp_nums = components
            for num in comp_nums:
                parname = 'logN%i_%s' % (num, ion)
                par = self.best_fit[parname]
                logN.append(par.value)
                logN_err.append(par.stderr)
            logN_pdf = [np.random.normal(n, e, 10000)
                        for n, e in zip(logN, logN_err)]
            logsum = np.log10(np.sum(10**np.array(logN_pdf), 0))
            total_logN[ion] = np.median(logsum)
            total_logN_err[ion] = np.std(logsum)

        return total_logN, total_logN_err

    def save_parameters(self, filename):
        """
        Save the best-fit parameters to ASCII table output.

        Parameters
        ----------
        filename : str
            Filename for the fit parameter file.
        """
        if self.best_fit:
            output.save_parameters_to_file(self, filename)
        else:
            print("\n [ERROR] - No fit parameters are defined.")

    def save_cont_parameters_to_file(self, filename):
        """
        Save the best-fit continuum parameters to ASCII table output.

        Parameters
        ----------
        filename : str   [default = None]
            If `None`, the :attr:`name <VoigtFit.DataSet.name>` attribute will be used.
        """
        if self.cheb_order >= 0:
            output.save_cont_parameters_to_file(self, filename)

    def save_fit_regions(self, filename=None, individual=False, path=''):
        """
        Save the fitting regions to ASCII table output.
        The format is as follows:
        (wavelength , normalized flux , normalized error , best-fit profile , mask)

        Parameters
        ----------
        filename : str   [default = None]
            Filename for the fitting regions.
            If `None`, the :attr:`name <VoigtFit.DataSet.name>` attribute will be used.

        individual : bool   [default = False]
            Save the fitting regions to individual files.
            By default all regions are concatenated into one file.

        path : str   [default = '']
            Specify a path to prepend to the filename in order to save output to a given
            directory or path. Can be given both as relative or absolute path.
            If the path doesn't end in `/` it will be appended automatically.
            The final filename will be:

                `path/` + `filename` [+ `_regN`] + `.reg`
        """
        if not filename:
            if self.name:
                filename = self.name + '.reg'
            else:
                print(" [ERROR] - Must specify dataset.name [dataset.set_name('name')]")
                print("           or give filename [dataset.save(filename='filename')]")
        output.save_fit_regions(self, filename, individual=individual, path=path)

    def save(self, filename=None, verbose=False):
        """Save the DataSet to file using the HDF5 format."""
        if not filename:
            if self.name:
                filename = self.name
            else:
                print(" [ERROR] - Must specify dataset.name [dataset.set_name('name')]")
                print("           or give filename [dataset.save(filename='filename')]")
        hdf5_save.save_hdf_dataset(self, filename, verbose=verbose)

    def get_NHI(self):
        if 'HI' in self.components.keys() and hasattr(self.best_fit, 'keys'):
            best_fit_NHI = self.best_fit['logN0_HI']
            return (best_fit_NHI.value, best_fit_NHI.stderr)

    def show_lines(self):
        """
        Print all defined lines to terminal.
        The output shows whether the line is active or not
        and the number of components for the given ion.
        """
        header = "%15s   State     " % 'Line ID'
        print(term.underline + header + term.reset)
        for line_tag, line in self.lines.items():
            active = 'active' if line.active else 'not active'
            fmt = '' if line.active else term.red
            output = "%15s : %s" % (line_tag, active)
            print(fmt + output + term.reset)

    def show_components(self, ion=None):
        """
        Show the defined components for a given `ion`.
        By default, all ions are shown.
        """
        z_sys = self.redshift
        for ion, comps in self.components.items():
            print("\n - %6s:" % ion)
            for num, comp in enumerate(comps):
                z = comp.z
                vel = (z - z_sys) / (z_sys + 1) * 299792.458
                print("   %2i  %+8.1f  %.6f   %6.1f   %5.2f" % (num, vel, z,
                                                                comp.b, comp.logN))

    def equivalent_width_limit(self, line_tag, ref=None, nofit=False, sigma=3., verbose=True, threshold=1.5):
        """
        Determine the equivalent width limit and corresponding limit on log(N).

        Parameters
        ----------
        line_tag : str
            The line ID of the absorption line for which the limit should be estimated.
            (Must be a line in the line-list of VoigtFit. Ex.: FeII_1611, TiII_1910, etc.)

        ref : str or `None`
            The reference line used to determine the integration range in velocity space,
            which contains 99.7% of the optical depth. By default, the strongest line of
            the same ionization state (e.g., I, II, IV, etc.) in the dataset is used.
            If the reference line has been fitted, the best-fit optical depth profile
            will be used, otherwise the range is determined by the observed apparent
            optical depth.

        nofit : bool   [default = False]
            If True, always use the observed apparent optical depth of the reference line
            (`ref`, see above).

        sigma : float   [default = 3]
            The significance level of the inferred upper limit, by default 3 sigma (99.7%) is used.

        verbose : bool   [default = True]
            Print informative messages from the function?

        threshold : float   [default = 15]
            The continuum noise threshold used to infer the edge of the apparent optical depth
            of the observed profile. The noise is estimated as the median noise per pixel

        """
        line = self.lines[line_tag]
        verbose = self.verbose | verbose
        regs_of_line = self.find_line(line_tag)
        reg = regs_of_line[0]
        if not reg.normalized:
            reg.normalize(z_sys=self.redshift)

        # Find a line that matches the ionization state
        # and determine the velocity extent of the line
        use_data = nofit
        if (self.best_fit is not None) and not use_data:
            if ref is not None:
                line_match = self.lines[ref]
            else:
                line_match, msg = match_ion_state(line, self.lines.values())
                if line_match is None:
                    if verbose:
                        print("           Could not find a matching line to determine velocity range.")
                        print(" [ERROR] - Aborting the measurement of equivalent width!\n")
                        return None
            # Check if best-fit parameters exist for the given line:
            if 'logN0_%s' % line_match.ion in self.best_fit:
                reg_match_all = self.find_line(line_match.tag)
                reg_match = reg_match_all[0]
                if not reg_match.normalized:
                    reg_match.normalize(z_sys=self.redshift)
                wl_ref = np.linspace(reg_match.wl.min(), reg_match.wl.max(), len(reg_match.wl)*10)
                lcen = line_match.l0 * (self.redshift + 1)
                vel_ref = (wl_ref - lcen) / lcen * 299792.458
                # vel_ref = reg_match.get_velocity(self.redshift, line_match.tag)
                profile = reg_match.evaluate_region(self.best_fit, wl=wl_ref, lines=[line_match])
                tau = -np.log(profile)
                vmin, vmax = tau_percentile(vel_ref, tau)
                use_data = False
                if verbose:
                    print("\n [INFO] - Determining limit for %s, using the fitted profile of %s as reference" % (line_tag, line_match.tag))
                    print(" [INFO] - Integrating from vel = %.1f to %.1f km/s\n" % (vmin, vmax))
            else:
                use_data = True

        if use_data:
            if ref is not None:
                line_match = self.lines[ref]
                reg_match_all = self.find_line(line_match.tag)
                reg_match = reg_match_all[0]
            else:
                matches = match_ion_state_all(line, self.lines.values())
                line_strength = [ll.l0 * ll.f for ll in matches]
                # Loop through the lines sorted by line-strength (strongest to weakest):
                for _, this_line in sorted(zip(line_strength, matches), reverse=True):
                    these_regs = self.find_line(this_line.tag)
                    this_reg = these_regs[0]
                    if len(this_reg.lines) > 1:
                        continue
                    flux = this_reg.flux
                    mask = this_reg.mask
                    if np.min(flux[mask]) > 0:
                        line_match = this_line
                        reg_match = this_reg
                        break
                else:
                    if verbose:
                        print("           Could not find a matching line to determine velocity range.")
                        print(" [ERROR] - Aborting the measurement of equivalent width!\n")
                    return None
            if not reg_match.normalized:
                reg_match.normalize(z_sys=self.redshift)
            vel_ref = reg_match.get_velocity(self.redshift, line_match.tag)
            _, flux, err, mask = reg_match.unpack()
            tau = -np.log(flux[mask])
            tau_err = err[mask] / flux[mask]
            vel_ref = vel_ref[mask]
            vmin, vmax = tau_noise_range(vel_ref, tau, tau_err, threshold=threshold)
            if verbose:
                print("\n [INFO] - Determining limit for %s, using the observed profile of %s as reference" % (line_tag, line_match.tag))
                print(" [INFO] - Integrating from vel = %.1f to %.1f km/s\n" % (vmin, vmax))

        vel = reg.get_velocity(self.redshift, line_tag)
        aper = (vel > vmin) & (vel < vmax)
        W_rest, W_err = equivalent_width(reg.wl, reg.flux, reg.err, aper=aper, z_sys=self.redshift)
        W_limit = sigma * W_err
        logN_limit = np.log10(1.13e20*W_limit / (line.l0**2 * line.f))
        logN = np.log10(1.13e20*W_rest / (line.l0**2 * line.f))
        logN_err = W_err / (W_rest * np.log(10))

        result = EquivalentWidth(W_rest=W_rest, W_err=W_err, logN=logN, logN_err=logN_err,
                                 logN_limit=logN_limit, line=line_tag, sigma=sigma)

        limit_fname = self.name + '_%s_limit.pdf' % line_tag
        output.plot_limit(self, line, line_match, vel_ref, tau, vmin, vmax, use_data, filename=limit_fname, EW=result)
        return result
