# -*- coding: UTF-8 -*-

__author__ = 'Jens-Kristian Krogager'
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import spline
import os

root_path = os.path.dirname(os.path.abspath(__file__))
datafile = root_path + '/static/telluric_em_abs.npz'

telluric_data = np.load(datafile)


def linfunc(x, a, b):
    """Linear fitting function of first order."""
    return a*x + b


class Region():
    def __init__(self, velspan, specID, line=None):
        """
        A Region contains the fitting data, exclusion mask and line information.
        The class is instantiated with the velocity span, `velspan`, and a spectral ID
        pointing to the raw data chunk from `DataSet.data`,
        and can include a :class:`dataset.Line` instance for the first line
        belonging to the region.

        .. rubric:: Attributes

        velspan : float
            The velocity range to used for the fitting region.

        lines : list(:class:`dataset.Line`)
            A list of Lines defined in the region.

        label : str
            A LaTeX label describing the lines in the region for plotting purposes.

        res : float
            Spectral resolution of the region in km/s.

        wl : array_like, shape (N)
            Data array of wavelengths in Ångstrøm.

        flux : array_like, shape (N)
            Data array of fluxes (normalized if :attr:`normalized` is `True`).

        err : array_like, shape (N)
            Array of uncertainties for each flux element.

        normalized : bool
            `True` if the data in the region are normlized.

        mask : array_like, shape (N)
            Exclusion mask for the region:
            0/`False` = pixel is *not* included in the fit.
            1/`True` = pixel is included in the fit.

        new_mask : bool
            Internal parameter for :meth:`VoigtFit.DataSet.prepare_dataset`.
            If `True`, an interactive masking process will be initiated in the
            preparation stage.

        cont_err : float
            An estimate of the uncertainty in the continuum fit.

        specID : str
            A spectral identifier to point back to the raw data chunk.

        """
        self.velspan = velspan
        self.specID = specID
        if line:
            self.lines = [line]
        else:
            self.lines = list()
        self.label = ''

    def add_data_to_region(self, data_chunk, cutout):
        """
        Define the spectral data for the fitting region.

        Parameters
        ----------
        data_chunk : dict()
            A `data_chunk` as defined in the data structure of :meth:`DataSet.data
            <VoigtFit.DataSet.add_data>`.

        cutout : bool array
            A boolean array defining the subset of the `data_chunk` which makes up the fitting region.
        """
        self.res = data_chunk['res']
        self.err = data_chunk['error'][cutout]
        self.flux = data_chunk['flux'][cutout]
        self.wl = data_chunk['wl'][cutout]
        self.normalized = data_chunk['norm']
        self.cont_err = 0.
        self.mask = data_chunk['mask'][cutout]
        if np.sum(self.mask) == len(self.mask):
            # If all pixels are 1 in the given mask,
            # let the user define new_mask in `prepare_dataset`:
            self.new_mask = True
        else:
            self.new_mask = False

    def add_line(self, line):
        """Add a new :class:`dataset.Line` to the fitting region."""
        self.lines.append(line)

    def has_line(self, line_tag):
        """Return `True` if a line with the given `line_tag` is defined in the region."""
        for line in self.lines:
            if line.tag == line_tag:
                return True

        return False

    def has_active_lines(self):
        """Return `True` is at least one line in the region is active."""
        active_lines = [line.active for line in self.lines]
        if np.any(active_lines):
            return True

        return False

    def remove_line(self, line_tag):
        """Remove absorption line with the given `line_tag` from the region."""
        if self.has_line(line_tag):
            for num, line in enumerate(self.lines):
                if line.tag == line_tag:
                    num_to_remove = num
            self.lines.pop(num_to_remove)

    def normalize(self, plot=True, norm_method='linear'):
        """
        Normalize the region if the data are not already normalized.
        Choose from two methods:

            1:  define left and right continuum regions
                and fit a linear continuum.

            2:  define the continuum as a range of points
                and use spline interpolation to infer the
                continuum.
        """

        if norm_method in ['linear', 'spline']:
            pass
        else:
            err_msg = "Invalid norm_method: %r" % norm_method
            raise ValueError(err_msg)

        plt.close('all')

        plt.figure()
        dx = 0.1*(self.wl.max() - self.wl.min())
        lines_title_string = ", ".join([line.tag for line in self.lines])
        plt.xlim(self.wl.min()-dx, self.wl.max()+dx)
        plt.ylim(0.8*self.flux.min(), 1.2*self.flux.max())
        plt.plot(self.wl, self.flux, color='k', drawstyle='steps-mid',
                 label=lines_title_string)
        plt.xlabel("Wavelength  [${\\rm \AA}$]")

        if norm_method == 'linear':
            # - Normalize by defining a left and right continuum region

            print "\n\n  Mark left continuum region, left and right boundary."
            plt.title("Mark left continuum region, left and right boundary.")

            bounds = plt.ginput(2, -1)
            left_bound = min(bounds[0][0], bounds[1][0])
            right_bound = max(bounds[0][0], bounds[1][0])
            region1 = (self.wl >= left_bound)*(self.wl <= right_bound)
            fit_wl = self.wl[region1]
            fit_flux = self.flux[region1]

            lines_title_string = ", ".join([line.tag for line in self.lines])
            plt.title(lines_title_string)
            print "\n  Mark right continuum region, left and right boundary."
            plt.title("Mark right continuum region, left and right boundary.")
            bounds = plt.ginput(2)
            left_bound = min(bounds[0][0], bounds[1][0])
            right_bound = max(bounds[0][0], bounds[1][0])
            region2 = (self.wl >= left_bound)*(self.wl <= right_bound)
            fit_wl = np.concatenate([fit_wl, self.wl[region2]])
            fit_flux = np.concatenate([fit_flux, self.flux[region2]])

            popt, pcov = curve_fit(linfunc, fit_wl, fit_flux)

            continuum = linfunc(self.wl, *popt)
            e_continuum = np.std(fit_flux - linfunc(fit_wl, *popt))

        elif norm_method == 'spline':
            # Normalize by drawing the continuum and perform spline
            # interpolation between the points

            print "\n\n Select a range of continuum spline points over the whole range"
            plt.title(" Select a range of continuum spline points over the whole range")
            points = plt.ginput(n=-1, timeout=-1)
            xk, yk = [], []
            for x, y in points:
                xk.append(x)
                yk.append(y)
            xk = np.array(xk)
            yk = np.array(yk)
            region_wl = self.wl.copy()
            continuum = spline(xk, yk, region_wl, order=3)
            e_continuum = np.sqrt(np.mean(self.err**2))

        if plot:
            new_flux = self.flux/continuum
            new_err = self.err/continuum
            plt.cla()
            plt.plot(self.wl, new_flux, color='k', drawstyle='steps-mid',
                     label=lines_title_string)
            plt.xlabel("Wavelength  [${\\rm \AA}$]")
            plt.title("Normalized")
            plt.axhline(1., ls='--', color='k')
            plt.axhline(1.+e_continuum/np.mean(continuum), ls=':', color='gray')
            plt.axhline(1.-e_continuum/np.mean(continuum), ls=':', color='gray')
            plt.draw()

            plt.title("Go back to terminal...")
            prompt = raw_input(" Is normalization correct?  (YES/no) ")
            if prompt.lower() in ['', 'y', 'yes']:
                self.flux = new_flux
                self.err = new_err
                self.cont_err = e_continuum/np.median(continuum)
                self.normalized = True
                return 1

            else:
                return 0

        else:
            self.flux = self.flux/continuum
            self.err = self.err/continuum
            self.cont_err = e_continuum/np.mean(continuum)
            self.normalized = True
            return 1

    def define_mask(self, z=None, dataset=None, telluric=True):
        """
        Use an interactive window to define the mask for the region.

        Parameters
        ----------
        z : float   [default = None]
            If a redshift is given, the lines in the region are shown as vertical lines
            at the given redshift.

        dataset : :class:`VoigtFit.DataSet`   [default = None]
            A dataset with components defined for the lines in the region.
            If a dataset is passed, the components of the lines in the region are shown.

        telluric : bool   [default = True]
            Show telluric absorption and sky emission line templates during the masking.
        """
        plt.close('all')

        plt.xlim(self.wl.min(), self.wl.max())
        # plt.ylim(max(0, 0.8*self.flux.min()), 1.2)
        lines_title = ", ".join([line.tag for line in self.lines])

        masked_spectrum = np.ma.masked_where(self.mask, self.flux)
        plt.plot(self.wl, self.flux, color='k', drawstyle='steps-mid', lw=0.5,
                 label=lines_title)
        plt.xlabel("Wavelength  [${\\rm \AA}$]")
        mask_line = plt.plot(self.wl, masked_spectrum, color='r', lw=1.5,
                             drawstyle='steps-mid', zorder=0)
        plt.legend()
        if telluric:
            wl_T = telluric_data['wl']
            cutout = (wl_T > self.wl.min()) * (wl_T < self.wl.max())
            flux_T = telluric_data['em'][cutout]
            abs_T = telluric_data['abs'][cutout]
            wl_T = wl_T[cutout]
            if self.normalized:
                cont = 1.
            else:
                cont = np.median(self.flux)
            plt.plot(wl_T, abs_T*1.2*cont, color='crimson', alpha=0.7, lw=0.5)
            # -- Test if telluric template is defined in this region:
            if len(flux_T) > 0:
                plt.plot(wl_T, (flux_T/flux_T.max() + 1.2)*cont,
                         color='orange', alpha=0.7, lw=0.5)

        if z is not None:
            for line in self.lines:
                # Load line properties
                l0, f, gam = line.get_properties()
                if dataset is not None:
                    ion = line.ion
                    n_comp = len(dataset.components[ion])
                    ion = ion.replace('*', 'x')
                    for n in range(n_comp):
                        z = dataset.pars['z%i_%s' % (n, ion)].value
                        plt.axvline(l0*(z+1), ls=':', color='r', lw=0.4)
                else:
                    plt.axvline(l0*(z+1), ls=':', color='r', lw=0.4)

        plt.title("Mark regions to mask, left and right boundary.")
        print "\n\n  Mark regions to mask, left and right boundary."
        plt.draw()

        ok = 0
        mask_vlines = list()
        while ok >= 0:
            sel = plt.ginput(0, timeout=-1)

            if len(sel) > 0 and len(sel) % 2 == 0:
                mask = self.mask.copy()
                sel = np.array(sel)
                selections = np.column_stack([sel[::2, 0], sel[1::2, 0]])
                for x1, x2 in selections:
                    cutout = (self.wl >= x1)*(self.wl <= x2)
                    mask[cutout] = False
                    mask_vlines.append(plt.axvline(x1, color='r', ls='--'))
                    mask_vlines.append(plt.axvline(x2, color='r', ls='--'))

                masked_spectrum = np.ma.masked_where(mask, self.flux)
                mask_line = plt.plot(self.wl, masked_spectrum, color='r', drawstyle='steps-mid')

                plt.draw()
                prompt = raw_input("Are the masked regions correct? (YES/no/clear)")
                if prompt.lower() in ['', 'y', 'yes']:
                    ok = -1
                    self.mask = mask
                    self.new_mask = False

                elif prompt.lower() in ['c', 'clear']:
                    ok = 0
                    self.mask = np.ones_like(mask, dtype=bool)
                    for linesegment in mask_line:
                        linesegment.remove()
                    mask_line = list()

                    for linesegment in mask_vlines:
                        linesegment.remove()
                    mask_vlines = list()

                else:
                    self.mask = mask
                    ok += 1

            elif len(sel) == 0:
                print "\nNo masks were defined."
                prompt = raw_input("Continue? (yes/no)")
                if prompt.lower() in ['', 'y', 'yes']:
                    ok = -1
                    self.new_mask = False
                else:
                    ok += 1

    def set_mask(self, mask):
        err_msg = " Mask must have same size as region!"
        assert len(mask) == len(self.flux), err_msg
        self.mask = mask

    def clear_mask(self):
        """Clear the already defined mask in the region."""
        self.mask = np.ones_like(self.wl, dtype=bool)
        self.new_mask = True

    def unpack(self):
        """Return the data of the region (wl, flux, error, mask)"""
        return (self.wl, self.flux, self.err, self.mask)

    def is_normalized(self):
        """Return `True` if the region data is normalized."""
        return self.normalized

    def set_label(self, text):
        """Set descriptive text label for the given region."""
        self.label = text

    def generate_label(self, active_only=True, ignore_finelines=True):
        """Automatically generate a descriptive label for the region."""
        transition_lines = list()
        if active_only and not ignore_finelines:
            for line in self.lines:
                if line.active is True:
                    transition_lines.append(line.tag)

        elif active_only and ignore_finelines:
            for line in self.lines:
                if line.active is True and line.ion[-1].isupper():
                    transition_lines.append(line.tag)

        elif not active_only and ignore_finelines:
            for line in self.lines:
                if line.ion[-1].isupper():
                    transition_lines.append(line.tag)

        else:
            for line in self.lines:
                transition_lines.append(line.tag)

        all_trans_str = ["${\\rm "+trans.replace('_', '\ \\lambda')+"}$"
                         for trans in transition_lines]
        line_string = "\n".join(all_trans_str)

        self.label = line_string
