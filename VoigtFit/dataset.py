# -*- coding: UTF-8 -*-

__author__ = 'Jens-Kristian Krogager'

import numpy as np
import matplotlib.pyplot as plt
import copy
import os

from lmfit import Parameters, Minimizer

from voigt import evaluate_profile, evaluate_continuum
from regions import Region
import output
import line_complexes
from line_complexes import fine_structure_complexes
import Asplund
import hdf5_save

options = {'nsamp': 1,
           'npad': 20}
myfloat = np.float64

root_path = os.path.dirname(os.path.abspath(__file__))
atomfile = root_path + '/static/atomdata_updated.dat'

lineList = np.loadtxt(atomfile, dtype=[('trans', 'S13'),
                                       ('ion', 'S6'),
                                       ('l0', 'f4'),
                                       ('f', 'f4'),
                                       ('gam', 'f4'),
                                       ('mass', 'f4')])


def calculate_velocity_bin_size(x):
    """Calculate the bin size of *x* in velocity units."""
    log_x = np.logspace(np.log10(x.min()), np.log10(x.max()), len(x))
    return np.diff(log_x)[0] / log_x[0] * 299792.458


class Line(object):
    def __init__(self, tag, active=True):
        """
        Line object containing atomic data for the given transition.
        Only the line_tag is passed, the rest of the information is
        looked up in the atomic database.

        .. rubric:: Attributes

        tag : str
            The line tag for the line, e.g., "FeII_2374"

        ion : str
            The ion for the line; The ion for "FeII_2374" is "FeII".

        element : str
            Equal to ``Line.ion`` for backwards compatibility.

        l0 : float
            Rest-frame resonant wavelength of the transition.
            Unit: Ångstrøm.

        f : float
            The oscillator strength for the transition.

        gam : float
            The radiation damping constant or Einstein coefficient.

        mass : float
            The atomic mass in atomic mass units.

        active : bool   [default = True]
            The state of the line in the dataset. Only active lines will
            be included in the fit.

        """
        self.tag = tag
        index = lineList['trans'].tolist().index(tag)
        tag, ion, l0, f, gam, mass = lineList[index]

        self.tag = tag
        self.ion = ion
        self.element = ion
        self.l0 = l0
        self.f = f
        self.gam = gam
        self.mass = mass
        self.active = active

    def get_properties(self):
        """Return the principal atomic constants for the transition: *l0*, *f*, and *gam*."""
        return (self.l0, self.f, self.gam)

    def set_inactive(self):
        """Set the line inactive; exclude the line in the fit."""
        self.active = False

    def set_active(self):
        """Set the line active; include the line in the fit."""
        self.active = True


# --- Definition of main class *DataSet*:
class DataSet(object):
    def __init__(self, redshift, name=''):
        """
        Main class of the package ``VoigtFit``. The DataSet handles all the major parts of the fit.
        Spectral data must be added using the :meth:`add_data <dataset.DataSet.add_data>` method.
        Hereafter the absorption lines to be fitted are added to the DataSet using the
        :meth:`add_line <dataset.DataSet.add_line>` or
        :meth:`add_many_lines <dataset.DataSet.add_many_lines>` methods.
        Lastly, the components of each element is defined using the
        :meth:`add_component <dataset.DataSet.add_component>` method.
        When all lines and components have been defined, the DataSet must be prepared by
        calling the :meth:`prepare_dataset <dataset.DataSet.prepare_dataset>`
        method and subsequently, the lines can be fitted using
        the :meth:`fit <dataset.DataSet.fit>` method.

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
            See :meth:`DataSet.add_data <dataset.DataSet.add_data>`.

        lines : dict
            A dictionary holding pairs of defined (*line_tag* : :class:`dataset.Line`)

        all_lines : list(str)
            A list of all the defined *line tags* for easy look-up.

        molecules : dict
            A dictionary holding a list of the defined molecular bands for each molecule:
            {*molecule* : list(str)}

        regions : list(:class:`regions.Region`)
            A list of the fitting regions.

        cheb_order : int   [default = 1]
            The maximum order of Chebyshev polynomials to use for the continuum
            fitting in each region.

        norm_method : str   [default = 'linear']
            Default normalization method to use for interactive normalization
            if Chebyshev polynomial fitting should not be used.

        components : dict
            A dictionary of components for each *ion* defined:
            (*ion* : [z, b, logN, options]). See :meth:`DataSet.add_component
            <dataset.DataSet.add_component>`.

        velspan : float   [default = 500]
            The default velocity range to use for the definition
            of fitting regions.

        ready2fit : bool   [default = False]
            This attribute is checked before fitting the dataset. Only when
            the attribute has been set to `True` can the dataset be fitted.
            This will be toggled after a successful run of
            :meth:`DataSet.prepare_dataset <dataset.DataSet.prepare_dataset>`.

        best_fit : lmfit.Parameters_   [default = None]
            Best-fit parameters from lmfit_. This attribute will be `None` until
            the dataset has been fitted.

        pars : lmfit.Parameters_   [default = None]
            Placeholder for the fit parameters initiated before the fit.
            The parameters will be defined during the call to :meth:`DataSet.prepare_dataset
            <dataset.DataSet.prepare_dataset>` based on the defined components.


        .. _lmfit.Parameters: https://lmfit.github.io/lmfit-py/parameters.html

        """
        # Define the systemic redshift
        self.redshift = redshift

        # container for data chunks to be fitted
        # data should be added by calling method 'add_data'
        self.data = list()

        self.verbose = True

        # container for absorption lines. Each line is defined as a class 'Line'.
        # a dictionary containing a Line class for each line-tag key:
        self.lines = dict()
        # a list containing all the line-tags defined. The same as lines.keys()
        self.all_lines = list()
        # a dictionary conatining a list of bands defined for each molecule:
        # Ex: self.molecules = {'CO': ['AX(0-0)', 'AX(1-0)']}
        self.molecules = dict()

        # container for the fitting regions containing Lines
        # each region is defined as a class 'Region'
        self.regions = list()
        self.cheb_order = 1
        self.norm_method = 'linear'

        # Number of components in each ion
        self.components = dict()

        # Define default velocity span for fitting region
        self.velspan = 500.  # km/s

        self.ready2fit = False
        self.best_fit = None
        self.pars = None
        self.name = name

    def set_name(self, name):
        """Set the name of the DataSet. This parameter is used when saving the dataset."""
        self.name = name

    def get_name(self):
        """Returns the name of the DataSet."""
        return self.name

    def add_data(self, wl, flux, res, err=None, normalized=False):
        """
        Add spectral data to the DataSet. This will be used to define fitting regions.

        Parameters
        ----------
        wl : ndarray, shape (n)
            Input vacuum wavelength array in Ångstrøm.

        flux : ndarray, shape (n)
            Input flux array, should be same length as wl

        res : float
            Spectral resolution in km/s  (c/R)

        err : ndarray, shape (n)   [default = None]
            Error array, should be same length as wl
            If `None` is given, a constant uncertainty of 1. is given to all pixels.

        normalized : bool   [default = False]
            If the input spectrum is normalized this should be given as True
            in order to skip normalization steps.
        """
        # assign specid:
        specid = "sid_%i" % len(self.data)

        if err is None:
            err = np.ones_like(flux)

        self.data.append({'wl': wl, 'flux': flux,
                          'error': err, 'res': res,
                          'norm': normalized, 'specID': specid})

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

    def reset_all_regions(self):
        """
        Reset the data in all :class:`Regions <regions.Region>`
        defined in the DataSet to use the raw input data.
        """
        for reg in self.regions:
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
                    output_msg = " Spectral resolution in the region around %s is %.1f km/s."
                    print output_msg % (line_tag, region.res)
                resolutions.append(region.res)
            return resolutions

    def set_resolution(self, res, line_tag=None, verbose=True):
        """
        Set the spectral resolution in km/s for the :class:`Region <regions.Region>`
        containing the line with the given `line_tag`. If multiple spectra are fitted
        simultaneously, this method will set the same resolution for *all* spectra.
        If `line_tag` is not given, the resolution will be set for *all* regions,
        including the raw data chunks defined in :attr:`dataset.DataSet.data`!

        Note -- If not all data chunks have the same resolution, this method
        should be used with caution. It is advised to check the spectral resolution beforehand
        and only update the appropriate regions using a for-loop.
        """
        if line_tag:
            regions_of_line = self.find_line(line_tag)
            for region in regions_of_line:
                region.res = res

        else:
            if verbose:
                warn_msg = " [WARNING] - Setting spectral resolution for all regions, R=%.1f km/s!"
                print warn_msg % res

            for region in self.regions:
                region.res = res

            for chunk in self.data:
                chunk['res'] = res

    def set_systemic_redshift(self, z_sys):
        """Update the systemic redshift of the dataset"""
        self.redshift = z_sys

    def remove_line(self, line_tag):
        """
        Remove an absorption line from the DataSet. If this is the last line in a fitting region
        the given region will be eliminated, and if this is the last line of a given ion,
        then the components will be eliminated for that ion.

        Parameters
        ----------
        line_tag : str
            Line tag of the transition that should be removed.
        """
        if line_tag in self.all_lines:
            self.all_lines.remove(line_tag)
            if line_tag in self.lines.keys():
                self.lines.pop(line_tag)

        # --- Check if the ion has transistions defined in other regions
        ion = line_tag.split('_')[0]
        ion_defined_elsewhere = False
        for line_tag in self.all_lines:
            if line_tag.find(ion) >= 0:
                ion_defined_elsewhere = True

        # --- If it is not defined elsewhere, remove it from components
        if not ion_defined_elsewhere:
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
                print ""
                print " The line is not defined. Nothing to remove."

    def normalize_line(self, line_tag, norm_method='spline'):
        """
        Normalize or re-normalize a given line

        Parameters
        ----------
        line_tag : str
            Line tag of the line whose fitting region should be normalized.

        norm_method : str   [default = 'spline']
            Normalization method used for the interactive continuum fit.
            Should be on of: ["spline", "linear"]
        """
        if norm_method == 'linear':
            norm_num = 1
        elif norm_method == 'spline':
            norm_num = 2
        else:
            err_msg = "Invalid norm_method: %r" % norm_method
            raise ValueError(err_msg)

        regions_of_line = self.find_line(line_tag)
        for region in regions_of_line:
            region.normalize(norm_method=norm_num)

    def mask_line(self, line_tag, reset=True, mask=None, telluric=True):
        """
        Define exclusion masks for the fitting region of a given line.
        Note that the masked regions are exclusion regions and will not be used for the fit.
        If components have been defined, these will be shown as vertical lines.

        Parameters
        ----------
        line_tag : str
            Line tag for the :class:`Line <dataset.DataSet.Line>` whose
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
        """
        regions_of_line = self.find_line(line_tag)
        for region in regions_of_line:
            if reset:
                region.clear_mask()

            if hasattr(mask, '__iter__'):
                region.mask = mask
                region.new_mask = False
            else:
                region.define_mask(z=self.redshift, dataset=self, telluric=telluric)

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
                print "\n The line (%s) is not defined." % line_tag

        return None

    def activate_line(self, line_tag):
        """Activate a given line defined by its `line_tag`"""
        if line_tag in self.lines.keys():
            line = self.lines[line_tag]
            line.set_active()

        else:
            regions_of_line = self.find_line(line_tag)
            for region in regions_of_line:
                for line in region.lines:
                    if line.tag == line_tag:
                        line.set_active()

    def deactivate_line(self, line_tag):
        """
        Deactivate a given line defined by its `line_tag`.
        This will exclude the line during the fit.
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

        # --- Check if the ion has transistions defined in other regions
        ion = line_tag.split('_')[0]
        ion_defined_elsewhere = False
        for line_tag in self.all_lines:
            if line_tag.find(ion) >= 0:
                ion_defined_elsewhere = True

        # --- If it is not defined elsewhere, remove it from components
        if not ion_defined_elsewhere:
            self.components.pop(ion)

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

    def reset_components(self, ion=None):
        """
        Reset component structure for a given ion.

        Parameters
        ----------
        ion : str   [default = None]
            The ion for which to reset the components: e.g., 'FeII', 'HI', 'CIa', etc.
            If `None`is given, *all* components for *all* ions will be reset.
        """

        if ion:
            if ion in self.components.keys():
                self.components.pop(ion)
            else:
                if self.verbose:
                    print " [ERROR] - No components defined for ion: %s" % ion

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
        options = {'var_z': var_z, 'var_b': var_b, 'var_N': var_N, 'tie_z': tie_z, 'tie_b': tie_b,
                   'tie_N': tie_N}
        if ion in self.components.keys():
            self.components[ion].append([z, b, logN, options])
        else:
            self.components[ion] = [[z, b, logN, options]]

    def add_component_velocity(self, ion, v, b, logN,
                               var_z=True, var_b=True, var_N=True, tie_z=None, tie_b=None, tie_N=None):
        """
        Same as for `add_component()` but input is given as relative velocity instead of redshift.
        """
        options = {'var_z': var_z, 'var_b': var_b, 'var_N': var_N, 'tie_z': tie_z, 'tie_b': tie_b,
                   'tie_N': tie_N}
        z = self.redshift + v/299792.458*(self.redshift + 1.)
        if ion in self.components.keys():
            self.components[ion].append([z, b, logN, options])
        else:
            self.components[ion] = [[z, b, logN, options]]

    def interactive_components(self, line_tag):
        """
        Define components interactively for a given ion. The components will be defined on the
        basis of the given line for that ion. If the line is defined in several spectral
        then the interactive window will show up for each.
        Running the interactive mode more times for different transitions of the same ion
        will append the components to the structure.
        If no components should be added, then simply click `enter` to terminate the process
        for the given transition.

        Parameters
        ----------
        line_tag : str
            Line tag for the line belonging to the ion for which components should be defined.

        Notes
        -----
        This will launch an interactive plot showing the fitting region of the given line.
        The user can then click on the positions of the components which. At the end, the
        redshifts and estimated column densities are printed to terminal. The b-parameter
        is assumed to be unresolved, i.e., taken from the resolution.
        """
        regions_of_line = self.find_line(line_tag)
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

            ax.plot(wl, masked_range, color='0.7', drawstyle='steps-mid', lw=0.9)
            ax.plot(wl, flux, 'k', drawstyle='steps-mid')

            line = self.lines[line_tag]
            if line.element in self.components.keys():
                l0, f, gam = line.get_properties()
                ion = line.ion
                for comp in self.components[ion]:
                    z = comp[0]
                    ax.axvline(l0*(z+1), ls=':', color='r', lw=0.4)

            if region.normalized:
                c_level = 1.
            else:
                ax.set_title("Click to Mark Continuum Level...")
                cont_level_point = plt.ginput(1, 30.)
                c_level = cont_level_point[0][1]
                ax.axhline(c_level, color='0.3', ls=':')

            ax.set_title("Mark central components for %s, finish with [enter]" % line.element)
            ax.set_xlabel(u"Wavelength  (Å)")
            if region.normalized:
                ax.set_ylabel(u"Normalized Flux")
            else:
                ax.set_ylabel(u"Flux")
            comps = plt.ginput(-1, 60)
            num = 0
            # Assume that components are unresolved:
            b = region.res/2.35482
            comp_list = list()
            for x0, y0 in comps:
                z0 = x0/line.l0 - 1.
                # Calculate logN from peak depth:
                y0 = max(y0/c_level, 0.01)
                logN = np.log10(-b * np.log(y0) / (1.4983e-15 * line.l0 * line.f))
                print "Component %i:  z = %.6f   log(N) = %.2f" % (num, z0, logN)
                ax.axvline(x0, color='darkblue', alpha=0.8)
                comp_list.append([z0, b, logN])
                num += 1
            plt.draw()

            if len(comp_list) > 0:
                for z, b, logN in comp_list:
                    self.add_component(line.element, z, b, logN)
            else:
                pass

    def delete_component(self, ion, index):
        """Remove component of the given `ion` with the given `index`."""
        if ion in self.components.keys():
            self.components[ion].pop(index)

        else:
            if self.verbose:
                print " [ERROR] - No components defined for ion: "+ion

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
        self.components[to_ion] = []

        if ref_comp is not None:
            offset_N = logN - reference[ref_comp][2]
        else:
            # Strip ionization state to get element:
            ion_tmp = to_ion[:1] + to_ion[1:].replace('I', '')
            element = ion_tmp[:1] + ion_tmp[1:].replace('V', '')
            anchor_tmp = from_ion[:1] + from_ion[1:].replace('I', '')
            element_anchor = anchor_tmp[:1] + anchor_tmp[1:].replace('V', '')
            # Use Solar abundance ratios:
            offset_N = Asplund.photosphere[element][0] - Asplund.photosphere[element_anchor][0]
        for num, comp in enumerate(reference):
            new_comp = copy.deepcopy(comp)
            if logN:
                new_comp[2] += offset_N
            if tie_z:
                new_comp[3]['tie_z'] = 'z%i_%s' % (num, from_ion)
            if tie_b:
                new_comp[3]['tie_b'] = 'b%i_%s' % (num, from_ion)

            self.components[to_ion].append(new_comp)

    def load_components_from_file(self, fname):
        """Load best-fit parameters from an output file `fname`."""
        parameters = open(fname)
        components_to_add = list()
        all_ions_in_file = list()
        for line in parameters.readlines():
            line = line.strip()
            if len(line) == 0:
                pass
            elif line[0] == '#':
                pass
            else:
                pars = line.split()
                ion = pars[1]
                z = float(pars[2])
                b = float(pars[4])
                logN = float(pars[6])
                components_to_add.append([ion, z, b, logN])
                if ion not in all_ions_in_file:
                    all_ions_in_file.append(ion)

        for ion in all_ions_in_file:
            if ion in self.components.keys():
                self.reset_components(ion)

        for comp_pars in components_to_add:
            ion, z, b, logN = comp_pars
            self.add_component(ion, z, b, logN)
        parameters.close()

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
                comp[3]['var_b'] = False
                comp[3]['var_z'] = False
        else:
            for ion in self.components.keys():
                for comp in self.components[ion]:
                    comp[3]['var_b'] = False
                    comp[3]['var_z'] = False

    def add_line(self, line_tag, velspan=None, active=True):
        """
        Add an absorption line to the DataSet.

        Parameters
        ----------
        line_tag : str
            The line tag for the transition which should be defined: e.g., "FeII_2374"

        velspan : float   [default = None]
            The velocity span around the line center, which will be included in the fit.
            If `None` is given, use the default `self.velspan` defined (500 km/s).

        active : bool   [default = True]
            Set the :class:`Line <dataset.DataSet.Line>` as active
            (i.e., included in the fit).

        Notes
        -----
        This will initiate a :class:`Line <dataset.DataSet.Line>` class
        with the atomic data for the transition, as well as creating a
        fitting :class:`Region <regions.Region>` containing the data cutout
        around the line center.
        """

        self.ready2fit = False
        if line_tag in self.all_lines:
            if self.verbose:
                print " [WARNING] - %s is already defined." % line_tag
            return False

        if line_tag in lineList['trans']:
            new_line = Line(line_tag)
        else:
            if self.verbose:
                print "\nThe transition (%s) not found in line list!" % line_tag
            return False

        if velspan is None:
            velspan = self.velspan

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
                    span = ((vel >= -velspan)*(vel <= velspan)).nonzero()[0]
                    new_wavelength = wl[span]

                    # Initiate new Region:
                    new_region = Region(velspan, chunk['specID'])
                    new_region.add_line(new_line)

                    # check if the line overlaps with another already defined region
                    if len(self.regions) > 0:
                        merge = -1
                        for num, region in enumerate(self.regions):
                            # Only allow regions arising from the same chunk to be merged:
                            if chunk['specID'] == region.specID:
                                if np.intersect1d(new_wavelength, region.wl).any():
                                    merge = num

                        if merge >= 0:
                            # If the region overlaps with another merge the two regions:
                            new_region.lines += self.regions[merge].lines

                            # merge the wavelength region
                            region_wl = np.union1d(new_wavelength, self.regions[merge].wl)

                            # and remove the overlapping region from the dataset
                            self.regions.pop(merge)

                        else:
                            region_wl = new_wavelength

                    else:
                        region_wl = new_wavelength

                    # Wavelength has now been defined and merged
                    # Cutout spectral chunks and add them to the Region
                    cutout = (wl >= region_wl.min()) * (wl <= region_wl.max())

                    new_region.add_data_to_region(chunk, cutout)

                    self.regions.append(new_region)
                    if line_tag not in self.all_lines:
                        self.all_lines.append(line_tag)
                        self.lines[line_tag] = new_line
                    success = True

            if not success:
                if self.verbose:
                    print "\n [ERROR] - The given line is not covered by the spectral data: " + line_tag
                    print ""
                return False

        else:
            if self.verbose:
                print " [ERROR]  No data is loaded. Run method `add_data` to add spectral data."

    def add_many_lines(self, tags, velspan=None):
        """
        Add many lines at once.

        Parameters
        ----------
        tags : list(str)
            A list of line tags for the transitions that should be added.

        velspan : float   [default = None]
            The velocity span around the line center, which will be included in the fit.
            If `None` is given, use the default :attr:`velspan <dataset.DataSet.velspan>`
            defined (500 km/s).
        """

        self.ready2fit = False
        if hasattr(velspan, '__iter__'):
            for tag, v in zip(tags, velspan):
                self.add_line(tag, v)
        elif velspan is None:
            for tag in tags:
                self.add_line(tag, self.velspan)
        else:
            for tag in tags:
                self.add_line(tag, velspan)

    def add_fine_lines(self, line_tag, levels=None, full_label=False):
        """
        Add fine-structure line complexes by providing only the main transition.
        This function is mainly useful for the CI complexes, where the many lines are closely
        located and often blended.

        Parameters
        ----------
        line_tag : str
            Line tag for the ground state transition, e.g., "CI_1656"

        levels : str, list(str)    [default = None]
            The levels of the fine-structure complexes to add, starting with "a" referring
            to the first excited level, "b" is the second, etc..
            Several levels can be given at once: ['a', 'b']
            By default, all levels are included.

        full_label : bool   [default = False]
            If `True`, the label will be translated to the full quantum mechanical description
            of the state.
        """
        if hasattr(levels, '__iter__'):
            for fineline in fine_structure_complexes[line_tag]:
                ion = fineline.split('_')[0]
                if ion[-1] in levels:
                    self.add_line(fineline, self.velspan)

        elif levels is None:
            for fineline in fine_structure_complexes[line_tag]:
                self.add_line(fineline, self.velspan)

        else:
            for fineline in fine_structure_complexes[line_tag]:
                ion = fineline.split('_')[0]
                if ion[-1] in levels:
                    self.add_line(fineline, self.velspan)

        # Set label:
        regions_of_line = self.find_line(line_tag)
        for reg in regions_of_line:
            if full_label:
                reg.label = line_complexes.CI_full_labels[line_tag]
            else:
                reg.label = line_complexes.CI_labels[line_tag]

    def remove_fine_lines(self, line_tag):
        """
        Remove all lines associated to a given fine-structure complex.

        Parameters
        ----------
        line_tag : str
            The line tag of the ground state transition to remove.
        """
        for fineline in fine_structure_complexes[line_tag]:
            if fineline in self.all_lines:
                self.remove_line(line_tag)

    def add_molecule(self, molecule, band, J=0, velspan=None, full_label=False):
        """
        Add molecular lines for a given band, e.g., "AX(0-0)" of CO.

        Parameters
        ----------
        molecule : str
            The molecular identifier, e.g., 'CO', 'H2'

        band : str
            The vibrational band of the molecule, e.g., for CO: "AX(0-0)"
            These bands are defined in the `line_complexes`.

        J : int   [default = 0]
            The maximal rotational level to include. All levels up to and including `J`
            will be included.

        velspan : float   [default = None]
            The velocity span around the line center, which will be included in the fit.
            If `None` is given, use the default :attr:`velspan <dataset.DataSet.velspan>`
            defined (500 km/s).

        full_label : bool   [default = False]
            If `True`, the label will be translated to the full quantum mechanical description
            of the state.
        """
        if molecule == 'CO':
            nu_level = line_complexes.CO[band]
            for transitions in nu_level[:J+1]:
                self.add_many_lines(transitions, velspan=velspan)

            ref_J0 = line_complexes.CO[band][0][0]
            regions_of_line = self.find_line(ref_J0)
            for region in regions_of_line:
                if full_label:
                    label = line_complexes.CO_full_labels[band]
                    region.label = label

                else:
                    region.label = "${\\rm CO\ %s}$" % band

        if molecule in self.molecules.keys():
            self.molecules[molecule].append(band)
        else:
            self.molecules[molecule] = [band]

    def remove_molecule(self, molecule, band):
        """Remove all lines for the given band of the given molecule."""
        if molecule == 'CO':
            if band not in self.molecules['CO']:
                if self.verbose:
                    print "\n [WARNING] - The %s band for %s is not defined!" % (band, molecule)
                return None

            nu_level = line_complexes.CO[band]
            for transitions in nu_level:
                for line_tag in transitions:
                    if line_tag in self.all_lines:
                        self.remove_line(line_tag)

            self.molecules['CO'].remove(band)
            if len(self.molecules['CO']) == 0:
                self.molecules.pop('CO')

    def deactivate_molecule(self, molecule, band):
        """
        Deactivate all lines for the given band of the given molecule.
        To see the available molecular bands defined, see the manual pdf or ``line_complexes.py``.
        """
        if molecule == 'CO':
            if band not in self.molecules['CO']:
                if self.verbose:
                    print "\n [WARNING] - The %s band for %s is not defined!" % (band, molecule)
                return None

            nu_level = line_complexes.CO[band]
            for transitions in nu_level:
                for line_tag in transitions:
                    if line_tag in self.all_lines:
                        self.deactivate_line(line_tag)

    def activate_molecule(self, molecule, band):
        """
        Activate all lines for the given band of the given molecule.
        Example: activate_molecule('CO', 'AX(0-0)')
        """
        if molecule == 'CO':
            if band not in self.molecules['CO']:
                if self.verbose:
                    print "\n [WARNING] - The %s band for %s is not defined!" % (band, molecule)
                return None

            nu_level = line_complexes.CO[band]
            for transitions in nu_level:
                for line_tag in transitions:
                    if line_tag in self.all_lines:
                        self.activate_line(line_tag)

    def prepare_dataset(self, norm=True, mask=True, verbose=True, active_only=False):
        """
        Prepare the data for fitting. This function sets up the parameter structure,
        and handles the normalization and masking of fitting regions.

        Parameters
        ----------
        norm : bool   [default = True]
            Opens an interactive window to let the user normalize each region
            using the defined :attr:`norm_method <dataset.DataSet.norm_method>`.

        mask : bool   [default = True]
            Opens an interactive window to let the user mask each fitting region.

        verbose : bool   [default = True]
            If this is set, the code will print small info statements during the run.

        Returns
        -------
        bool
            The function returns `True` when the dataset has passed all the steps.
            If one step fails, the function returns `False`.
            The :attr:`ready2fit <dataset.DataSet.ready2fit>` attribute of the dataset is also
            updated accordingly.

        """

        plt.close('all')
        # --- Normalize fitting regions manually, or use polynomial fitting
        if norm:
            for region in self.regions:
                if not region.normalized:
                    go_on = 0
                    while go_on == 0:
                        go_on = region.normalize(norm_method=self.norm_method)
                        # region.normalize returns 1 when continuum is fitted

            if verbose and self.verbose:
                print ""
                print " [DONE] - Continuum fitting successfully finished."
                print ""

        # --- Prepare fit parameters  [class: lmfit.Parameters]
        self.pars = Parameters()
        # - First setup parameters with values only:
        for ion in self.components.keys():
            for n, comp in enumerate(self.components[ion]):
                ion = ion.replace('*', 'x')
                z, b, logN, opts = comp
                z_name = 'z%i_%s' % (n, ion)
                b_name = 'b%i_%s' % (n, ion)
                N_name = 'logN%i_%s' % (n, ion)

                self.pars.add(z_name, value=myfloat(z), vary=opts['var_z'])
                self.pars.add(b_name, value=myfloat(b), vary=opts['var_b'], min=0., max=800.)
                self.pars.add(N_name, value=myfloat(logN), vary=opts['var_N'], min=0., max=40.)

        # - Then setup parameter links:
        for ion in self.components.keys():
            for n, comp in enumerate(self.components[ion]):
                ion = ion.replace('*', 'x')
                z, b, logN, opts = comp
                z_name = 'z%i_%s' % (n, ion)
                b_name = 'b%i_%s' % (n, ion)
                N_name = 'logN%i_%s' % (n, ion)

                if opts['tie_z']:
                    self.pars[z_name].expr = opts['tie_z']
                if opts['tie_b']:
                    self.pars[b_name].expr = opts['tie_b']
                if opts['tie_N']:
                    self.pars[N_name].expr = opts['tie_N']

        # Setup Chebyshev parameters:
        if self.cheb_order >= 0:
            for reg_num, reg in enumerate(self.regions):
                p0 = np.median(reg.flux)
                for cheb_num in range(self.cheb_order+1):
                    if cheb_num == 0:
                        self.pars.add('R%i_cheb_p%i' % (reg_num, cheb_num), value=p0)
                    else:
                        self.pars.add('R%i_cheb_p%i' % (reg_num, cheb_num), value=0.0)

        # --- mask spectral regions that should not be fitted
        if mask:
            for region in self.regions:
                # if region.new_mask:
                if active_only and region.has_active_lines() and region.new_mask:
                    # region.define_mask()
                    region.define_mask(z=self.redshift, dataset=self)
            if verbose and self.verbose:
                print ""
                print " [DONE] - Spectral masks successfully created."
                print ""

        self.ready2fit = True

        # --- Check that all active elements have components defined:
        for line_tag in self.all_lines:
            ion = line_tag.split('_')[0]
            line = self.lines[line_tag]
            if ion not in self.components.keys() and line.active:
                if self.verbose:
                    print ""
                    print " [ERROR] - Components are not defined for element: "+ion
                    print ""
                self.ready2fit = False

                return False

        # --- Check that no components for inactive elements are defined:
        for this_ion in self.components.keys():
            lines_for_this_ion = list()
            for region in self.regions:
                for line in region.lines:
                    if line.ion == this_ion:
                        lines_for_this_ion.append(line.active)

            if np.any(lines_for_this_ion):
                pass
            else:
                if self.verbose:
                    print "\n [WARNING] - Components defined for inactive element: %s\n" % this_ion

        if self.ready2fit:
            if verbose and self.verbose:
                print "\n  Dataset is ready to be fitted."
                print ""
            return True

    def fit(self, rebin=1, verbose=True, plot=False, **kwargs):
        """
        Fit the absorption lines using chi-square minimization.

        Parameters
        ----------
        rebin : int   [default = 1]
            Rebin data by a factor *rebin* before fitting.

        verbose : bool   [default = True]
            This will print the fit results to terminal.

        plot : bool   [default = False]
            This will make the best-fit solution show up in a new window.

        **kwargs
            Keyword arguments are derived from the `scipy.optimize`_ minimization methods.
            The default method is leastsq_, used in `lmfit <https://lmfit.github.io/lmfit-py/>`_.
            This can be changed using the `method` keyword.
            See documentation in lmfit_ and scipy.optimize_.

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
                print " [Error]  - Dataset is not ready to be fitted."
                print "            Run `.prepare_dataset()` before fitting."
            return False

        if rebin > 1:
            print "\n  Rebinning the data by a factor of %i \n" % rebin

        print "  Fit is running... Please, be patient.\n"
        # npad = options['npad']

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

                    res = region.res

                    # Define flexible subsampling:
                    dv_pix = calculate_velocity_bin_size(x)
                    # Generate line profile
                    profile_obs = evaluate_profile(x, pars, self.redshift,
                                                   # region.lines, self.components,
                                                   self.lines.values(), self.components,
                                                   res, dv=dv_pix/3.)

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

        minimizer = Minimizer(chi, self.pars, nan_policy='omit')
        popt = minimizer.minimize(**kwargs)
        self.best_fit = popt.params

        if self.cheb_order >= 0:
            # Normalize region data with best-fit polynomial:
            for reg_num, region in enumerate(self.regions):
                x, y, err, mask = region.unpack()
                cont_model = evaluate_continuum(x, self.best_fit, reg_num)
                region.flux /= cont_model
                region.err /= cont_model
                region.normalized = True

        print "\n The fit has finished with the following exit message:"
        print "  " + popt.message
        print ""
        if verbose and self.verbose:
            output.print_results(self, self.best_fit, velocity=False)
            if self.cheb_order >= 0:
                output.print_cont_parameters(self)

        if plot:
            self.plot_fit(rebin=rebin, subsample_profile=rebin)

        chi2 = popt.chisqr
        return popt, chi2

    def plot_fit(self, linestyles=['--'], colors=['RoyalBlue'],
                 rebin=1, fontsize=12, xmin=None, xmax=None, max_rows=4,
                 filename=None, show=True, subsample_profile=1, npad=50,
                 highlight=[], residuals=True, norm_resid=False):
        """
        Plot *all* the absorption lines and the best-fit profiles.
        For details, see :func:`output.plot_all_lines`.
        """
        output.plot_all_lines(self, plot_fit=True, linestyles=linestyles,
                              colors=colors, rebin=rebin, fontsize=fontsize,
                              xmin=xmin, xmax=xmax, max_rows=max_rows,
                              filename=filename, show=show,
                              subsample_profile=subsample_profile, npad=npad,
                              highlight=highlight, residuals=residuals,
                              norm_resid=norm_resid)
        plt.show()

    def velocity_plot(self, **kwargs):
        """
        Create a velocity plot, showing all the fitting regions defined, in order to compare
        different lines and to identify blends and contamination.
        """
        output.velocity_plot(self, **kwargs)

    def plot_line(self, line_tag, plot_fit=False, linestyles=['--'], colors=['RoyalBlue'],
                  loc='left', rebin=1, nolabels=False, axis=None, fontsize=12,
                  xmin=None, xmax=None, ymin=None, show=True, subsample_profile=1,
                  npad=50, highlight=[], residuals=True):
        """
        Plot a single fitting :class:`Region <regions.Region>`
        containing the line corresponding to the given `line_tag`.
        For details, see :func:`output.plot_single_line`.
        """
        output.plot_single_line(self, line_tag, plot_fit=plot_fit,
                                linestyles=linestyles, colors=colors,
                                loc=loc, rebin=rebin, nolabels=nolabels,
                                axis=axis, fontsize=fontsize,
                                xmin=xmin, xmax=xmax, ymin=ymin, show=show,
                                subsample_profile=subsample_profile, npad=npad,
                                highlight=highlight, residuals=residuals)

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
        """Print the total column densities for each element relative to HI in Solar units."""
        output.print_metallicity(self, self.best_fit, logNHI, err)

    def print_abundance(self):
        """Print the total column densities of all components."""
        output.print_abundance(self)

    def save_fit_regions(self, filename=None, individual=False, path=''):
        """
        Save the fitting regions to ASCII table output.
        The format is as follows:
        (wavelength , normalized flux , normalized error , best-fit profile , mask)

        Parameters
        ----------
        filename : str   [default = None]
            Filename for the fitting regions.
            If `None`, the :attr:`name <dataset.DataSet.name>` attribute will be used.

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
                print " [ERROR] - Must specify dataset.name [dataset.set_name('name')]"
                print "           or give filename [dataset.save(filename='filename')]"
        output.save_fit_regions(self, filename, individual=individual, path=path)

    def save(self, filename=None, verbose=False):
        """Save the DataSet to file using the HDF5 format."""
        if not filename:
            if self.name:
                filename = self.name
            else:
                print " [ERROR] - Must specify dataset.name [dataset.set_name('name')]"
                print "           or give filename [dataset.save(filename='filename')]"
        hdf5_save.save_hdf_dataset(self, filename, verbose=verbose)
