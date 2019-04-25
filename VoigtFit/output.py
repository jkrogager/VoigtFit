# -*- coding: UTF-8 -*-

__author__ = 'Jens-Kristian Krogager'

import os
from os.path import splitext
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import fftconvolve, gaussian
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import RectBivariateSpline as spline2d
import numpy as np

import voigt
import Asplund
import molecules
from dataset import Line
from voigt import evaluate_profile
import terminal_attributes as term

plt.rcParams['lines.linewidth'] = 1.0


valid_kwargs = Line2D.properties(Line2D([0], [0])).keys()
valid_kwargs += ['ymin', 'ymax', 'ls', 'lw']

default_comp = {'color': 'b', 'ls': '-',
                'alpha': 1.0, 'lw': 1.0,
                'ymin': 0.87, 'ymax': 0.92,
                'text': None, 'loc': 'above'}
default_highlight_comp = {'color': 'DarkOrange', 'ls': '-',
                          'alpha': 0.7, 'lw': 2.0,
                          'ymin': 0.85, 'ymax': 1.0,
                          'text': None, 'loc': 'above'}

default_line = {'color': 'r', 'ls': '-',
                'lw': 1.0, 'alpha': 1.0}
default_hl_line = {'color': 'orange', 'ls': '--',
                   'lw': 1.0, 'alpha': 1.0}


def load_lsf(lsf_fname, wl, nsub=1):
    """
    Load a Line-Spread Function table following format from HST:
    First line gives wavelength in Angstrom and the column below
    each given wavelength defines the kernel in pixel space:

    | wl1    wl2    wl3   ...  wlN
    | lsf11  lsf21  lsf31 ...  lsfN1
    | lsf12  lsf22  lsf32 ...  lsfN2
    | :
    | :
    | lsf1M  lsf2M  lsf3M ...  lsfNM

    Parameters
    ----------
    lsf_fname : string
        The filename containing the LSF data.

    wl : array like, shape (N)
        The wavelength grid onto which the LSF will be evaluated

    nsub : integer  [default = 1]
        Kernel subsampling factor relative to the data.
        This is only used if the resolution is given as a LSF file.

    Returns
    -------
    kernel : np.array, shape(N, M)
        A grid of interpolated LSF evaluated at each given input wavelength
        of the array `wl` of shape N, where M is the number of pixels in the LSF.

    Notes
    -----
    The output kernel is transposed with respect to the input format
    for ease of computation in the convolution since indexing is faster
    along rows than columns.

    """
    if nsub > 1:
        wl = np.linspace(wl.min(), wl.max(), nsub*len(wl))

    lsf_tab = np.loadtxt(lsf_fname)
    # Get the wavelength array from the first line in the file:
    lsf_wl = lsf_tab[0]

    # The LSF data is the resulting table excluding the first line:
    lsf = lsf_tab[1:, :]

    # Make an array of pixel indeces:
    lsf_pix = np.arange(lsf.shape[0])

    # Linearly interpolate the LSF grid:
    LSF = spline2d(lsf_pix, lsf_wl, lsf, kx=1, ky=1)
    kernel = LSF(lsf_pix, wl).T
    return kernel


def merge_two_dicts(default, x):
    """Merge the keys of dictionary `x` into dictionary `default`. """
    z = default.copy()
    z.update(x)
    return z


class CompProp(object):
    def __init__(self, list_of_ions, prop=None):
        self.properties = dict()
        if prop:
            prop = merge_two_dicts(default_comp, prop)
            for ion in list_of_ions:
                self.properties[ion] = prop
        else:
            for ion in list_of_ions:
                self.properties[ion] = default_comp.copy()

    def set_properties(self, ion, prop):
        """Set properties of `ion` from a dictionary."""
        if ion not in self.properties.keys():
            self.properties[ion] = default_comp.copy()

        for key, val in prop.items():
            self.properties[ion][key] = val

    def set_value(self, ion, key, value):
        """Set single value for the property `key` of the given `ion`."""
        self.properties[ion][key] = value

    def get_value(self, ion, key):
        """Set single value for the property `key` of the given `ion`."""
        return self.properties[ion][key]

    def get_line_props(self, ion):
        """Return only properties appropriate for matplotlib.axvline"""
        vline_props = self.properties[ion].copy()
        for key in vline_props.keys():
            if key not in valid_kwargs:
                vline_props.pop(key)
        return vline_props

    def get_text_props(self, ion):
        """Return properties related to the text label"""
        label = self.properties[ion]['text']
        if self.properties[ion]['loc'].lower() == 'top':
            pos = self.properties[ion]['ymax']+0.02
            text_align = 'bottom'
        elif self.properties[ion]['loc'].lower() == 'bottom':
            pos = self.properties[ion]['ymin']-0.02
            text_align = 'top'
        else:
            pos = self.properties[ion]['ymax']+0.02
            text_align = 'bottom'

        return label, pos, text_align


def mad(x):
    """Calculate Median Absolute Deviation"""
    return np.median(np.abs(x - np.median(x)))


def chunks(l, n):
    """Yield successive `n`-sized chunks from `l`."""
    for i in xrange(0, len(l), n):
        yield l[i:i+n]


def rebin_spectrum(wl, spec, err, n, method='mean'):
    """
    Rebin input spectrum by a factor of `n`.
    Method is either *mean* or *median*, default is mean.

    Parameters
    ----------
    wl : array_like, shape (N)
        Input wavelength array.

    spec : array_like, shape (N)
        Input flux array.

    err : array_like, shape (N)
        Input error array.

    n : int
        Integer rebinning factor.

    method : str   [default = 'mean']
        Rebin method, either 'mean' or 'median'.

    Returns
    -------
    wl_r : array_like, shape (M)
        Rebinned wavelength array, the new shape will be N/n.

    spec_r : array_like, shape (M)
        Rebinned flux array, the new shape will be N/n.

    err_r : array_like, shape (M)
        Rebinned error array, the new shape will be N/n.

    """
    if method.lower() == 'mean':
        combine = np.mean
    elif method.lower() == 'median':
        combine = np.median
    else:
        combine = np.mean
    spec_chunks = list(chunks(spec, n))
    err_chunks = list(chunks(err, n))
    wl_chunks = list(chunks(wl, n))

    spec_r = np.zeros(len(spec_chunks))
    err_r = np.zeros(len(spec_chunks))
    wl_r = np.zeros(len(spec_chunks))
    for num in range(len(spec_chunks)):
        spec_r[num] = combine(spec_chunks[num])
        err_r[num] = np.sqrt(np.mean(err_chunks[num]**2)/n)
        wl_r[num] = combine(wl_chunks[num])

    return wl_r, spec_r, err_r


def rebin_bool_array(x, n):
    """Rebin input boolean array `x` by an integer factor of `n`."""

    array_chunks = list(chunks(x, n))

    x_r = np.zeros(len(array_chunks))
    for num in range(len(array_chunks)):
        x_r[num] = np.prod(array_chunks[num], dtype=bool)

    return np.array(x_r, dtype=bool)


def create_blank_input():
    """Create a blank template input parameter file."""
    # Read file from static
    root_path = os.path.dirname(os.path.abspath(__file__))
    temp_filename = os.path.join(root_path, 'static/input_template.txt')
    with open(temp_filename) as template:
        parameter_lines = template.readlines()

    # Save it to 'vfit.pars'
    with open('vfit.pars', 'w') as output_file:
        output_file.write(''.join(parameter_lines))


# ===================================================================================
#
#   Graphics output functions:
# ------------------------------
#  A4 figuse size:
#  fig = plt.figure(figsize=(7, 9.899))


def plot_all_lines(dataset, plot_fit=True, rebin=1, fontsize=12, xmin=None,
                   xmax=None, ymin=None, ymax=None, max_rows=4, filename=None,
                   subsample_profile=1, npad=50, residuals=True,
                   norm_resid=False, legend=True, loc='left', show=True,
                   default_props={}, element_props={}, highlight_props=None,
                   label_all_ions=False, xunit='vel',
                   line_props=None, hl_line_props=None):
    """
    Plot all active absorption lines. This function is a wrapper of
    :func:`plot_single_line`. For a complete description of input parameters,
    see the documentation for :func:`plot_single_line`.

    Parameters
    ----------
    dataset : :class:`VoigtFit.DataSet`
        Instance of the :class:`VoigtFit.DataSet` class containing the line
        regions to plot.

    max_rows : int   [default = 4]
        The maximum number of rows of figures.
        Each row consists of two figure panels.

    filename : str
        If a filename is given, the figures are saved to a pdf file.

    """
    if 'velocity'.find(xunit) == 0:
        # X-axis units should be velocity.
        xunit = 'vel'
    elif 'wavelength'.find(xunit) == 0:
        xunit = 'wl'
    elif xunit.lower() == 'wl':
        xunit = 'wl'
    else:
        xunit = 'vel'

    molecule_warning = """
        H2 Molecules are defined in the dataset. These will be skipped
        in the normal DataSet.plot_fit() function.
        A figure can be generated using \033[1mDataSet.plot_molecule()\033[0m.
    """
    show_molecule_warning = False

    # --- First figure out which lines to plot to avoid overlap
    #     of several lines defined in the same region.
    included_lines = list()
    lines_to_plot = list()
    for ref_line in dataset.lines.values():
        # If the line is not active, skip this line:
        if not ref_line.active:
            continue

        if ref_line.tag in included_lines:
            pass

        elif 'H2' in ref_line.tag:
            show_molecule_warning = True
            pass

        elif ref_line.ion[-1].islower():
            # Check if the gorund state is defined in same region.
            regions_of_line = dataset.find_line(ref_line.tag)
            ground_state = ref_line.ion[:-1]
            reg = regions_of_line[0]
            ions_in_region = [line.ion for line in reg.lines]
            if ground_state in ions_in_region:
                included_lines.append(ref_line.tag)

        else:
            regions_of_line = dataset.find_line(ref_line.tag)
            for region in regions_of_line:
                lines_to_plot.append(ref_line.tag)

            if len(region.lines) == 1:
                included_lines.append(ref_line.tag)

            else:
                l_ref = ref_line.l0*(dataset.redshift + 1)
                for line in region.lines:
                    l0 = line.l0
                    delta_lam = (l0*(dataset.redshift + 1) - l_ref)
                    delta_v = delta_lam / l_ref * 299792.458
                    if np.abs(delta_v) <= 150 or line.ion[-1].islower():
                        included_lines.append(line.tag)

    # --- Pack keyword arguments for plot_single_line:
    plot_line_kwargs = dict(plot_fit=plot_fit, rebin=rebin,
                            loc=loc, nolabels=True,
                            show=False, xmin=xmin, xmax=xmax,
                            ymin=ymin, ymax=ymax,
                            fontsize=fontsize,
                            subsample_profile=subsample_profile,
                            npad=npad, residuals=residuals,
                            norm_resid=norm_resid,
                            legend=legend, xunit=xunit,
                            label_all_ions=label_all_ions,
                            default_props=default_props,
                            element_props=element_props,
                            highlight_props=highlight_props,
                            line_props=line_props, hl_line_props=hl_line_props)

    # --- If *filename* is given, set up a PDF container for saving to file:
    if filename:
        if '.pdf' not in filename:
            filename += '.pdf'
        pdf = matplotlib.backends.backend_pdf.PdfPages(filename)

    # -- Create container for multiple regions of same line:
    regionidx_of_line = {}

    # --- Determine number of pages to create:
    pages = list(chunks(sorted(lines_to_plot), 2*max_rows))
    # lines_in_figure = list()
    for contents in pages:
        lines_in_figure = list()
        # --- Determine figure size:
        if len(contents) > 1:
            width = 8.5
            columns = 2
        else:
            width = 8.5
            columns = 2

        heigth = (len(contents) + 2) / 2 * 7.5/(max_rows)
        rows = (len(contents) + 1) / 2
        if len(contents) == 1:
            heigth = 6
            columns = 1
            rows = 1
        elif len(contents) == 2:
            heigth = 3
            columns = 2
            rows = 1

        fig = plt.figure(figsize=(width, heigth))
        fig.subplots_adjust(left=0.10, right=0.98, top=0.98,
                            hspace=0.03, bottom=0.14)

        if len(contents) % 2 == 1 and len(contents) > 1:
            add_on = 1
        else:
            add_on = 0
        num = 1 + add_on
        for line_tag in contents:
            if line_tag in lines_in_figure:
                pass
            else:
                num_regions = len(dataset.find_line(line_tag))
                if line_tag not in regionidx_of_line.keys():
                    regionidx_of_line[line_tag] = list()
                for idx in range(num_regions):
                    if idx in regionidx_of_line[line_tag]:
                        # skip this region idx
                        pass
                    elif num > rows*columns:
                        break
                    else:
                        ax = fig.add_subplot(rows, columns, num)
                        _, LIV = plot_single_line(dataset, line_tag, index=idx,
                                                  axis=ax, **plot_line_kwargs
                                                  )
                        regionidx_of_line[line_tag].append(idx)
                        num += 1

                lines_in_figure += LIV
                ax.tick_params(length=7, labelsize=fontsize)
                if num <= len(contents) - (1-add_on):
                    # xtl = ax.get_xticklabels()
                    # print [ticklabel.get_text() for ticklabel in xtl]
                    # ax.set_xticklabels([''])
                    pass
                else:
                    if xunit == 'wl':
                        ax.set_xlabel("${\\rm Wavelength\ \ (\\AA)}$",
                                      fontsize=12)
                    else:
                        ax.set_xlabel("${\\rm Rel. velocity\ \ (km\ s^{-1})}$",
                                      fontsize=12)

                if num % 2 == 0:
                    ax.set_ylabel("Normalized Flux", fontsize=12)
                # num += 1
                # LIV is a shorthand for 'lines_in_view'

        if filename:
            pdf.savefig(fig)

    if len(pages) > 0:
        fig.set_tight_layout(True)
        if filename:
            pdf.close()
            print("\n  Output saved to PDF file:  " + filename)

        if show:
            plt.show()

    if show_molecule_warning:
        print(molecule_warning)


def plot_single_line(dataset, line_tag, index=0, plot_fit=False,
                     loc='left', rebin=1, nolabels=False, axis=None,
                     fontsize=12, subsample_profile=1,
                     xmin=None, xmax=None, ymin=None, ymax=None,
                     show=True, npad=50, legend=True,
                     residuals=False, norm_resid=False,
                     default_props={}, element_props={},
                     highlight_props=None,
                     label_all_ions=False, xunit='velocity',
                     line_props=None, hl_line_props=None,
                     sort_f=True):
    """
    Plot a single absorption line.

    Parameters
    ----------
    dataset : :class:`VoigtFit.DataSet`
        Instance of the :class:`VoigtFit.DataSet` class containing the lines

    line_tag : str
        The line tag of the line to show, e.g., 'FeII_2374'

    index : int   [default = 0]
        The line index. When fitting the same line in multiple
        spectra this indexed points to the index of the given
        region to be plotted.

    plot_fit : bool   [default = False]
        If `True`, the best-fit profile will be shown

    loc : str   [default = 'left']
        Places the line tag (right or left).

    rebin : int   [default = 1]
        Rebinning factor for the spectrum

    nolabels : bool   [default = False]
        If `True`, show the axis x- and y-labels.

    axis : `matplotlib.axes.Axes <https://matplotlib.org/api/axes_api.html>`_
        The plotting axis of matplotlib.
        If `None` is given, a new figure and axis will be created.

    fontsize : int   [default = 12]
        The fontsize of text labels.

    xmin : float
        The lower x-limit in relative velocity (km/s).
        If nothing is given, the extent of the region is used.

    xmax : float
        The upper x-limit in relative velocity (km/s).
        If nothing is given, the extent of the region is used.

    ymin : float   [default = None]
        The lower y-limit in normalized flux units.
        Default is determined from the data.

    ymax : float   [default = None]
        The upper y-limit in normalized flux units.
        Default is determined from the data.

    show : bool   [default = True]
        Show the figure.

    subsample_profile : int   [default = 1]
        Subsampling factor to calculate the profile on a finer grid than
        the data sampling.
        By default the profile is evaluated on the same grid as the data.

    npad : int   [default = 50]
        Padding added to the synthetic profile before convolution.
        This removes end artefacts from the `FFT` routine.

    residuals : bool   [default = False]
        Add a panel above the absorption line to show the residuals of the fit.

    norm_resid : bool   [default = False]
        Show normalized residuals.

    legend : bool   [default = True]
        Show line labels as axis legend.

    default_props : dict
        Dictionary of transition tick marker properties. The dictionary is
        passed to matplotlib.axes.Axes.axvline to control color, linewidth,
        linestyle, etc.. Two additional keywords can be defined:
        The keyword `text` is a string that will be printed above or below
        each tick mark for each element.
        The keyword `loc` controls the placement of the tick mark text
        for the transistions, and must be one either 'above' or 'below'.

    element_props : dict
        Dictionary of properties for individual elements.
        Each element defines a dictionary with individual properties following
        the format for `default_props`.

            Ex: ``element_props={'SiII': {'color': 'red', 'lw': 1.5},``
                ``'FeII': {'ls': '--', 'alpha': 0.2}}``

        This will set the color and linewidth of the tick marks
        of SiII transitions and the linestyle and alpha-parameter
        of FeII transitions.

    highlight_props : dict/list   [default = None]
        A dictionary of `ions` (e.g., "FeII", "CIa", etc.) used to calculate
        a separate profile for this subset of ions. Each `ion` as a keyword
        must specify a dictionary which can change individual properties for
        the given `ion`. Similar to `element_props`.
        If an empty dictionary is given, the default parameters will be used.
        Alternatively, a list of `ions` can be given to use default properties
        for all `ions`.

            Ex: ``highlight_props={'SiII':{}, 'FeII':{'color': 'blue'}}``

        This will highlight SiII transitions with default highlight
        properties, and FeII transistions with a user specified color.

            Ex: ``highlight_props=['SiII', 'FeII']``

        This will highlight SiII and FeII transitions using default
        highlight properties.

    label_all_ions : bool   [default = False]
        Show labels for all `ions` defined.
        The labels will appear above the component tick marks.

    xunit : string   [default = 'velocity']
        The unit of the x-axis, must be either 'velocity' or 'wavelength'.
        Shortenings are acceptable too, e.g., 'vel'/'v' or 'wave'/'wl'.

    line_props : dict   [default = None]
        A dictionary of keywords to change the default line properties
        of the best-fit profile; e.g., 'color', 'lw', 'linestyle'.
        All keywords will be passed to the `plot function`_ of matplotlib.

    hl_line_props : dict   [default = None]
        A dictionary of keywords to change the default line properties
        of the best-fit profile for highlighted ions.
        All keywords will be passed to the `plot function`_ of matplotlib.

    sort_f : bool   [default = True]
        If `True`, calculate velocities with respect to the line with the
        largest oscillator strength. Otherwise, use the given `line_tag`
        as reference.


    .. _plot function: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html

    """

    if line_props is None:
        line_props = default_line
    else:
        line_props = merge_two_dicts(default_line, line_props)

    if hl_line_props is None:
        hl_line_props = default_hl_line
    else:
        hl_line_props = merge_two_dicts(default_hl_line, hl_line_props)

    # Convert highlight_props to dictionary if a list of `ions` is given:
    if isinstance(highlight_props, list):
        highlight_props = {key: {} for key in highlight_props}
    elif highlight_props is None:
        highlight_props = {}

    # Setup the line properties:
    ions_in_dataset = list()
    for line in dataset.lines.values():
        if line.ion not in ions_in_dataset:
            ions_in_dataset.append(line.ion)

    comp_props = CompProp(ions_in_dataset, default_props)
    hl_comp_props = CompProp(ions_in_dataset)
    for this_ion, these_props in element_props.items():
        comp_props.set_properties(this_ion, these_props)

    for this_ion, these_props in highlight_props.items():
        hl_comp_props.set_properties(this_ion, these_props)

    if line_tag not in dataset.all_lines:
        dataset.add_line(line_tag, active=False)
        dataset.prepare_dataset()

    if label_all_ions:
        for ion in ions_in_dataset:
            if comp_props.get_value(ion, 'text') is None:
                comp_props.set_value(ion, 'text', ion)

    if 'velocity'.find(xunit) == 0:
        # X-axis units should be velocity.
        xunit = 'vel'
    elif 'wavelength'.find(xunit) == 0:
        xunit = 'wl'
    elif xunit.lower() == 'wl':
        xunit = 'wl'
    else:
        xunit = 'vel'

    regions_of_line = dataset.find_line(line_tag)
    region = regions_of_line[index]

    x, y, err, mask = region.unpack()
    cont_err = region.cont_err
    kernel = region.kernel
    kernel_nsub = region.kernel_nsub
    x_orig = x.copy()

    if rebin > 1:
        x, y, err = rebin_spectrum(x, y, err, rebin)
        mask = rebin_bool_array(mask, rebin)

    ref_line = dataset.lines[line_tag]
    l0_ref, f_ref, _ = ref_line.get_properties()
    l_ref = l0_ref*(dataset.redshift + 1)

    # - Check if lines are separated by more than 200 km/s
    #   if so, then remove the line from the view.
    lines_in_view = list()
    for line in region.lines:
        l0 = line.l0
        if line.f > f_ref and sort_f:
            l0_ref = line.l0
            l_ref = l0_ref*(dataset.redshift + 1)
            f_ref = line.f
            ref_line = line
        delta_v = (l0*(dataset.redshift + 1) - l_ref) / l_ref * 299792.
        if np.abs(delta_v) <= 150 or line.ion[-1].islower() is True:
            lines_in_view.append(line.tag)

    if axis:
        ax = axis
    else:
        # plt.close('all')
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        fig.subplots_adjust(bottom=0.15, right=0.97, top=0.98)

    if not xmin:
        if xunit == 'vel':
            xmin = -region.velspan
        else:
            xmin = x.min()
    if not xmax:
        if xunit == 'vel':
            xmax = region.velspan
        else:
            xmax = x.max()
    ax.set_xlim(xmin, xmax)
    if np.abs(xmin) > 900 or np.abs(xmax) > 900:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    if plot_fit and (isinstance(dataset.best_fit, dict) or
                     isinstance(dataset.pars, dict)):

        if residuals:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('top', size='20%', pad=0., sharex=ax)

        if isinstance(kernel, float):
            N_pix = len(x)*2
            dx = np.diff(x)[0]
            wl_line = np.logspace(np.log10(x.min() - 50*dx), np.log10(x.max() + 50*dx), N_pix)
            pxs = np.diff(wl_line)[0] / wl_line[0] * 299792.458
        elif isinstance(kernel, np.ndarray):
            assert kernel.shape[0] == len(x_orig) * kernel_nsub
            if kernel_nsub > 1:
                N = kernel_nsub * len(x)
                wl_line = np.linspace(x.min(), x.max(), N)
            else:
                wl_line = x_orig
        else:
            err_msg = "Invalid type of `kernel`: %r" % type(kernel)
            raise TypeError(err_msg)

        # ref_line = dataset.lines[line_tag]
        # l0, f, gam = ref_line.get_properties()
        # l_ref = l0*(dataset.redshift + 1)

        tau = np.zeros_like(wl_line)
        tau_hl = np.zeros_like(wl_line)
        N_highlight = 0

        if isinstance(dataset.best_fit, dict):
            params = dataset.best_fit
        else:
            params = dataset.pars

        for line in dataset.lines.values():
            if line.active:
                # Get transition mark properties for this element
                component_prop = comp_props.get_line_props(line.ion)
                (comp_text,
                 comp_y_loc,
                 loc_string) = comp_props.get_text_props(line.ion)
                hl_component_prop = hl_comp_props.get_line_props(line.ion)

                # Load line properties
                l0, f, gam = line.get_properties()
                ion = line.ion
                n_comp = len(dataset.components[ion])
                l_center = l0*(dataset.redshift + 1.)
                vel = (wl_line - l_center)/l_center*299792.458
                ion = ion.replace('*', 'x')

                for n in range(n_comp):
                    z = params['z%i_%s' % (n, ion)].value
                    b = params['b%i_%s' % (n, ion)].value
                    logN = params['logN%i_%s' % (n, ion)].value
                    tau += voigt.Voigt(wl_line, l0, f,
                                       10**logN, 1.e5*b, gam, z=z)
                    if ion in highlight_props.keys():
                        tau_hl += voigt.Voigt(wl_line, l0, f,
                                              10**logN, 1.e5*b, gam, z=z)
                        if xunit == 'vel':
                            ax.axvline((l0*(z+1) - l_ref)/l_ref*299792.458,
                                       **hl_component_prop)
                        else:
                            ax.axvline(l0*(z+1), **hl_component_prop)
                        N_highlight += 1

                    if xunit == 'vel':
                        comp_x_loc = (l0*(z+1) - l_ref)/l_ref*299792.458
                    else:
                        comp_x_loc = l0*(z+1)

                    ax.axvline(comp_x_loc, **component_prop)
                    if comp_text:
                        ax.text((comp_x_loc - xmin)/(xmax-xmin), comp_y_loc,
                                comp_text, transform=ax.transAxes,
                                ha='center', va=loc_string, clip_on=True)

        profile_int = np.exp(-tau)
        profile_int_hl = np.exp(-tau_hl)
        if isinstance(kernel, float):
            sigma_instrumental = kernel / 2.35482 / pxs
            LSF = gaussian(len(wl_line)/2, sigma_instrumental)
            LSF = LSF/LSF.sum()
            profile = fftconvolve(profile_int, LSF, 'same')
            profile_hl = fftconvolve(profile_int_hl, LSF, 'same')
            # profile = profile_broad[50:-50]
            # profile_hl = profile_broad_hl[50:-50]
            # wl_line = wl_line[50:-50]
        else:
            profile = voigt.convolve_numba(profile_int, kernel)
        vel_line = (wl_line - l_ref)/l_ref*299792.458

    vel = (x - l_ref) / l_ref * 299792.458

    if residuals and plot_fit:
        cax.set_xlim(xmin, xmax)

    if xunit == 'vel':
        view_part = (vel > xmin) * (vel < xmax)
    else:
        view_part = (x > xmin) * (x < xmax)

    if ymin is None:
        ymin = np.nanmin(y[view_part]) - 3.5*np.nanmedian(err[view_part])
    if not ymax:
        ymax = max(1. + 4*np.nanmedian(err[view_part]), 1.08)
    ax.set_ylim(ymin, ymax)

    # Expand mask by 1 pixel around each masked range
    # to draw the lines correctly
    mask_idx = np.where(mask == 0)[0]
    big_mask_idx = np.union1d(mask_idx+1, mask_idx-1)

    # Trim the edges to avoid IndexError
    if len(mask) in big_mask_idx:
        big_mask_idx = np.delete(big_mask_idx, -1)
    if -1 in big_mask_idx:
        big_mask_idx = np.delete(big_mask_idx, 0)

    big_mask = np.ones_like(mask, dtype=bool)
    big_mask[big_mask_idx] = False
    masked_range = np.ma.masked_where(big_mask, y)
    spectrum = np.ma.masked_where(~mask, y)
    if xunit == 'wl':
        ax.plot(x, masked_range, color='0.7', drawstyle='steps-mid', lw=0.9)
        ax.errorbar(x, spectrum, err, ls='', color='gray', lw=1.)
        ax.plot(x, spectrum, color='k', drawstyle='steps-mid', lw=1.)
    else:
        ax.plot(vel, masked_range, color='0.7', drawstyle='steps-mid', lw=0.9)
        ax.errorbar(vel, spectrum, err, ls='', color='gray', lw=1.)
        ax.plot(vel, spectrum, color='k', drawstyle='steps-mid', lw=1.)
    ax.axhline(0., ls='--', color='0.7', lw=0.7)

    if plot_fit and (isinstance(dataset.best_fit, dict) or
                     isinstance(dataset.pars, dict)):
        if xunit == 'wl':
            ax.plot(wl_line, profile, **line_props)
        else:
            ax.plot(vel_line, profile, **line_props)

        if N_highlight > 0:
            if xunit == 'wl':
                ax.plot(wl_line, profile_hl, **hl_line_props)
            else:
                ax.plot(vel_line, profile_hl, **hl_line_props)

        if residuals:
            p_data = np.interp(vel, vel_line, profile)
            if norm_resid:
                masked_resid = (masked_range - p_data)/err
                resid = (spectrum - p_data)/err
            else:
                masked_resid = masked_range - p_data
                resid = spectrum - p_data

            if xunit == 'wl':
                cax.plot(x, masked_resid, color='0.7',
                         drawstyle='steps-mid', lw=0.9)
                cax.plot(x, resid, color='k', drawstyle='steps-mid', lw=1.)
            else:
                cax.plot(vel, masked_resid, color='0.7',
                         drawstyle='steps-mid', lw=0.9)
                cax.plot(vel, resid, color='k', drawstyle='steps-mid', lw=1.)
            if norm_resid:
                cax.axhline(3, ls=':', color='crimson', lw=0.5)
                cax.axhline(-3, ls=':', color='crimson', lw=0.5)
                res_min = 4
                res_max = -4
            else:
                if xunit == 'wl':
                    cax.errorbar(x, resid, err, ls='', color='gray', lw=1.)
                    cax.plot(x, 3*err, ls=':', color='crimson', lw=1.)
                    cax.plot(x, -3*err, ls=':', color='crimson', lw=1.)
                else:
                    cax.errorbar(vel, resid, err, ls='', color='gray', lw=1.)
                    cax.plot(vel, 3*err, ls=':', color='crimson', lw=1.)
                    cax.plot(vel, -3*err, ls=':', color='crimson', lw=1.)
                res_min = np.nanmax(4*err)
                res_max = np.nanmin(-4*err)
            cax.axhline(0., ls='--', color='0.7', lw=0.7)
            # cax.set_xticklabels([''])
            cax.tick_params(labelbottom='off')
            cax.set_yticklabels([''])
            cax.set_ylim(res_min, res_max)

    if nolabels:
        if axis:
            pass
        else:
            fig.subplots_adjust(bottom=0.07, right=0.98, left=0.08, top=0.98)
    else:
        if xunit == 'wl':
            ax.set_xlabel("${\\rm Wavelength}\ \ [{\\rm \\AA}]$")
        else:
            ax.set_xlabel("${\\rm Rel. velocity}\ \ [{\\rm km\,s^{-1}}]$")
        ax.set_ylabel("${\\rm Normalized\ flux}$")

    ax.minorticks_on()
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.axhline(1., ls='--', color='k')
    ax.axhline(1. + cont_err, ls=':', color='gray')
    ax.axhline(1. - cont_err, ls=':', color='gray')

    # Check if the region has a predefined label or not:
    if hasattr(region, 'label'):
        if region.label == '':
            all_trans_str = ["${\\rm "+trans.replace('_', '\ ')+"}$"
                             for trans in lines_in_view]
            line_string = "\n".join(all_trans_str)
        else:
            line_string = region.label

    else:
        all_trans_str = ["${\\rm "+trans.replace('_', '\ ')+"}$"
                         for trans in lines_in_view]
        line_string = "\n".join(all_trans_str)

    if loc == 'right':
        label_x = 0.97
    elif loc == 'left':
        label_x = 0.03
    else:
        label_x = 0.03
        loc = 'left'

    if legend:
        ax.text(label_x, 0.08, line_string, va='bottom', ha=loc,
                transform=ax.transAxes, fontsize=fontsize,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='white'))

    # plt.tight_layout()
    if axis:
        pass
    else:
        fig.set_tight_layout(True)

    if show:
        plt.show()

    return (ax, lines_in_view)


def plot_residual(dataset, line_tag, index=0, rebin=1,
                  xmin=None, xmax=None, axis=None):
    """
    Plot residuals for the best-fit to a given absorption line.

    Parameters
    ----------
    dataset : class DataSet
        An instance of DataSet class containing the line region to plot.

    line_tag : str
        The line tag of the line to show, e.g., 'FeII_2374'

    index : int   [default = 0]
        The line index. When fitting the same line in multiple
        spectra this indexed points to the index of the given region
        to be plotted.

    rebin: int
        Integer factor for rebinning the spectral data.

    xmin : float
        The lower x-limit in relative velocity (km/s).
        If nothing is given, the extent of the region is used.

    xmax : float
        The upper x-limit in relative velocity (km/s).
        If nothing is given, the extent of the region is used.

    axis : `matplotlib.axes.Axes <https://matplotlib.org/api/axes_api.html>`_
        The plotting axis of matplotlib.
        If `None` is given, a new figure and axis will be created.
    """

    if line_tag not in dataset.all_lines:
        dataset.add_line(line_tag, active=False)
        dataset.prepare_dataset()

    regions_of_line = dataset.find_line(line_tag)
    region = regions_of_line[index]

    x, y, err, mask = region.unpack()
    x_orig = x.copy()
    # cont_err = region.cont_err
    kernel = region.kernel
    kernel_nsub = region.kernel_nsub

    if rebin > 1:
        x, y, err = rebin_spectrum(x, y, err, rebin)
        mask = rebin_bool_array(mask, rebin)

    ref_line = dataset.lines[line_tag]
    l0, f, gam = ref_line.get_properties()
    l_ref = l0*(dataset.redshift + 1)

    # - Check if lines are separated by more than 200 km/s
    #   if so, then remove the line from the view.
    lines_in_view = list()
    for line in region.lines:
        l0 = line.l0
        delta_v = (l0*(dataset.redshift + 1) - l_ref) / l_ref * 299792.458
        if np.abs(delta_v) <= 150 or line.ion[-1].islower() is True:
            lines_in_view.append(line.tag)

    if axis:
        ax = axis
    else:
        # plt.close('all')
        fig = plt.figure(figsize=(6, 3.5))
        ax = fig.add_subplot(111)
        fig.subplots_adjust(bottom=0.15, right=0.97, top=0.98)

    if (isinstance(dataset.best_fit, dict) or isinstance(dataset.pars, dict)):
        if isinstance(kernel, float):
            npad = 50
            N_pix = len(x)*3
            dx = np.diff(x)[0]
            wl_line = np.logspace(np.log10(x.min() - npad*dx),
                                  np.log10(x.max() + npad*dx),
                                  N_pix)
            pxs = np.diff(wl_line)[0] / wl_line[0] * 299792.458
        elif isinstance(kernel, np.ndarray):
            assert kernel.shape[0] == len(x) * kernel_nsub
            # evaluate on the input grid
            if kernel_nsub > 1:
                N = kernel_nsub * len(x)
                wl_line = np.linspace(x.min(), x.max(), N)
            else:
                wl_line = x_orig
        else:
            err_msg = "Invalid type of `kernel`: %r" % type(kernel)
            raise TypeError(err_msg)
        ref_line = dataset.lines[line_tag]
        l0, f, gam = ref_line.get_properties()
        l_ref = l0*(dataset.redshift + 1)

        tau = np.zeros_like(wl_line)

        if isinstance(dataset.best_fit, dict):
            params = dataset.best_fit
        else:
            params = dataset.pars

        for line in region.lines:
            # Load line properties
            l0, f, gam = line.get_properties()
            ion = line.ion
            n_comp = len(dataset.components[ion])

            ion = ion.replace('*', 'x')
            if line.active:
                for n in range(n_comp):
                    z = params['z%i_%s' % (n, ion)].value
                    b = params['b%i_%s' % (n, ion)].value
                    logN = params['logN%i_%s' % (n, ion)].value
                    tau += voigt.Voigt(wl_line, l0, f,
                                       10**logN, 1.e5*b, gam, z=z)

        profile_int = np.exp(-tau)
        if isinstance(kernel, float):
            sigma_instrumental = kernel / 2.35482 / pxs
            LSF = gaussian(len(wl_line)/2, sigma_instrumental)
            LSF = LSF/LSF.sum()
            profile_broad = fftconvolve(profile_int, LSF, 'same')
            profile = profile_broad[npad:-npad]
            wl_line = wl_line[npad:-npad]
        else:
            profile = voigt.convolve_numba(profile_int, kernel)
            if rebin > 1:
                _, profile, _ = rebin_spectrum(x_orig, profile, 0.1*profile, rebin)

    vel = (x - l_ref) / l_ref * 299792.458
    y = y - profile

    if not xmin:
        xmin = -region.velspan

    if not xmax:
        xmax = region.velspan
    ax.set_xlim(xmin, xmax)

    view_part = (vel > xmin) * (vel < xmax)

    ymin = y[view_part].min() - 1.5*err.mean()
    ymax = (y*mask)[view_part].max() + 2*err.mean()
    ax.set_ylim(ymin, ymax)

    # Expand mask by 1 pixel around each masked range
    # to draw the lines correctly
    mask_idx = np.where(mask == 0)[0]
    big_mask_idx = np.union1d(mask_idx+1, mask_idx-1)
    big_mask = np.ones_like(mask, dtype=bool)
    big_mask[big_mask_idx] = False
    masked_range = np.ma.masked_where(big_mask, y)
    ax.plot(vel, masked_range, color='0.7', drawstyle='steps-mid', lw=0.9)

    spectrum = np.ma.masked_where(~mask, y)
    # error = np.ma.masked_where(~mask, err)
    ax.plot(vel, spectrum, color='0.4', drawstyle='steps-mid')
    ax.axhline(0., ls='--', color='0.7', lw=0.7)

    ax.axhline(0., ls='--', color='k')
    ax.plot(vel, err, ls=':', color='b')
    ax.plot(vel, -err, ls=':', color='b')

    if not axis:
        plt.show(block=True)

    return (ax, lines_in_view)


def plot_excitation(dataset, molecule):
    """Plot the excitation diagram for a given `molecule`"""
    if molecule == 'H2':
        def g(J):
            Ij = J % 2
            return (2*J + 1.)*(2*Ij + 1.)
    elif molecule == 'CO':
        def g(J):
            return 2*J + 1.
    else:
        def g(J):
            return 2*J + 1.

    if dataset.best_fit is not None:
        pars = dataset.best_fit
    else:
        pars = dataset.pars

    Jmax = np.max([band[1] for band in dataset.molecules[molecule]])
    logN = list()
    logN_err = list()
    E = list()
    g_J = list()
    for num in range(Jmax + 1):
        par_name = 'logN0_%sJ%i' % (molecule, num)
        logN.append(pars[par_name].value)
        err = pars[par_name].stderr if pars[par_name].stderr else 0.
        logN_err.append(err)
        E.append(molecules.energy_of_level(molecule, num))
        g_J.append(g(num))

    logN = np.array(logN)
    logN_err = np.array(logN_err)
    g_J = np.array(g_J)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    y = np.log(10**(logN-logN[0]))+np.log(g_J[0]/g_J)
    y_err = np.log(10.) * logN_err
    ax.errorbar(E, y, y_err, color='k', marker='s', ls='')
    ax.set_xlabel(r"Energy ${\rm E}_J-{\rm E}_0$ (K)", fontsize=14)
    ax.set_ylabel(r"${\rm ln}\left(\frac{N_J}{N_0}\ \frac{g_0}{g_J}\right)$",
                  fontsize=14)

    N0 = np.random.normal(logN[0], logN_err[0], 10000)
    N1 = np.random.normal(logN[1], logN_err[1], 10000)
    T_dist = molecules.calculate_T(molecule, N0, N1, 0, 1)
    T_01 = np.median(T_dist)
    T_01_err = tuple(np.percentile(T_dist, [16., 84.]) - T_01)
    T_array = np.linspace(0., np.max(E), 100)
    ax.plot(T_array, -T_array/T_01, 'k:')
    ax.text(0.98, 0.98,
            r"$T_{01} = %.0f_{%+.0f}^{%+.0f}$ K" % ((T_01,) + T_01_err),
            ha='right', va='top', transform=ax.transAxes,
            fontsize=14)
    plt.tight_layout()


def show_H2_bands(ax, z, bands, Jmax, color='blue', short_labels=False):
    """
    Add molecular H2 band identifications to a given matplotlib axis.

    Parameters
    ==========
    ax : `matplotlib.axes.Axes <https://matplotlib.org/api/axes_api.html>`_
        The axis instance to decorate with H2 line identifications.

    z : float
        The redshift of the H2 system.

    bands : list(str)
        A list of molecular bands of H2.
        Ex: ['BX(0-0)', 'BX(1-0)']

    Jmax : int or list(int)
        The highest rotational J-level to include in the identification.
        A list of Jmax for each band can be passed as well.
        Lines are currently only defined up to J=7.

    """
    label_threshold = 0.01
    lyman_bands = 'B' in [b[0] for b in bands]
    werner_bands = 'C' in [b[0] for b in bands]
    N_lines = 2*lyman_bands + 1*werner_bands

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    for band, this_Jmax in zip(bands, Jmax):
        if band not in molecules.H2.keys():
            print(" Invalid band name: %s" % band)
            continue

        transitions = molecules.H2[band]
        nu = int(band.split('(')[1].split('-')[0])
        if 'B' in band:
            y0 = 0.55
            # Lyman band
            y1 = y0 + (nu % 2)*(0.90-y0)/float(N_lines)
            y2 = y1 + (0.90-y0)/float(N_lines)
        else:
            if N_lines == 1:
                y0 = 0.67
                # Werner band
                y1 = y0
                y2 = y1 + (0.87-y0)/float(N_lines)
            else:
                y0 = 0.55
                # Werner band
                y1 = y0 + 2*(0.90-y0)/float(N_lines)
                y2 = y1 + (0.89-y0)/float(N_lines)

        if N_lines == 3:
            va = 'center'
        elif N_lines == 2:
            va = 'bottom'
        else:
            va = 'bottom'

        band_l0 = list()
        labels = list()
        for j in range(this_Jmax+1):
            for tag in transitions[j]:
                line = Line(tag)
                l0 = line.l0*(z+1)
                band_l0.append(l0)
                label_x = (l0 - xmin)/(xmax - xmin)
                ax.axvline(l0, y1+(y2-y1)/1.5, y2, color=color)
                # Check overlap with other text labels:
                for label in labels:
                    x_0, y_0 = label.get_position()
                    if np.fabs(x_0 - label_x) <= label_threshold:
                        if label_x > x_0:
                            x_new = x_0 - label_threshold/2.
                            label_x += label_threshold/2.
                        else:
                            x_new = x_0 + label_threshold/2.
                            label_x -= label_threshold/2.
                        label.set_position((x_new, y_0))

                text = ax.text(label_x, y1, "%i" % j, fontsize=10, color=color,
                               ha='center', va=va, transform=ax.transAxes,
                               clip_on=True)
                labels.append(text)

        # Draw horizontal line to connect the marks:
        bar_min = (min(band_l0) - xmin)/(xmax - xmin)
        bar_max = (max(band_l0) - xmin)/(xmax - xmin)
        ax.axhline(y2*(ymax-ymin) + ymin, bar_min, bar_max, color=color)
        band_x = (min(band_l0) + max(band_l0))/2.
        if short_labels:
            band_str = 'L%i' if 'B' in band else 'W%i'
            band_str = band_str % nu
        else:
            band_str = band
        ax.text(band_x, (y2+0.02)*(ymax-ymin) + ymin, band_str, color=color,
                clip_on=True, ha='center')
    plt.draw()


def plot_H2(dataset, n_rows=None, xmin=None, xmax=None,
            ymin=-0.1, ymax=2.5, short_labels=False,
            rebin=1, smooth=0):
    """
    Generate plot for H2 absorption lines.

    Parameters
    ==========
    dataset : :class:`VoigtFit.DataSet`
        An instance of the class :class:`VoigtFit.DataSet` containing
        the H2 lines to plot.

    n_rows : int   [default = None]
        Number of rows to show in figure.
        If None, the number will be determined automatically.

    xmin : float
        The lower x-limit in Å.
        If nothing is given, the extent of the fit region is used.

    xmax : float
        The upper x-limit in Å.
        If nothing is given, the extent of the fit region is used.

    ymin : float   [default = -0.1]
        The lower y-limit in normalized flux units.

    ymax : float   [default = 2.5]
        The upper y-limit in normalized flux units.

    rebin : int   [defualt = 1]
        Rebinning factor for the spectrum, default is no binning.

    smooth : float   [default = 0]
        Width of Gaussian kernel for smoothing.

    """
    molecule = 'H2'

    if len(dataset.data) > 1:
        complex_warning = """
        This appears to be a complex dataset.
        Consider writing a dedicated script for plotting.
        """
        specIDs = list()
        min_wl = list()
        max_wl = list()
        for reg in dataset.regions:
            molecules_in_region = [molecule in l.tag for l in reg.lines]
            if np.any(molecules_in_region):
                specIDs.append(reg.specID)
                min_wl.append(reg.wl.min())
                max_wl.append(reg.wl.max())

        # Remove duplicate spectral IDs:
        specIDs = list(set(specIDs))
        if len(specIDs) > 1:
            print(complex_warning)
            return
        elif len(specIDs) == 0:
            print("No lines for %s were found!" % molecule)
            return

        # Find the data chunk that defines the molecular lines:
        for this_chunk in dataset.data:
            if this_chunk['specID'] == specIDs[0]:
                data_chunk = this_chunk

    else:
        data_chunk = dataset.data[0]
        min_wl = list()
        max_wl = list()
        for reg in dataset.regions:
            molecules_in_region = [molecule in l.tag for l in reg.lines]
            if np.any(molecules_in_region):
                min_wl.append(reg.wl.min())
                max_wl.append(reg.wl.max())

    if not data_chunk['norm']:
        print("    The spectrum is not normalized")
        print("    Consider writing a dedicated script for plotting")
        return

    wl = data_chunk['wl']
    flux = data_chunk['flux']
    error = data_chunk['error']
    res = data_chunk['res']
    nsub = data_chunk['nsub']
    if isinstance(res, str):
        kernel = load_lsf(res, wl, nsub)
    else:
        kernel = res

    wl_profile = wl.copy()
    if rebin > 1:
        wl, flux, error = rebin_spectrum(wl, flux, error, int(rebin))
    if smooth > 0:
        flux = gaussian_filter1d(flux, smooth)

    bands = [item[0] for item in dataset.molecules[molecule]]
    Jmax = [item[1] for item in dataset.molecules[molecule]]
    molecular_lines = [line for line in dataset.lines.values()
                       if molecule in line.tag]
    profile = evaluate_profile(wl_profile, dataset.best_fit, dataset.redshift,
                               molecular_lines, dataset.components, kernel,
                               sampling=3)
    if not xmin:
        xmin = min(min_wl)
    if not xmax:
        xmax = max(max_wl)
    if not n_rows:
        n_rows = round((xmax-xmin)/120.)

    width = 10.
    height = 2*(n_rows+0.1) + 0.1
    fig = plt.figure(figsize=(width, height))
    wl_range = (xmax-xmin)/n_rows
    n_rows = int(n_rows)
    z = dataset.best_fit['z0_H2J0'].value
    for num in range(n_rows):
        ax = fig.add_subplot(n_rows, 1, num+1)
        ax.plot(wl, flux, color='k', lw=0.5, drawstyle='steps-mid')
        ax.plot(wl_profile, profile, color='r', lw=0.8)
        ax.axhline(0., ls=':', color='0.4')
        ax.axhline(1., ls=':', color='0.4')
        ax.set_xlim(xmin + num*wl_range, xmin + (num+1)*wl_range)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(u"Normalized Flux", fontsize=12)
        ax.minorticks_on()

        if num == n_rows-1:
            ax.set_xlabel(u"Observed Wavelength  (Å)", fontsize=14)

        show_H2_bands(ax, z, bands, Jmax, color='blue',
                      short_labels=short_labels)

    fig.set_tight_layout(True)
    plt.show()


# ===================================================================================
#
#   Text output functions:
# --------------------------


def print_results(dataset, params, elements='all', velocity=True, systemic=0):
    """
    Print the parameters of the best-fit.

    Parameters
    ----------
    dataset : :class:`VoigtFit.DataSet`
        An instance of the :class:`VoigtFit.DataSet` class containing
        the line region to plot.

    params : lmfit.Parameters_
        Output parameter dictionary, e.g., :attr:`dataset.Dataset.best_fit`.
        See lmfit_ for details.

    elements : list(str)   [default = 'all']
        A list of ions for which to print the best-fit parameters.
        By default all ions are shown.

    velocity : bool   [default = True]
        Show the components in relative velocity or redshift.

    systemic : float   [default = 0]
        Systemic redshift used to calculate the relative velocities.
        By default the systemic redshift defined in the dataset is used.



    .. _lmfit: https://lmfit.github.io/lmfit-py/index.html
    """

    if systemic:
        z_sys = systemic
    else:
        z_sys = dataset.redshift

    print "\n  Best fit parameters\n"
    # print "\t\t\t\tlog(N)\t\t\tb"
    print "\t\t\t\tb\t\t\tlog(N)"
    if elements == 'all':
        for ion in sorted(dataset.components.keys()):
            lines_for_this_ion = []
            for line_tag, line in dataset.lines.items():
                if line.ion == ion and line.active:
                    lines_for_this_ion.append(line_tag)

            all_transitions = [trans.split('_')[1]
                               for trans in sorted(lines_for_this_ion)]
            # Split list of transitions into chunks of length=4
            # join the transitions in each chunks
            # and join each chunk with 'newline'
            trans_chunks = [", ".join(sublist)
                            for sublist in list(chunks(all_transitions, 4))]
            indent = '\n'+(len(ion)+2)*' '
            trans_string = indent.join(trans_chunks)

            print ion + "  "+trans_string
            n_comp = len(dataset.components[ion])
            for n in range(n_comp):
                ion = ion.replace('*', 'x')
                z = params['z%i_%s' % (n, ion)].value
                b = params['b%i_%s' % (n, ion)].value
                logN = params['logN%i_%s' % (n, ion)].value
                z_err = params['z%i_%s' % (n, ion)].stderr
                b_err = params['b%i_%s' % (n, ion)].stderr
                logN_err = params['logN%i_%s' % (n, ion)].stderr
                if z_err is None:
                    z_err = -1.
                if b_err is None:
                    b_err = -1.
                if logN_err is None:
                    logN_err = -1.

                if velocity:
                    z_std = z_err/(z_sys+1)*299792.458
                    z_val = (z-z_sys)/(z_sys+1)*299792.458
                    z_format = "v = %5.1f +/- %.1f\t"
                else:
                    z_std = z_err
                    z_val = z
                    z_format = "z = %.6f +/- %.6f"

                output_string = z_format % (z_val, z_std) + "\t"
                output_string += "%6.2f +/- %6.2f\t" % (b, b_err)
                output_string += "%.3f +/- %.3f" % (logN, logN_err)

                print output_string

            print ""

    else:
        for ion in elements:
            lines_for_this_ion = []
            for line_tag, line in dataset.lines.items():
                if line.ion == ion and line.active:
                    lines_for_this_ion.append(line_tag)

            all_transitions = [trans.split('_')[1]
                               for trans in sorted(lines_for_this_ion)]
            all_transitions = ", ".join(all_transitions)
            print ion + "  "+all_transitions
            n_comp = len(dataset.components[ion])
            for n in range(n_comp):
                ion = ion.replace('*', 'x')
                z = params['z%i_%s' % (n, ion)].value
                b = params['b%i_%s' % (n, ion)].value
                logN = params['logN%i_%s' % (n, ion)].value
                b_err = params['b%i_%s' % (n, ion)].stderr
                logN_err = params['logN%i_%s' % (n, ion)].stderr

                if velocity:
                    z_val = (z-z_sys)/(z_sys+1)*299792.458
                    z_format = "v = %5.1f\t"
                else:
                    z_val = z
                    z_format = "z = %.6f"

                output_string = z_format % (z_val, z_std) + "\t"
                output_string += "%6.2f +/- %6.2f\t" % (b, b_err)
                output_string += "%.3f +/- %.3f" % (logN, logN_err)

                print output_string

            print ""


def print_cont_parameters(dataset):
    """ Print the Chebyshev coefficients of the continuum normalization."""
    if dataset.cheb_order >= 0:
        print ""
        print "  Chebyshev coefficients for fitting regions:"
        for reg_num, region in enumerate(dataset.regions):
            lines_in_region = ", ".join([line.tag for line in region.lines])
            print "   Region no. %i : %s" % (reg_num, lines_in_region)
            cheb_parnames = list()
            # Find Chebyshev parameters for this region:
            # They are named like 'R0_cheb_p0, R0_cheb_p1, R1_cheb_p0, etc...'
            for parname in dataset.best_fit.keys():
                if 'R%i_cheb' % reg_num in parname:
                    cheb_parnames.append(parname)
            # This could be calculated at the point of generating
            # the parameters, since this is a fixed data structure
            # Sort the names, to arange the coefficients right:
            cheb_parnames = sorted(cheb_parnames)
            for i, parname in enumerate(cheb_parnames):
                coeff = dataset.best_fit[parname]
                line = " p%-2i  =  %.3e    %.3e" % (i,
                                                    coeff.value, coeff.stderr)
                print line
            print ""
    else:
        print "\n  No Chebyshev polynomials have been defined."
        print "  cheb_order = %i " % dataset.cheb_order


def print_metallicity(dataset, params, logNHI, err=0.1):
    """
    Print the metallicity derived from different species.
    This will add the column densities for all components of a given ion.

    Parameters
    ----------
    dataset : :class:`VoigtFit.DataSet`
        An instance of the :class:`VoigtFit.DataSet` class containing
        the definition of data and absorption lines.

    params : lmfit.Parameters_
        Output parameter dictionary, e.g., :attr:`VoigtFit.DataSet.best_fit`.
        See lmfit_ for details.

    logNHI : float
        Column density of neutral hydrogen.

    err : float   [default = 0.1]
        Uncertainty (1-sigma) on `logNHI`.


    .. _lmfit: https://lmfit.github.io/lmfit-py/index.html

    """

    print "\n  Metallicities\n"
    print "  log(NHI) = %.3f +/- %.3f\n" % (logNHI, err)
    logNHI = np.random.normal(logNHI, err, 10000)
    for ion in sorted(dataset.components.keys()):
        element = ion[:2] if ion[1].islower() else ion[0]
        logN = []
        logN_err = []
        N_tot = []
        for par in params.keys():
            if par.find('logN') >= 0 and par.find(ion) >= 0:
                N_tot.append(params[par].value)
                if params[par].stderr < 0.8:
                    logN.append(params[par].value)
                    logN_err.append(params[par].stderr)

        ION = [np.random.normal(n, e, 10000) for n, e in zip(logN, logN_err)]
        log_sum = np.log10(np.sum(10**np.array(ION), 0))
        l68, abundance, u68 = np.percentile(log_sum, [16, 50, 84])
        std_err = np.std(np.log10(np.sum(10**np.array(ION), 0)))

        logN_tot = np.random.normal(abundance, std_err, 10000)
        N_solar, N_solar_err = Asplund.photosphere[element]
        solar_abundance = np.random.normal(N_solar, N_solar_err, 10000)

        metal_array = logN_tot - logNHI - (solar_abundance - 12.)
        metal = np.mean(metal_array)
        metal_err = np.std(metal_array)
        print "  [%s/H] = %.3f +/- %.3f" % (ion, metal, metal_err)


def print_total(dataset):
    """
    Print the total column densities of all species. This will sum *all*
    the components of each ion. The uncertainty on the total column density
    is calculated using random resampling within the errors of each component.
    """

    if isinstance(dataset.best_fit, dict):
        params = dataset.best_fit
        print "\n  Total Column Densities\n"
        for ion in sorted(dataset.components.keys()):
            # element = ion[:2] if ion[1].islower() else ion[0]
            logN = []
            logN_err = []
            N_tot = []
            for par in params.keys():
                if par.find('logN') >= 0 and par.split('_')[1] == ion:
                    N_tot.append(params[par].value)
                    if params[par].stderr < 0.5:
                        logN.append(params[par].value)
                        if params[par].stderr < 0.01:
                            logN_err.append(0.01)
                        else:
                            logN_err.append(params[par].stderr)

            ION = [np.random.normal(n, e, 10000)
                   for n, e in zip(logN, logN_err)]
            logsum = np.log10(np.sum(10**np.array(ION), 0))
            l68, abundance, u68 = np.percentile(logsum, [16, 50, 84])
            std_err = np.std(logsum)

            print "  logN(%s) = %.2f +/- %.2f" % (ion, abundance, std_err)

    else:
        error_msg = """
        [ERROR] - The dataset has not yet been fitted. No parameters found!
        """
        print(error_msg)


def print_T_model_pars(dataset, thermal_model, filename=None):
    """Print the turbulence and temperature parameters for physical model."""
    print("")
    print(u"  No:     Temperature [K]       Turbulence [km/s]")
    if filename:
        out_file = open(filename, 'w')
        out_file.write(u"# No:     Temperature [K]       Turbulence [km/s] \n")

    thermal_components = list(set(sum(thermal_model.values(), [])))

    for comp_num in thermal_components:
        T_name = 'T_%i' % comp_num
        turb_name = 'turb_%i' % comp_num
        T_fit = dataset.best_fit[T_name]
        turb_fit = dataset.best_fit[turb_name]
        par_tuple = (comp_num, T_fit.value, T_fit.stderr,
                     turb_fit.value, turb_fit.stderr)
        print(u"  %-3i   %.2e ± %.2e    %.2e ± %.2e" % par_tuple)
        if filename:
            out_file.write(u"  %-3i   %.2e ± %.2e    %.2e ± %.2e" % par_tuple)

    print("\n")
    if filename:
        out_file.close()


def sum_components(dataset, ion, components):
    """
    Calculate the total abundance for given `components` of the given `ion`.

    Parameters
    ----------
    dataset : :class:`VoigtFit.DataSet`
        An instance of the :class:`VoigtFit.DataSet` class containing
        the definition of data and absorption lines.

    ion : str
        Ion for which to calculate the summed abundance.

    components : list(int)
        List of indeces of the components to sum over.

    Returns
    -------
    total_logN : float
        The 10-base log of total column density.

    total_logN_err : float
        The error on the 10-base log of total column density.
    """
    if hasattr(dataset.best_fit, 'keys'):
        pass
    else:
        print " [ERROR] - Best fit parameters are not found."
        print "           Make sure the fit has converged..."
        return None

    logN = list()
    logN_err = list()
    for num in components:
        parname = 'logN%i_%s' % (num, ion)
        par = dataset.best_fit[parname]
        logN.append(par.value)
        logN_err.append(par.stderr)

    logN_pdf = [np.random.normal(n, e, 10000) for n, e in zip(logN, logN_err)]
    logsum = np.log10(np.sum(10**np.array(logN_pdf), 0))
    total_logN = np.median(logsum)
    total_logN_err = np.std(logsum)

    return total_logN, total_logN_err


def save_parameters_to_file(dataset, filename):
    """Save best-fit parameters to file."""
    header = "#comp   ion   redshift               b (km/s)       log(N/cm^-2)"
    with open(filename, 'w') as output:
        output.write(header + "\n")
        for ion in sorted(dataset.components.keys()):
            for i in range(len(dataset.components[ion])):
                z = dataset.best_fit['z%i_%s' % (i, ion)]
                logN = dataset.best_fit['logN%i_%s' % (i, ion)]
                b = dataset.best_fit['b%i_%s' % (i, ion)]
                par_tuple = (i, ion, z.value, z.stderr,
                             b.value, b.stderr,
                             logN.value, logN.stderr)
                line_fmt = "%3i  %7s  %.6f %.6f    %6.2f %6.2f    %.3f %.3f"
                output.write(line_fmt % par_tuple + "\n")
            output.write("\n")

        # Write a python script friendly version to copy into script:
        output.write("\n\n# Python script version:\n")
        output.write("# The commands below can be copied directly to a script.\n")
        z_sys = dataset.redshift
        output.write("# dataset.redshift = %.6f.\n" % z_sys)
        for ion in dataset.components.keys():
            for i in range(len(dataset.components[ion])):
                z = dataset.best_fit['z%i_%s' % (i, ion)]
                logN = dataset.best_fit['logN%i_%s' % (i, ion)]
                b = dataset.best_fit['b%i_%s' % (i, ion)]
                vel_value = (z.value - z_sys)/(z_sys + 1)*299792.458
                par_tuple = (ion, vel_value, b.value, logN.value)
                line_fmt = "# dataset.add_component_velocity('%s', %.1f, %.1f, %.1f)"
                output.write(line_fmt % par_tuple + "\n")
            output.write("\n")


def save_cont_parameters_to_file(dataset, filename):
    """Save Chebyshev coefficients to file."""
    with open(filename, 'w') as output:
        header = "# Chebyshev Polynomial Coefficients"
        output.write(header + "\n")

        for reg_num, region in enumerate(dataset.regions):
            output.write("# Region %i: \n" % reg_num)
            cheb_parnames = list()
            # Find Chebyshev parameters for this region:
            # They are named like 'R0_cheb_p0, R0_cheb_p1, R1_cheb_p0, etc...'
            for parname in dataset.best_fit.keys():
                if 'R%i_cheb' % reg_num in parname:
                    cheb_parnames.append(parname)
            # This should be calculated at the point of generating
            # the parameters, since this is a fixed data structure
            # Sort the names, to arange the coefficients right:
            cheb_parnames = sorted(cheb_parnames)
            for i, parname in enumerate(cheb_parnames):
                coeff = dataset.best_fit[parname]
                line = " p%-2i  =  %.3e    %.3e" % (i, coeff.value, coeff.stderr)
                output.write(line + "\n")
            output.write("\n")


def save_fit_regions(dataset, filename, individual=False, path=''):
    """
    Save fit regions to ASCII file.

    Parameters
    ----------
    filename : str
        Filename to be used. If the filename already exists, it will be overwritten.
        A `.reg` file extension will automatically be append if not present already.

    individual : bool   [default = False]
        If `True`, save each fitting region to a separate file.
        The individual filenames will be the basename given as `filename`
        with `_regN` appended, where `N` is an integer referring to the region number.

    path : str   [default = '']
        Specify a path to prepend to the filename in order to save output to a given
        directory or path. Can be given both as relative or absolute path.
        If the directory does not exist, it will be created.
        The final filename will be:

            `path/` + `filename` [+ `_regN`] + `.reg`

    """
    base, file_ext = splitext(filename)
    if file_ext != '.reg':
        filename += '.reg'

    if path:
        if not os.path.exists(path):
            os.mkdir(path)

        if path[-1] != '/':
            path += '/'

    elif path is None:
        path = ''

    filename = path + filename

    if individual:
        for reg_num, region in enumerate(dataset.regions):
            wl, flux, err, mask = region.unpack()
            if dataset.best_fit:
                p_obs = voigt.evaluate_profile(wl, dataset.best_fit, dataset.redshift,
                                               dataset.lines.values(), dataset.components,
                                               region.kernel)
            else:
                p_obs = np.ones_like(wl)
            data_table = np.column_stack([wl, flux, err, p_obs, mask])
            line_string = ", ".join([line.tag for line in region.lines])
            filebase, ext = splitext(filename)
            filebase += '_reg%i' % reg_num
            reg_filename = filebase + ext
            with open(reg_filename, 'w') as out_file:
                out_file.write("# Best-fit normalized data and profile from VoigtFit\n")
                out_file.write("# Lines in regions: %s \n" % line_string)
                out_file.write("# column 1 : Wavelength\n")
                out_file.write("# column 2 : Normalized flux\n")
                out_file.write("# column 3 : Normalized error\n")
                out_file.write("# column 4 : Best-fit profile\n")
                out_file.write("# column 5 : Pixel mask, 1=included, 0=excluded\n")
                np.savetxt(out_file, data_table, fmt="%.3f   %.4f   %.4f   %.4f   %i")

    else:
        # Concatenate all regions and save to one file
        l_tot = list()
        f_tot = list()
        e_tot = list()
        p_tot = list()
        m_tot = list()
        for region in dataset.regions:
            wl, flux, err, mask = region.unpack()
            if dataset.best_fit:
                p_obs = voigt.evaluate_profile(wl, dataset.best_fit, dataset.redshift,
                                               dataset.lines.values(), dataset.components,
                                               region.kernel)
            else:
                p_obs = np.ones_like(wl)
            l_tot.append(wl)
            f_tot.append(flux)
            e_tot.append(err)
            p_tot.append(p_obs)
            m_tot.append(mask)

        l_tot = np.concatenate(l_tot)
        f_tot = np.concatenate(f_tot)
        e_tot = np.concatenate(e_tot)
        p_tot = np.concatenate(p_tot)
        m_tot = np.concatenate(m_tot)

        data_table = np.column_stack([l_tot, f_tot, e_tot, p_tot, m_tot])
        with open(filename, 'w') as out_file:
            out_file.write("# Best-fit normalized data and profile from VoigtFit\n")
            out_file.write("# column 1 : Wavelength\n")
            out_file.write("# column 2 : Normalized flux\n")
            out_file.write("# column 3 : Normalized error\n")
            out_file.write("# column 4 : Best-fit profile\n")
            out_file.write("# column 5 : Pixel mask, 1=included, 0=excluded\n")
            np.savetxt(out_file, data_table, fmt="%.3f   %.4f   %.4f   %.4f   %i")


def show_components(self, ion=None):
    """
    Show the defined components for a given `ion`.
    By default, all ions are shown.
    """
    z_sys = self.redshift
    for ion, comps in self.components.items():
        print term.underline + "  %s:" % ion + term.reset
        for num, comp in enumerate(comps):
            z = comp[0]
            vel = (z - z_sys) / (z_sys + 1) * 299792.458
            print "   %2i  %+8.1f  %.6f   %6.1f   %5.2f" % (num, vel, z,
                                                            comp[1], comp[2])
