# -*- coding: UTF-8 -*-

__author__ = 'Jens-Kristian Krogager'

import os
from os.path import splitext
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import fftconvolve, gaussian
import numpy as np
import itertools

import voigt
import Asplund

plt.rcParams['lines.linewidth'] = 1.0


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


# ===================================================================================
#
#   Graphics output functions:
# ------------------------------
#  A4 figuse size:
#  fig = plt.figure(figsize=(7, 9.899))


# --- Deprecated Function:
def velocity_plot(dataset, vmin=-400, vmax=400, filename=None, max_rows=6, max_columns=2,
                  rebin=1, fontsize=12, subsample_profile=1, npad=50, ymin=None):
    """
    Similar functionality can be acheived using :func:`plot_all_lines`.
    This function is deprecated and will be removed in future versions.

    """
    # --- First figure out which lines to plot to avoid overlap
    #     of several lines defined in the same region.
    included_lines = list()
    lines_to_plot = list()
    vrange = (vmax - vmin)/4.
    for ref_line in dataset.lines.values():
        if ref_line.tag in included_lines:
            pass
        elif ref_line.ion[-1].islower():
            # do not plot individual figures for fine-structure lines
            included_lines.append(ref_line)
        else:
            region = dataset.find_line(ref_line.tag)
            lines_to_plot.append(ref_line.tag)
            if len(region.lines) == 1:
                included_lines.append(ref_line.tag)
            else:
                l_ref = ref_line.l0*(dataset.redshift + 1)
                for line in region.lines:
                    l0 = line.l0
                    delta_v = (l0*(dataset.redshift + 1) - l_ref) / l_ref * 299792.
                    if np.abs(delta_v) <= vrange or line.ion[-1].islower() is True:
                        included_lines.append(line.tag)

    # --- If a filename is given, set up a PDF container for saving to file:
    if filename:
        if '.pdf' not in filename:
            filename += '.pdf'
        pdf = matplotlib.backends.backend_pdf.PdfPages(filename)

    # --- Determine number of pages to create:
    pages = list(chunks(lines_to_plot, 2*max_rows))
    lines_in_figure = list()
    for contents in pages:
        # --- Determine figure size:
        # if len(contents) > 1:
        #     width = 10.5
        #     columns = 2
        # else:
        #     width = 5.25
        #     columns = 1
        #
        # heigth = (len(contents) + 1) / 2 * 15./max_rows
        # rows = (len(contents) + 1) / 2

        fig = plt.figure(figsize=(7, 8))
        # fig = plt.figure(figsize=(width, heigth))
        # fig.subplots_adjust(left=0.10, right=0.98, top=0.98, hspace=0.03, bottom=0.14)

        num = 1
        for line_tag in contents:
            if line_tag in lines_in_figure:
                pass
            else:
                ax = fig.add_subplot(max_rows, max_columns, num)
                _, LIV = plot_single_line(dataset, line_tag,
                                          plot_fit=False, linestyles=['--'],
                                          colors=['RoyalBlue'], rebin=rebin, nolabels=True,
                                          axis=ax, fontsize=fontsize, xmin=vmin, xmax=vmax,
                                          subsample_profile=subsample_profile, ymin=ymin)
                lines_in_figure += LIV
                ax.tick_params(length=7, labelsize=fontsize)
                ax.grid(True, color='0.6', ls='--', lw=0.5)
                if num < len(contents)-1:
                    ax.set_xticklabels([''])
                else:
                    ax.set_xlabel("${\\rm Velocity\ \ (km\ s^{-1})}$", fontsize=12)

                if num % max_columns == 1:
                    ax.set_ylabel("Norm. flux", fontsize=12)
                num += 1
                # LIV is a shorthand for 'lines_in_view'
        # fig.text(0.5, 0.02, "${\\rm Velocity\ \ (km\ s^{-1})}$",
        #          ha='center', va='bottom', transform=fig.transFigure,
        #          fontsize=14)
        # fig.text(0.01, 0.5, "Normalized flux",
        #          ha='left', va='center', transform=fig.transFigure,
        #          fontsize=14, rotation=90)

        # plt.tight_layout()
        fig.set_tight_layout(True)

        if filename:
            pdf.savefig(fig)

    if filename:
        pdf.close()
        print "\n  Output saved to PDF file:  " + filename

    plt.show()


def plot_all_lines(dataset, plot_fit=True, linestyles=['--'], colors=['b'],
                   rebin=1, fontsize=12, xmin=None, xmax=None, max_rows=4,
                   filename=None, show=True, subsample_profile=1, npad=50,
                   highlight=[], residuals=True, norm_resid=False):
    """
    Plot all active absorption lines. This function is a wrapper of the function
    :func:`plot_single_line`. For a complete description of input parameters,
    see the documentation for :func:`plot_single_line`.

    Parameters
    ----------
    dataset : :class:`dataset.DataSet`
        Instance of the :class:`dataset.DataSet` class containing the line regions to plot.

    max_rows : int   [default = 4]
        The maximum number of rows of figures. Each row consists of two figure panels.

    filename : str
        If a filename is given, the figures are saved to a pdf file.
    """

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
        else:
            region = dataset.find_line(ref_line.tag)
            lines_to_plot.append(ref_line.tag)
            if len(region.lines) == 1:
                included_lines.append(ref_line.tag)
            else:
                l_ref = ref_line.l0*(dataset.redshift + 1)
                for line in region.lines:
                    l0 = line.l0
                    delta_v = (l0*(dataset.redshift + 1) - l_ref) / l_ref * 299792.
                    if np.abs(delta_v) <= 150 or line.ion[-1].islower() is True:
                        included_lines.append(line.tag)

    # --- If *filename* is given, set up a PDF container for saving to file:
    if filename:
        if '.pdf' not in filename:
            filename += '.pdf'
        pdf = matplotlib.backends.backend_pdf.PdfPages(filename)

    # --- Determine number of pages to create:
    pages = list(chunks(lines_to_plot, 2*max_rows))
    lines_in_figure = list()
    for contents in pages:
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
        fig.subplots_adjust(left=0.10, right=0.98, top=0.98, hspace=0.03, bottom=0.14)

        num = 1
        for line_tag in contents:
            if line_tag in lines_in_figure:
                pass
            else:
                ax = fig.add_subplot(rows, columns, num)
                _, LIV = plot_single_line(dataset, line_tag,
                                          plot_fit=plot_fit, linestyles=linestyles,
                                          colors=colors, rebin=rebin, nolabels=True, axis=ax,
                                          fontsize=fontsize, xmin=xmin, xmax=xmax, show=False,
                                          subsample_profile=subsample_profile, npad=npad,
                                          highlight=highlight, residuals=residuals,
                                          norm_resid=norm_resid)
                lines_in_figure += LIV
                ax.tick_params(length=7, labelsize=fontsize)
                if num <= len(contents)-2:
                    # xtl = ax.get_xticklabels()
                    # print [ticklabel.get_text() for ticklabel in xtl]
                    pass
                else:
                    ax.set_xlabel("${\\rm Rel. velocity\ \ (km\ s^{-1})}$", fontsize=12)

                if num % 2 == 1:
                    ax.set_ylabel("Normalized Flux", fontsize=12)
                num += 1
                # LIV is a shorthand for 'lines_in_view'

        if filename:
            pdf.savefig(fig)

    # plt.tight_layout()
    fig.set_tight_layout(True)
    if filename:
        pdf.close()
        print "\n  Output saved to PDF file:  " + filename

    if show:
        plt.show()


def plot_single_line(dataset, line_tag, plot_fit=False, linestyles=['--'], colors=['b'],
                     loc='left', rebin=1, nolabels=False, axis=None, fontsize=12,
                     xmin=None, xmax=None, ymin=None, show=True, subsample_profile=1, npad=50,
                     residuals=False, highlight=[], norm_resid=False):
    """
    Plot a single absorption line.

    Parameters
    ----------
    dataset : :class:`dataset.DataSet`
        Instance of the :class:`dataset.DataSet` class containing the line regions

    line_tag : str
        The line tag of the line to show, e.g., 'FeII_2374'

    plot_fit : bool   [default = False]
        If `True`, the best-fit profile will be shown

    linestyles : list(linestyle)
        A list of matplotlib linestyles to show velocity components

    colors : list(colors)
        A loit of matplotlib color strings to show the velocity components
        The colors and linestyles are combined to form an iterator
        which cycles through a set of (linestyle, color).

    loc : str   [default = 'left']
        Places the line tag (right or left).

    rebin : int   [default = 1]
        Rebinning factor for the spectrum

    nolabels : bool   [default = False]
        If `True`, show the axis x- and y-labels.

    axis : matplotlib.axes.Axes_
        The plotting axis of matplotlib. If `None` is given, a new figure and axis will be created.

    fontsize : int   [default = 12]
        The fontsize of text labels.

    xmin : float
        The lower x-limit in relative velocity (km/s).
        If nothing is given, the extent of the region is used.

    xmax : float
        The upper x-limit in relative velocity (km/s).
        If nothing is given, the extent of the region is used.

    ymin : float
        The lower y-limit in normalized flux units. Default is determined from the data.

    show : bool   [default = True]
        Show the figure.

    subsample_profile : int   [default = 1]
        Subsampling factor to calculate the profile on a finer grid than the data sampling.
        By default the profile is evaluated on the same grid as the data.

    npad : int   [default = 50]
        Padding added to the synthetic profile before convolution.
        This removes end artefacts from the `FFT` routine.

    residuals : bool   [default = False]
        Add a panel above the absorption line view to show the residuals of the fit.

    norm_resid : bool   [default = False]
        Show normalized residuals.

    highlight : list(str)
        A list of `ions` (e.g., "FeII", "CIa", etc.) used to calculate a separate profile
        for this subset of ions.



    .. _matplotlib.axes.Axes: https://matplotlib.org/api/axes_api.html

    """

    if line_tag not in dataset.all_lines:
        dataset.add_line(line_tag, active=False)
        dataset.prepare_dataset()

    region = dataset.find_line(line_tag)

    x, y, err, mask = region.unpack()
    cont_err = region.cont_err
    res = region.res

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

    if plot_fit and (isinstance(dataset.best_fit, dict) or
                     isinstance(dataset.pars, dict)):

        if residuals:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('top', size='20%', pad=0., sharex=ax)

        N_pix = len(x)*2
        dx = np.diff(x)[0]
        wl_line = np.logspace(np.log10(x.min()), np.log10(x.max()), N_pix)
        pxs = np.diff(wl_line)[0] / wl_line[0] * 299792.458
        front_pad = np.arange(x.min()-50*dx, x.min(), dx)
        end_pad = np.arange(x.max(), x.max()+50*dx, dx)
        wl_line = np.concatenate([front_pad, wl_line, end_pad])
        ref_line = dataset.lines[line_tag]
        l0, f, gam = ref_line.get_properties()
        l_ref = l0*(dataset.redshift + 1)

        tau = np.zeros_like(wl_line)
        tau_hl = np.zeros_like(wl_line)
        N_highlight = 0

        if isinstance(dataset.best_fit, dict):
            params = dataset.best_fit
        else:
            params = dataset.pars

        # # Determine range in which to evaluate the profile:
        # max_logN = max([val.value for key, val in params.items() if 'logN' in key])
        # if max_logN > 19.0:
        #     velspan = 20000.*(1. + 1.0*(max_logN-19.))
        # else:
        #     velspan = 20000.

        for line in dataset.lines.values():
            if line.active:
                # Reset line properties for each element
                component_prop = itertools.product(linestyles, colors)
                component_prop = itertools.cycle(component_prop)

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
                    tau += voigt.Voigt(wl_line, l0, f, 10**logN, 1.e5*b, gam, z=z)
                    if ion in highlight:
                        tau_hl += voigt.Voigt(wl_line, l0, f, 10**logN, 1.e5*b, gam, z=z)
                        ax.axvline((l0*(z+1) - l_ref)/l_ref*299792.458,
                                   ls='-', lw=2.5, color='darkorange', alpha=0.7)
                        N_highlight += 1

                    ls, color = component_prop.next()
                    ax.axvline((l0*(z+1) - l_ref)/l_ref*299792.458, ls=ls, color=color)

        profile_int = np.exp(-tau)
        profile_int_hl = np.exp(-tau_hl)
        fwhm_instrumental = res
        sigma_instrumental = fwhm_instrumental / 2.35482 / pxs
        LSF = gaussian(len(wl_line)/2, sigma_instrumental)
        LSF = LSF/LSF.sum()
        profile_broad = fftconvolve(profile_int, LSF, 'same')
        profile_broad_hl = fftconvolve(profile_int_hl, LSF, 'same')
        profile = profile_broad[50:-50]
        profile_hl = profile_broad_hl[50:-50]
        wl_line = wl_line[50:-50]
        vel_profile = (wl_line - l_ref)/l_ref*299792.458

    vel = (x - l_ref) / l_ref * 299792.458

    if not xmin:
        xmin = -region.velspan
    if not xmax:
        xmax = region.velspan
    ax.set_xlim(xmin, xmax)
    if np.abs(xmin) > 900 or np.abs(xmax) > 900:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    if residuals and plot_fit:
        cax.set_xlim(xmin, xmax)

    view_part = (vel > xmin) * (vel < xmax)

    if not ymin:
        ymin = np.nanmin(y[view_part]) - 3.5*np.nanmedian(err[view_part])
    ymax = max(1. + 2*np.nanmedian(err[view_part]), 1.08)
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
    ax.errorbar(vel, spectrum, err, ls='', color='gray', lw=1.)
    ax.plot(vel, spectrum, color='k', drawstyle='steps-mid', lw=1.)
    ax.axhline(0., ls='--', color='0.7', lw=0.7)

    if plot_fit and (isinstance(dataset.best_fit, dict) or
                     isinstance(dataset.pars, dict)):
        ax.plot(vel_profile, profile, color='r', lw=1.0)
        if N_highlight > 0:
            ax.plot(vel_profile, profile_hl, color='orange', lw=1.0, ls='--')

        if residuals:
            p_data = np.interp(vel, vel_profile, profile)
            if norm_resid:
                masked_resid = (masked_range - p_data)/err
                resid = (spectrum - p_data)/err
            else:
                masked_resid = masked_range - p_data
                resid = spectrum - p_data

            cax.plot(vel, masked_resid, color='0.7', drawstyle='steps-mid', lw=0.9)
            cax.plot(vel, resid, color='k', drawstyle='steps-mid', lw=1.)
            if norm_resid:
                cax.axhline(3, ls=':', color='crimson', lw=0.5)
                cax.axhline(-3, ls=':', color='crimson', lw=0.5)
                res_min = 4
                res_max = -4
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
        ax.set_xlabel("${\\rm Rel. velocity}\ \ [{\\rm km\,s^{-1}}]$")
        ax.set_ylabel("${\\rm Normalized\ flux}$")

    ax.minorticks_on()
    ax.axhline(1., ls='--', color='k')
    ax.axhline(1. + cont_err, ls=':', color='gray')
    ax.axhline(1. - cont_err, ls=':', color='gray')

    # Check if the region has a predefined label or not:
    if hasattr(region, 'label'):
        if region.label == '':
            region.generate_label()
        all_trans_str = ["${\\rm "+trans.replace('_', '\ ')+"}$" for trans in lines_in_view]
        region.label = "\n".join(all_trans_str)
        line_string = region.label

    else:
        all_trans_str = ["${\\rm "+trans.replace('_', '\ ')+"}$" for trans in lines_in_view]
        line_string = "\n".join(all_trans_str)
        region.label = line_string

    if loc == 'right':
        label_x = 0.97
    elif loc == 'left':
        label_x = 0.03
    else:
        label_x = 0.03
        loc = 'left'
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


def plot_residual(dataset, line_tag, rebin=1, xmin=None, xmax=None, axis=None):
    """
    Plot residuals for the best-fit to a given absorption line.

    Parameters
    ----------
    dataset : class DataSet
        An instance of DataSet class containing the line region to plot.

    line_tag : str
        The line tag of the line to show, e.g., 'FeII_2374'

    rebin: int
        Integer factor for rebinning the spectral data.

    xmin : float
        The lower x-limit in relative velocity (km/s).
        If nothing is given, the extent of the region is used.

    xmax : float
        The upper x-limit in relative velocity (km/s).
        If nothing is given, the extent of the region is used.

    axis : matplotlib.axes.Axes_
        The plotting axis of matplotlib. If `None` is given, a new figure and axis will be created.
    """

    if line_tag not in dataset.all_lines:
        dataset.add_line(line_tag, active=False)
        dataset.prepare_dataset()

    region = dataset.find_line(line_tag)

    x, y, err, mask = region.unpack()
    # cont_err = region.cont_err
    res = region.res

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
        npad = 50
        N_pix = len(x)*3
        dx = np.diff(x)[0]
        wl_line = np.logspace(np.log10(x.min() - npad*dx), np.log10(x.max() + npad*dx), N_pix)
        pxs = np.diff(wl_line)[0] / wl_line[0] * 299792.458
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
                    tau += voigt.Voigt(wl_line, l0, f, 10**logN, 1.e5*b, gam, z=z)

        profile_int = np.exp(-tau)
        fwhm_instrumental = res
        sigma_instrumental = fwhm_instrumental / 2.35482 / pxs
        LSF = gaussian(len(wl_line), sigma_instrumental)
        LSF = LSF/LSF.sum()
        profile_broad = fftconvolve(profile_int, LSF, 'same')
        profile = profile_broad[npad:-npad]
        wl_line = wl_line[npad:-npad]

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

# ===================================================================================
#
#   Text output functions:
# --------------------------


def print_results(dataset, params, elements='all', velocity=True, systemic=0):
    """
    Print the parameters of the best-fit.

    Parameters
    ----------
    dataset : :class:`dataset.DataSet`
        An instance of the :class:`dataset.DataSet` class containing
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
    """

    if systemic:
        z_sys = systemic
    else:
        z_sys = dataset.redshift

    print "\n  Best fit parameters\n"
    # print "\t\t\t\tlog(N)\t\t\tb"
    print "\t\t\t\tb\t\t\tlog(N)"
    if elements == 'all':
        for ion in dataset.components.keys():
            lines_for_this_ion = []
            for line_tag, line in dataset.lines.items():
                if line.ion == ion and line.active:
                    lines_for_this_ion.append(line_tag)

            all_transitions = [trans.split('_')[1] for trans in sorted(lines_for_this_ion)]
            # Split list of transitions into chunks of length=4
            # join the transitions in each chunks
            # and join each chunk with 'newline'
            trans_chunks = [", ".join(sublist) for sublist in list(chunks(all_transitions, 4))]
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

            all_transitions = ", ".join([trans.split('_')[1] for trans in sorted(lines_for_this_ion)])
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
                line = " p%-2i  =  %.3e    %.3e" % (i, coeff.value, coeff.stderr)
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
    dataset : :class:`dataset.DataSet`
        An instance of the :class:`dataset.DataSet` class containing
        the definition of data and absorption lines.

    params : lmfit.Parameters_
        Output parameter dictionary, e.g., :attr:`dataset.DataSet.best_fit`.
        See lmfit_ for details.

    logNHI : float
        Column density of neutral hydrogen.

    err : float   [default = 0.1]
        Uncertainty (1-sigma) on `logNHI`.

    """

    print "\n  Metallicities\n"
    print "  log(NHI) = %.3f +/- %.3f\n" % (logNHI, err)
    logNHI = np.random.normal(logNHI, err, 10000)
    for ion in dataset.components.keys():
        element = ion[:2] if ion[1].islower() else ion[0]
        logN = []
        logN_err = []
        N_tot = []
        for par in params.keys():
            if par.find('logN') >= 0 and par.find(ion) >= 0:
                N_tot.append(params[par].value)
                if params[par].stderr < 0.9:
                    logN.append(params[par].value)
                    if params[par].stderr < 0.01:
                        logN_err.append(0.01)
                    else:
                        logN_err.append(params[par].stderr)

        ION = [np.random.normal(n, e, 10000) for n, e in zip(logN, logN_err)]
        l68, abundance, u68 = np.percentile(np.log10(np.sum(10**np.array(ION), 0)), [16, 50, 84])
        std_err = np.std(np.log10(np.sum(10**np.array(ION), 0)))

        logN_tot = np.random.normal(abundance, std_err, 10000)
        N_solar, N_solar_err = Asplund.photosphere[element]
        solar_abundance = np.random.normal(N_solar, N_solar_err, 10000)

        metal_array = logN_tot - logNHI - (solar_abundance - 12.)
        metal = np.mean(metal_array)
        metal_err = np.std(metal_array)
        print "  [%s/H] = %.3f +/- %.3f" % (element, metal, metal_err)


def print_abundance(dataset):
    """
    Print the total column densities of all species. This will sum *all*
    the components of each ion. The uncertainty on the total column density
    is calculated using random resampling within the errors of each component.
    """

    if isinstance(dataset.best_fit, dict):
        params = dataset.best_fit
        print "\n  Total Abundances\n"
        for ion in dataset.components.keys():
            # element = ion[:2] if ion[1].islower() else ion[0]
            logN = []
            logN_err = []
            N_tot = []
            for par in params.keys():
                if par.find('logN') >= 0 and par.find(ion) >= 0:
                    N_tot.append(params[par].value)
                    if params[par].stderr < 0.5:
                        logN.append(params[par].value)
                        if params[par].stderr < 0.01:
                            logN_err.append(0.01)
                        else:
                            logN_err.append(params[par].stderr)

            ION = [np.random.normal(n, e, 10000) for n, e in zip(logN, logN_err)]
            logsum = np.log10(np.sum(10**np.array(ION), 0))
            l68, abundance, u68 = np.percentile(logsum, [16, 50, 84])
            std_err = np.std(logsum)

            print "  logN(%s) = %.2f +/- %.2f" % (ion, abundance, std_err)

    else:
        print "\n [ERROR] - The dataset has not yet been fitted. No parameters found!"


def save_parameters_to_file(dataset, filename):
    """Save best-fit parameters to file."""
    with open(filename, 'w') as output:
        header = "#comp   ion   redshift               b (km/s)       log(N/cm^-2)"
        output.write(header + "\n")
        for ion in dataset.components.keys():
            for i in range(len(dataset.components[ion])):
                z = dataset.best_fit['z%i_%s' % (i, ion)]
                logN = dataset.best_fit['logN%i_%s' % (i, ion)]
                b = dataset.best_fit['b%i_%s' % (i, ion)]
                line = "%3i  %7s  %.6f %.6f    %6.2f %6.2f    %.3f %.3f" % (i, ion,
                                                                            z.value, z.stderr,
                                                                            b.value, b.stderr,
                                                                            logN.value, logN.stderr)
                output.write(line + "\n")
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
                                               region.res)
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
                                               region.res)
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
