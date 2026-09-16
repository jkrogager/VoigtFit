# -*- coding: UTF-8 -*-
"""
Module for the Component class used to define individual velocity components
for the overall line profile for the ions.
"""

import numpy as np
from scipy.signal import find_peaks
from numpy import abs
from lmfit import Parameters


class Component(object):
    def __init__(self, z, b, logN, var_z=True, var_b=True, var_N=True, tie_z=None, tie_b=None, tie_N=None):
        """
        Component object which contains the parameters for each velocity component
        of an absorption profile: redshift, z; broadning parameter, b; and column density
        in log(cm^-2).
        Options can control whether the components parameters are variable during the fit
        or whether they are tied to another parameter using the `var_` and `tie_` options.
        """
        self._z = z
        self._b = b
        self._logN = logN
        self.options = {'var_z': var_z, 'var_b': var_b, 'var_N': var_N,
                        'tie_z': tie_z, 'tie_b': tie_b, 'tie_N': tie_N}

    @property
    def z(self):
        return self._z

    @property
    def b(self):
        return self._b

    @property
    def logN(self):
        return self._logN

    @z.setter
    def z(self, val):
        self._z = val

    @b.setter
    def b(self, val):
        if val < 0:
            print(" WARNING - Negative b is non-physical! Converting to absolute value.")
            self._b = abs(val)
        else:
            self._b = val

    @logN.setter
    def logN(self, val):
        self._logN = val

    @property
    def tie_z(self):
        return self.options['tie_z']

    @property
    def tie_b(self):
        return self.options['tie_b']

    @property
    def tie_N(self):
        return self.options['tie_N']

    @property
    def var_z(self):
        return self.options['var_z']

    @property
    def var_b(self):
        return self.options['var_b']

    @property
    def var_N(self):
        return self.options['var_N']

    @tie_z.setter
    def tie_z(self, val):
        self.options['tie_z'] = val

    @tie_b.setter
    def tie_b(self, val):
        self.options['tie_b'] = val

    @tie_N.setter
    def tie_N(self, val):
        self.options['tie_N'] = val

    @var_z.setter
    def var_z(self, val):
        self.options['var_z'] = val

    @var_b.setter
    def var_b(self, val):
        self.options['var_b'] = val

    @var_N.setter
    def var_N(self, val):
        self.options['var_N'] = val

    def set_option(self, key, value):
        """
        .set_option(key, value)

        Set the `value` for a given option, must be either `tie_` or `var_`.
        """
        self.options[key] = value

    def get_option(self, key):
        """
        .get_option(key)

        Return the `value` for the given option, key must be either `tie_` or `var_`.
        """
        return self.options[key]

    def get_pars(self):
        """Unpack the physical parameters [z, b, logN]"""
        return [self.z, self.b, self.logN]

    def __repr__(self):
        """String representation of the :class:`Component <VoigtFit.container.components.Component>` instance"""
        line_string = "<Component: z=%.5f  b=%.1f  logN=%.1f>" % (self.z, self.b, self.logN)
        return line_string


def load_components_from_file(fname):
    """
    Load best-fit parameters from an output file, ex: 'dataset.fit'

    Parameters
    ----------
    fname : str
        The filename of the VoigtFit output file.

    Returns
    -------
    pars : lmfit.Parameters
        Parameter dictionary of `lmfit.Parameter` instances.
    """
    components_to_add = list()
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

    pars = Parameters()
    for comp_pars in components_to_add:
        (num, ion, z, b, logN, z_err, b_err, logN_err) = comp_pars
        ion = ion.replace('*', 'x')
        z_name = 'z%i_%s' % (num, ion)
        b_name = 'b%i_%s' % (num, ion)
        N_name = 'logN%i_%s' % (num, ion)

        pars.add(z_name, value=z)
        pars.add(b_name, value=b)
        pars.add(N_name, value=logN)

        pars[z_name].stderr = z_err
        pars[b_name].stderr = b_err
        pars[N_name].stderr = logN_err

    return pars


def components_from_array(ion, *, z, b, logN):
    """
    Create a `lmfit.Parameters` dictionary for a given `ion`
    and arrays/lists of redshift (z), broadening parameter (b) and column density (logN).
    A component will be generated for each element in the z, b, logN arrays.
    """
    pars = Parameters()
    for num, vals in enumerate(zip(z, b, logN)):
        z_name = 'z%i_%s' % (num, ion)
        b_name = 'b%i_%s' % (num, ion)
        N_name = 'logN%i_%s' % (num, ion)

        pars.add(z_name, value=vals[0])
        pars.add(b_name, value=vals[1])
        pars.add(N_name, value=vals[2])
    return pars


def find_peaks_in_tau(vel, tau, p=0.01):
    """
    Find peaks in optical depth for a given threshold in peak prominence
    See the documentation for `scipy.signal.find_peaks`:
    docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

    Parameters
    ----------
    vel : np.ndarray
        Array of line-of-sight velocity for the optical depth model
    tau: np.ndarray
        Array of optical depth for each pixel, corresponding to `vel`
    p : float
        Prominence threshold for peak detection. See scipy documentation

    Returns
    -------
    np.ndarray
        Array of velocity centroid for each peak in optical depth.
    """
    prominence = np.nanmax(tau)*p
    peaks, _ = find_peaks(tau, prominence=prominence)
    return vel[peaks]


def group_component_velocities(component_vel, group_vel):
    """
    Assign components into groups based on component velocity.

    Parameters
    ----------
    component_vel : np.ndarray
        The relative line-of-sight velocities of each component.

    group_vel : np.ndarray
        The group central velocity, must be same units as component_vel.

    Returns
    -------
    groups : list[list[int]]
        A list of lists, where each sub-list holds the indeces of the
        components that belong to a given group defined by the entries
        in `group_vel`.
    """
    groups = []
    for _ in group_vel:
        groups.append([])

    for num, v0 in enumerate(component_vel):
        index = np.argmin(np.abs(group_vel - v0))
        groups[index].append(num)
    
    if any([len(group) == 0 for group in groups]):
        print("WARNING - some groups have no components!")

    return groups


def sum_logN_per_group(logN, logN_err, groups):
    """
    Add column densities for components belonging to a group

    Parameters
    ----------
    logN : np.ndarray
        Array of log10 of column densities of all fit components.

    logN_err : np.ndarray
        Array of uncertainties on log10 of column densities

    groups : list[list[int]]
        Grouping of component indeces as a list of lists of integers.
        Each sublist holds the indeces corresponding to components of the fit.
        Example:
            groups = [[0, 1], [2, 3, 4], [5], [6]]
        This means that the first two components are grouped together,
        the following three components are grouped together, and the last
        two components are individual groups.

    Returns
    -------
    logN_tot : np.ndarray
        The log10 of the total column density for each group.
    l68 : np.ndarray
        The lower 1-sigma uncertainty on logN_tot
    u68 : np.ndarray
        The uppwer 1-sigma uncertainty on logN_tot
    """
    logN_tot = []
    l68 = []
    u68 = []
    for group in groups:
        if len(group) == 0:
            logN_tot.append(np.nan)
            l68.append(np.nan)
            u68.append(np.nan)
            continue
        logN_pdf = [np.random.normal(n, e, 10000)
                    for n, e in zip(logN[group], logN_err[group])]
        logsum = np.log10(np.sum(10**np.array(logN_pdf), 0))
        lower, total_logN, upper = np.percentile(logsum, [16, 50, 84])
        logN_tot.append(total_logN)
        l68.append(np.abs(total_logN - lower))
        u68.append(np.abs(total_logN - upper))
    return np.array(logN_tot), np.array(l68), np.array(u68)
