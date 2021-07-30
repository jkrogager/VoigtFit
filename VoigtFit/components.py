# -*- coding: UTF-8 -*-
"""
Module for the Component class used to define individual velocity components
for the overall line profile for the ions.
"""

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
