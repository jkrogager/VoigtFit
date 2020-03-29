# -*- coding: UTF-8 -*-
"""
Module for the Component class used to define individual velocity components
for the overall line profile for the ions.
"""

class Component(object):
    def __init__(self, z, b, logN, var_z=True, var_b=True, var_N=True, tie_z=None, tie_b=None, tie_N=None):
        self.z = z
        self.b = b
        self.logN = logN
        self.options = {'var_z': var_z, 'var_b': var_b, 'var_N': var_N,
                        'tie_z': tie_z, 'tie_b': tie_b, 'tie_N': tie_N}

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
        self.options[key] = value

    def get_option(self, key):
        return self.options[key]

    def get_pars(self):
        return [self.z, self.b, self.logN]
