
from numpy import loadtxt
from os.path import dirname, abspath

root_path = dirname(abspath(__file__))
atomfile = root_path + '/static/linelist.dat'

lineList = loadtxt(atomfile, dtype=[('trans', 'U13'),
                                    ('ion', 'U6'),
                                    ('l0', 'f4'),
                                    ('f', 'f8'),
                                    ('gam', 'f8'),
                                    ('mass', 'f4')])

def show_transitions(ion=None, lower=0., upper=1.e4, fine_lines=False, flim=0.):
    """
    Show the transitions defined in the atomic database.

    Parameters
    ----------
    ion : str   [default = '']
        Which ion to search for in the atomic database.

    lower : float   [default = 0.]
        The lower limit on the rest-frame wavelength of the transition.

    upper : float   [default = 0.]
        The upper limit on the rest-frame wavelength of the transition.

    fine_lines : bool   [default = False]
        If `True`, then fine-structure transistions for the given ion is included.

    flim : float  [default = 0.]
        Only return transitions whose oscillator strength is larger than flim.

    Returns
    -------
    all_lines : list(trans)
        A list of transitions. Each `transition` is taken from the atomic database,
        and contains the following indices: `l0`, `trans`, `ion`, `f`, `gam`, `mass`.
    """
    all_lines = list()
    if ion:
        # only return given ion
        for trans in lineList:
            if trans['ion'] == ion:
                if trans['l0'] > lower and trans['l0'] < upper:
                    if trans['f'] > flim:
                        all_lines.append(trans)

            elif trans['ion'][:-1] == ion and trans['ion'][-1].islower() and fine_lines is True:
                if trans['l0'] > lower and trans['l0'] < upper:
                    if trans['f'] > flim:
                        all_lines.append(trans)

    else:
        for trans in lineList:
            if trans['l0'] > lower and trans['l0'] < upper and trans['f'] > flim:
                if trans['ion'][-1].islower():
                    if fine_lines is True:
                        all_lines.append(trans)
                else:
                    all_lines.append(trans)

    return all_lines


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
            Equal to ``ion`` for backwards compatibility.

        l0 : float
            Rest-frame resonant wavelength of the transition.
            Unit: Angstrom.

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
        self.element = ion      # This is for backwards compatibility only! Does not follow the use of `element` otherwise
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

    def __repr__(self):
        """String representation of the `Line` instance"""
        line_string = "<Line: %s  %.2fÅ  f=%.2e>" % (self.ion, self.l0, self.f)
        return line_string
