
import numpy as np


def get_ion_state(line):
    """
    Get the ionization state of a `VoigtFit.Line` instance or of `line_tag` string:
    ex: Line<'FeII_2374'>  -->  II
    ex: Line<'CIa_1656'>  -->  I
    ex: 'CIV_1550'  -->  IV
    """
    if isinstance(line, str):
        ion = line.split('_')[0]
    else:
        ion = line.ion

    if 'H2' in ion:
        return ''
    elif 'CO' in ion:
        return ''
    else:
        pass

    element = ion[:2] if ion[1].islower() else ion[0]
    length = len(element)
    ion_state = ion[length:]
    if ion_state[-1].islower():
        ion_state = ion_state[:-1]
    return ion_state


def match_ion_state(line, all_lines):
    """
    Find a line that matches the ionization state of the input `line`.
    If more lines match, then choose the strongest line.
    """
    matches = match_ion_state_all(line, all_lines)

    N_matches = len(matches)
    if N_matches == 0:
        msg = "No matches found!"
        line_match = None

    elif N_matches == 1:
        line_match = matches[0]
        msg = "Found 1 match: %s" % line_match.tag

    else:
        line_strength = [ll.l0 * ll.f for ll in matches]
        idx = np.argmax(line_strength)
        line_match = matches[idx]
        msg = "Found %i matches. Strongest line: %s" % (N_matches, line_match.tag)

    return line_match, msg


def match_ion_state_all(line, all_lines):
    """
    Find all lines that match the ionization state of the input `line`.
    """
    if isinstance(line, str):
        line_tag = line
    else:
        line_tag = line.tag

    ion_state = get_ion_state(line)
    matches = list()
    for this_line in all_lines:
        if this_line.tag == line_tag:
            continue

        this_state = get_ion_state(this_line)
        if this_state == ion_state:
            matches.append(this_line)

    return matches


def tau_percentile(x, tau, a=0.997):
    """
    Determine the range of x that encompasses the fraction `a`
    of the total apparent optical depth of the absorption profile, `tau`
    """
    y = np.cumsum(tau)
    y = y/y.max()

    a_low = (1 - a) / 2
    a_high = (1 + a) / 2

    x_range = list()
    for p in [a_low, a_high]:
        i1 = max((y < p).nonzero()[0])
        i2 = i1 + 1
        slope = (y[i2] - y[i1]) / (x[i2] - x[i1])
        x_int = x[i2] + (p - y[i2])/slope
        x_range.append(x_int)

    return x_range

# vmin, vmax = tau_noise_range(vel_ref[mask], tau, tau_err, threshold=threshold)
def tau_noise_range(x, tau, tau_err, threshold=1.5):
    """
    Determine the range of x for which the cumulative `tau` is significantly
    above the noise-level determined from the cumulative error.
    """
    y = np.cumsum(tau)
    y_err = np.sqrt(np.cumsum(tau_err**2))
    N_pix = len(tau)

    low_noise = np.median(y_err[:N_pix//2])
    upper_noise = np.median(y_err[N_pix//2:])
    y_low = threshold * low_noise
    y_high = max(y) - threshold * upper_noise

    # For the upper range:
    imax = min((y > y_high).nonzero()[0])
    xmax = x[imax]

    # For the lower range:
    imin = max((y < y_low).nonzero()[0])
    xmin = x[imin]

    return (xmin, xmax)


def equivalent_width(wl, flux, err, *, aper, z_sys=0.):
    """
    Measure equivalent width in a region determined by the aperture `aper`,
    given as a boolean array with `True` where the equivalent width should
    be evaluated.

    Returns rest-frame equivalent width and its uncertainty in same units as `wl`,
    by default in Å.
    """
    assert np.sum(aper) > 0, "Must have a non-zero number of pixels in the aperture"
    assert len(aper) == len(wl), "Aperture must have same number of elements as wl, flux, err"

    nan_values = np.isnan(flux[aper])
    if nan_values.any():
        # Skip NaN values by removing them from the aperture:
        aper = aper & ~nan_values
    W_rest = np.trapz(1. - flux[aper], wl[aper]) / (z_sys + 1)
    W_err = np.sqrt(np.nansum(err[aper]**2)) * np.mean(np.diff(wl)) / (z_sys + 1)

    return W_rest, W_err
