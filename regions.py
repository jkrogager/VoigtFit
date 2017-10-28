import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import spline


def linfunc(x, a, b):
    # Linear fitting function
    return a*x + b


class Region():
    def __init__(self, v, line):
        self.velocity_span = v
        self.lines = [line]
        self.label = ''

    def add_data_to_region(self, data_chunk, cutout):
        self.res = data_chunk['res']
        self.err = data_chunk['error'][cutout]
        self.flux = data_chunk['flux'][cutout]
        self.wl = data_chunk['wl'][cutout]
        self.normalized = data_chunk['norm']
        self.cont_err = 0.
        self.mask = np.ones_like(self.wl, dtype=bool)
        self.new_mask = True

    def has_line(self, tag):
        for line in self.lines:
            if line.tag == tag:
                return True

        return False

    def has_active_lines(self):
        active_lines = [line.active for line in self.lines]
        if np.any(active_lines):
            return True

        return False

    def remove_line(self, tag):
        if self.has_line(tag):
            for num, line in enumerate(self.lines):
                if line.tag == tag:
                    num_to_remove = num
            self.lines.pop(num_to_remove)

    def normalize(self, plot=True, norm_method='linear'):
        """
        Normalize the region if the data were not normalized.
        Choose from two methods:
            1:  define left and right continuum regions
                and fit a linear continuum.
            2:  define the continuum as a range of points
                and use spline interpolation to infer the
                continuum.
        """

        if norm_method == 'linear':
            norm_num = 1
        elif norm_method == 'spline':
            norm_num = 2
        else:
            err_msg = "Invalid norm_method: %r" % norm_method
            raise ValueError(err_msg)

        plt.close('all')

        plt.figure()
        dx = 0.1*(self.wl.max() - self.wl.min())
        plt.xlim(self.wl.min()-dx, self.wl.max()+dx)
        plt.ylim(0.8*self.flux.min(), 1.2*self.flux.max())
        plt.plot(self.wl, self.flux, color='k', drawstyle='steps-mid')
        plt.xlabel("Wavelength  [${\\rm \AA}$]")
        lines_title_string = ", ".join([line.tag for line in self.lines])
        plt.title(lines_title_string)

        if norm_num == 1:
            # - Normalize by defining a left and right continuum region

            print "\n\n  Mark continuum region 1, left and right boundary."

            bounds = plt.ginput(2, -1)
            left_bound = min(bounds[0][0], bounds[1][0])
            right_bound = max(bounds[0][0], bounds[1][0])
            region1 = (self.wl >= left_bound)*(self.wl <= right_bound)
            fit_wl = self.wl[region1]
            fit_flux = self.flux[region1]

            lines_title_string = ", ".join([line.tag for line in self.lines])
            plt.title(lines_title_string)
            print "\n  Mark continuum region 2, left and right boundary."
            bounds = plt.ginput(2)
            left_bound = min(bounds[0][0], bounds[1][0])
            right_bound = max(bounds[0][0], bounds[1][0])
            region2 = (self.wl >= left_bound)*(self.wl <= right_bound)
            fit_wl = np.concatenate([fit_wl, self.wl[region2]])
            fit_flux = np.concatenate([fit_flux, self.flux[region2]])

            popt, pcov = curve_fit(linfunc, fit_wl, fit_flux)

            continuum = linfunc(self.wl, *popt)
            e_continuum = np.std(fit_flux - linfunc(fit_wl, *popt))

            plt.close()

        elif norm_num == 2:
            # Normalize by drawing the continuum and perform spline
            # interpolation between the points

            print "\n\n  Select continuum points to fit"
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
            plt.plot(self.wl, new_flux, color='k', drawstyle='steps-mid')
            plt.xlabel("Wavelength  [${\\rm \AA}$]")
            plt.title("Normalized")
            plt.axhline(1., ls='--', color='k')
            plt.axhline(1.+e_continuum/np.mean(continuum), ls=':', color='gray')
            plt.axhline(1.-e_continuum/np.mean(continuum), ls=':', color='gray')
            plt.show(block=False)

            prompt = raw_input(" Is normalization correct?  (YES/no)")
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

    def define_mask(self, z=None, dataset=None):
        plt.close('all')

        plt.xlim(self.wl.min(), self.wl.max())
        # plt.ylim(max(0, 0.8*self.flux.min()), 1.2)
        lines_title = ", ".join([line.tag for line in self.lines])
        plt.plot(self.wl, self.flux, color='k', drawstyle='steps-mid', lw=0.5,
                 label=lines_title)
        plt.xlabel("Wavelength  [${\\rm \AA}$]")
        plt.legend()

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

        # lines_title = ", ".join([line.tag for line in self.lines])
        plt.title("Mark regions to mask, left and right boundary.")
        print "\n\n  Mark regions to mask, left and right boundary."

        ok = 0
        while ok >= 0:
            sel = plt.ginput(0, timeout=-1)

            if len(sel) > 0 and len(sel) % 2 == 0:
                mask = self.mask.copy()
                sel = np.array(sel)
                selections = np.column_stack([sel[::2, 0], sel[1::2, 0]])
                for x1, x2 in selections:
                    cutout = (self.wl >= x1)*(self.wl <= x2)
                    mask[cutout] = False
                    plt.axvline(x1, color='r', ls='--')
                    plt.axvline(x2, color='r', ls='--')

                masked_spectrum = np.ma.masked_where(mask, self.flux)
                plt.plot(self.wl, masked_spectrum, color='r', drawstyle='steps-mid')

                plt.draw()
                prompt = raw_input("Are the masked regions correct? (YES/no)")
                if prompt.lower() in ['', 'y', 'yes']:
                    ok = -1
                    self.mask = mask
                    self.new_mask = False

                else:
                    ok += 1

            elif len(sel) == 0:
                print "\nNo masks were defined."
                prompt = raw_input("Continue? (yes/no)")
                if prompt.lower() in ['', 'y', 'yes']:
                    ok = -1
                    self.new_mask = False
                else:
                    ok += 1

    def clear_mask(self):
        self.mask = np.ones_like(self.wl, dtype=bool)
        # self.new_mask = True

    def unpack(self):
        return (self.wl, self.flux, self.err, self.mask)

    def set_label(self, text):
        self.label = text

    def generate_label(self, active_only=True):
        transition_lines = list()
        if active_only:
            for line in self.lines:
                if line.active is True:
                    transition_lines.append(line.tag)
            all_trans_str = ["${\\rm "+trans.replace('_', '\ \\lambda')+"}$" for trans in transition_lines]
            line_string = "\n".join(all_trans_str)

        else:
            for line in self.lines:
                transition_lines.append(line.tag)
            all_trans_str = ["${\\rm "+trans.replace('_', '\ \\lambda')+"}$" for trans in transition_lines]
            line_string = "\n".join(all_trans_str)

        self.label = line_string
