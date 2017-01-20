import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def linfunc(x, a, b):
	# Linear fitting function
	return a*x + b


class Region():
	def __init__(self, v, line):
		self.velocity_span = v
		self.lines = [line]

	def add_data_to_region(self, data_chunk, cutout):
		self.res = data_chunk['res']
		self.err = data_chunk['error'][cutout]
		self.flux = data_chunk['flux'][cutout]
		self.wl	  = data_chunk['wl'][cutout]
		self.normalized	= data_chunk['norm']
		self.cont_err	= 0.
		self.mask		= np.ones_like(self.wl, dtype=bool)
		self.new_mask   = True

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

	def normalize(self, plot=True, norm_method=1):
		"""
		Normalize the region if the data were not normalized.
		Choose from two methods:
			1: 	define left and right continuum regions
				and fit a linear continuum.
			2:	define the continuum as a range of points
				and use spline interpolation to infer the
				continuum.
		"""

		plt.close('all')

		plt.figure()
		plt.xlim(self.wl.min()-10, self.wl.max()+10)
		plt.ylim(0.8*self.flux.min(), 1.2*self.flux.max())
		plt.plot(self.wl, self.flux, color='k', drawstyle='steps-mid')
		plt.xlabel("Wavelength  [${\\rm \AA}$]")
		lines_title_string = ", ".join([line.tag for line in self.lines])
		plt.title(lines_title_string)

		if not norm_method:
			print "\n\n  Choose normalization method:"
			print "   1: linear (left, right)"
			print "   2: spline to points"
			print ""
			norm_method = int(raw_input("Method number: "))

		if norm_method == 1:
		### Normalize by defining a left and right continuum region

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

		elif norm_method == 2:
		### Normalize by drawing the continuum and perform spline
		### interpolation between the points
			from scipy.interpolate import spline

			print "\n\n  Select continuum points to fit"
			points = plt.ginput(n=-1, timeout=-1)
			xk, yk = [],[]
			for x,y in points:
				xk.append(x)
				yk.append(y)
			xk = np.array(xk)
			yk = np.array(yk)
			region_wl = self.wl.copy()
			continuum = spline(xk, yk, region_wl, order=3)
			e_continuum = np.sqrt(np.mean(self.err**2))


		if plot:
			plt.cla()
			plt.plot(self.wl, self.flux/continuum, color='k', drawstyle='steps-mid')
			plt.xlabel("Wavelength  [${\\rm \AA}$]")
			plt.title("Normalized")
			plt.axhline(1., ls='--', color='k')
			plt.axhline(1.+e_continuum/np.mean(continuum), ls=':', color='gray')
			plt.axhline(1.-e_continuum/np.mean(continuum), ls=':', color='gray')
			plt.show(block=False)

			prompt = raw_input(" Is normalization correct?  (yes/no)")
			if prompt.lower() in ['', 'y','yes']:
				self.flux = self.flux/continuum
				self.err  = self.err/continuum
				self.cont_err = e_continuum/np.mean(continuum)
				self.normalized = True
				return 1

			else:
				return 0

		else:
			self.flux = self.flux/continuum
			self.err  = self.err/continuum
			self.cont_err = e_continuum/np.mean(continuum)
			self.normalized = True
			return 1


	def define_mask(self):
		plt.close('all')

		plt.xlim(self.wl.min(), self.wl.max())
		plt.ylim(max(0, 0.8*self.flux.min()), 1.2)
		plt.plot(self.wl, self.flux, color='k', drawstyle='steps-mid')
		plt.xlabel("Wavelength  [${\\rm \AA}$]")
		lines_title = ", ".join([line.tag for line in self.lines])
		plt.title(lines_title)
		print "\n\n  Mark regions to mask, left and right boundary."

		ok = 0
		while ok>=0:
			sel = plt.ginput(0, timeout=-1)

			if len(sel)>0 and len(sel)%2 == 0:
				mask = self.mask.copy()
				sel = np.array(sel)
				selections = np.column_stack([sel[::2,0], sel[1::2,0]])
				for x1,x2 in selections:
					cutout = (self.wl >= x1)*(self.wl <= x2)
					mask[cutout] = False
					plt.axvline(x1, color='r', ls='--')
					plt.axvline(x2, color='r', ls='--')

				masked_spectrum = np.ma.masked_where(mask, self.flux)
				plt.plot(self.wl, masked_spectrum, color='r', drawstyle='steps-mid')

				plt.draw()
				prompt = raw_input("Are the masked regions correct? (yes/no)")
				if prompt.lower() in ['', 'y', 'yes']:
					ok = -1
					self.mask = mask
					self.new_mask = False

				else:
					ok += 1

			elif len(sel)==0:
				print "\nNo masks were defined."
				prompt = raw_input("Continue? (yes/no)")
				if prompt.lower() in ['', 'y', 'yes']:
					ok = -1
					self.new_mask = False
				else:
					ok += 1

	def clear_mask(self):
		self.mask = np.ones_like(self.wl, dtype=bool)
		#self.new_mask = True

	def unpack(self):
		return (self.wl, self.flux, self.err, self.mask)
