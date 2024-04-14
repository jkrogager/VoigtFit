
.. include:: voigtfit_logo.rst

VoigtFit
=========

VoigtFit is written to make absorption line fitting a *breeze*.
Dig in, and you will be constraining the contents of metals in remote galaxies before you know it!



Development of VoigtFit happens `on GitHub
<https://github.com/jkrogager/VoigtFit>`_ so if you encounter any issues or you have
ideas for great new features, you can `raise an issue on the GitHub page
<https://github.com/jkrogager/VoigtFit/issues>`_.


Acknowledgements
----------------

If you use VoigtFit, please cite the description paper on `arXiv (Krogager 2018) <http://arxiv.org/abs/1803.01187>`_.

|

.. rst-class:: boxnote

  I want to thank Jonatan Selsing and Kasper E. Heintz
  who helped putting parts of the documentation together.
  I also thank Svea Hernandez, Johan Fynbo, Christina Thöne, Luca Izzo, Antonio de Ugarte-Postigo,
  Bo Milvang-Jensen, Lise Christensen, and Annalisa De Cia for their help in extensively testing
  the software and for their constructive feedback and ideas for new features.



Recent updates
--------------

New in version 3.21.5
  Updated the scipy.signal function calls to remove deprecated `scipy.signal.gaussian`.

New in version 3.21.4
  Added SiII* 1265 line to the line list.

New in version 3.21.3
  Small bugfixed in the fitting of relative offsets between spectra. Also updated the atomic data for CrII 2056.
  Included an icon for the documentation website.


[**New features!**] in version 3.21
  You can now define relative offsets when fitting multiple spectra simultaneously. This can correct for systematics
  in the wavelength calibrations or uncertainties in barycentri velocity corrections. The new parameters allows the user
  to define a constant offset in wavelength and/or velocity, and each term can be set to variable or kept fixed during the fit.
  The new parameters are added as keywords to the `data` statement. See more details in the :ref:`documentation`.
  This version also includes a new `mask-range` statement, which allows you to define fixed masking ranges in velocity space
  without using the interactive masking mode. Remember that masks in VoigtFit are *exclusion masks* and thus defines pixels
  that should not be fitted. See more details in the :ref:`documentation`.
  The best-fit output has also been renamed from *.fit* to *.out* in order to avoid confusion with FITS files that are sometimes
  called *.fit* as well.


[**New feature!**] since version 3.20
  The new function `vfit-lines` can be run straight from terminal. For more information run `vfit-lines --help`.
  An example: `vfit-lines FeI FeII` will show all the lines of FeI and FeII in the line list.
  You choose to print only lines in a given wavelength range: `vfit-lines FeI FeII --lower 1200 --upper 2300`.
  You can also filter on oscillator strength, to show only lines stronger than a given limit:
  `vfit-lines FeII --flim 0.01`.


New in version 3.20:
  Added new line to the line list: OI 1355. New functionality to print the line list for a specific set of ions.
  The new function `vfit-lines` can be run straight from terminal. For more information run `vfit-lines --help`.
  New `overview` statement introduced in the parameter file. This statement allows you to create a fast overview
  figure of all lines defined in the dataset and then exit the program. See more details in the :ref:`documentation`.

New in version 3.19:
  Printing the Chi2 value and number of degrees of freedom after the best-fit and also saving them to the output file.

New in version 3.18:
  Included new transitions of Fe II 2234 and 2367 using oscillator strengths of 1.29E-5 and 8.18E-5, respectively,
  based on results from Welty et al. 1999 (ApJS, 124, 465) and Miller et al. 2007 (ApJ, 659, 441).

New in version 3.17:
  Updated the oscillator strength of the 1301 transition of P II to the value determined by Brown et al. 2018, f=0.0196.

New in version 3.16:
  Updated a bug in matplotlib deprecation warnings. Added depletion parameters for future implementation of dust corrections.
  Numba is no longer supported. Using user-defined LSF files may therefore take much longer to fit now. I'm working on a work-around.

New in version 3.14.1:
  Updated the Solar abundances from Asplund et al. (2009) to use the newer values by `Asplund et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021A&A...653A.141A>`_. 

New in version 3.14:
  The plotting can now be controlled more detailed by using the *plot-options* statement in the parameter file.
  See more details in the :ref:`documentation`.

New in version 3.13.6:
  The spectral cutout for each line can now be defined as a velocity range (`vmin` and `vmax`) instead of a symmetric velocity span (`velspan=`). The new syntax can be used in the `lines`, `fine-lines`, `molecule` statements. The old syntax is still allowed. The velocity range can also be set globally using the `velspan` statement. See the full documentation on the :ref:`documentation`.
  The Voigt profile has also been reformulated in terms of frequency (and not wavelength) following the suggestions by `Webb, Carswell and Lee (2021) <https://ui.adsabs.harvard.edu/abs/2021MNRAS.tmp.2672W/abstract>`_.

New in version 3.13.5:
  One extra digit is shown for best-fit logN and total column density. New option to command line: `-V` or `--version`, will just show the welcome message with the version number and subsequently quit the program.

New in version 3.13.4:
  Small stability update in interactive normalization plot.

New in version 3.13.3.2:
  Bugfixes: loading of fine-structure lines, plotting limits, counting number of components.
  I also restructured the code and cleaned up imports (for better automated docs). Note that this may break some import statements if you are using advanced Python scripting. In that case, check the source code to find the location of the right module.


New in version 3.13.3:
  Bugfix in saving and loading datasets (.hdf5) and restoring masks and normalization.


New in version 3.13.2:
  Bugfix in masking and continuum normalization of lines used with the `limit` statement. Updated the input-parameter template to reflect the usage of `limit` and `def` introduced in versions 3.13 and 3.12.

New in version 3.13.1:
  User-defined variables can now be defined in order to define flexible parameter constraints.
  For details, see the documentation on :ref:`documentation`. Small updates in measurement of equivalent width: do not use blended lines to determine the integration limits, unless fitted profile is used. Generate plots for limits.

New in version 3.12:
  Determination of equivalent widths and upper limits is now possible using a new parameter statement `limit`.
  For more details about how to use this, see the documentation on :ref:`documentation`.
  I want to thank Annalisa De Cia for the motivation to include this functionality in VoigtFit, for helpful discussions along the way, and for the financial support to keep developing VoigtFit!

New in version 3.11.15:
  Updated oscillator strengths for NiII following Boissé & Bergeron (2019).

New in version 3.11.14:
  Small bugfixes (numba deprecation warning, output formatting, etc) and it is now possible to load components
  from a fit with one set of ions and copy those to other ions even if the imported ions are not defined in the dataset.
  This is helpful when constraining different ions in different spectra and you don't want to fit them simultaneously.

New in version 3.11.7:

  Critical update of the atomic data for the hydrogen Lyman series.
  Oscillator strengths for the Lyman series were incorrectly compiled into the master linelist.

New in version 3.11.6.1:

  New data keyword `no-mask` allows the user to ignore any pixel mask present in the input data.
  VoigtFit will automatically try to identify pixel masks in the ASCII or FITS data, so if such
  automatically retrieved masks are not appropriate for the fitting, they can be ignored by setting
  this keyword in the `data` statement (see :ref:`documentation`).

New in version 3.11.6:

  Improved FITS data import of a large variety of file formats, FITS Tables, Multi Extension FITS files,
  and IRAF format.
  Updated the `systemic` and `data` statements of the parameter file definitions (see :ref:`documentation`).


New in version 3.11.5:

  New atomic data for lines of hydrogen Lyman series. New damping constants for missing levels (n > 6)
  have been calculated using data from Jitrik & Bunge (2004).
  Small bugs related to the Python 3 migration have been fixed.


New in version 3.11.4:

  Python 3 is now supported! Solar values have been updated following the recommendations by
  Lodders et al. (2009) as to whether photospheric, meteoritic or their average value is used.


New in version 0.11.3:

  The user can now reset the masking of specific lines in a dataset by using the *force* option in the
  `mask` statement. For more details, see the `Mask` section of the :ref:`documentation`.
  The user can also include separate components for ions where the component structure is otherwise
  copied from another ion. Before, these component definitions were overwritten by the copy statement.
  The `reset` statement is now obsolete. The data are automatically reset when using the Chebyshev continuum model.
  A new verbose option can be used to print more details about the dataset, mostly useful for debugging.
  Include `-v` in the call to VoigtFit from the terminal.
  Bugfixes: The indexing of components to delete is automatically sorted so that the component structure
  is kept intact during the removal of components.
  Small bugs in the handling of fine-structure lines has been fixed by including a new `fine_lines` data
  structure in the dataset to indicate whether lines have been defined in a line complex or individually.


New in version 0.11.2:

  Technical update: Implementing tests.


New in version 0.11.1:

  Bugfixes: The Chebyshev parameters were not saved in the HDF5 dataset and they were sorted incorrectly
  as strings instead of numbers.


New in version 0.11:

  The code now checks that all lines for the defined ions have been added to the fit
  if they are covered by the data. The user can adjust this behaviour or turn it off.
  Further small bugfixes were added and the code has been prepared for python 3 migration.
  A new function to output individual components of all the best-fit profiles has been added.
  See updates in the sections `Check-Lines` and `Output` of the :ref:`documentation`.


New in version 0.10.2:

  Added support for LSF file format to specify the convolution kernel as a function of wavelength.
  See details under the `Data` section of the :ref:`documentation`.


New in version 0.9.10:

  Added two keywords to the parameter language in order to allow the user to view fit regions
  in velocity space when defining masks, continuum normalization and components.
  See details under sections `Mask`, `Continuum Normalization` and `Interactive Components` of the :ref:`documentation`.


New in version 0.9.9:

  New atomic data consistently gathered from the `VALD <http://vald.astro.uu.se/>`_ database.
  The linelist has additional lines from `Cashman et al. 2017 <https://ui.adsabs.harvard.edu/#abs/2017ApJS..230....8C/abstract>`_
  and for some lines not covered by VALD the damping constants have been taken from the compilation of
  `Morton et al. 2003 <https://ui.adsabs.harvard.edu/#abs/2003ApJS..149..205M/abstract>`_.

  When masking lines, any predefined masks in the spectral data (loaded with the add_data statement)
  are now retained.



Installation instructions
-------------------------

.. toctree::
   :maxdepth: 2

   install.rst




Documentation
-------------

.. toctree::
   :maxdepth: 2

   documentation.rst


Interface
---------

.. toctree::
   :maxdepth: 2

   api.rst


Examples
--------

.. toctree::
   :maxdepth: 2

   physical_model_results.rst
