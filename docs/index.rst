
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
  Bo Milvang-Jensen, and Lise Christensen for their help in extensively testing
  the software and for their constructive feedback and ideas for new features.



Recent updates
--------------

New in version 0.11.7:

  Critical update of the atomic data for the hydrogen Lyman series.
  Oscillator strengths for the Lyman series were incorrectly compiled into the master linelist.

New in version 0.11.6.1:

  New data keyword `no-mask` allows the user to ignore any pixel mask present in the input data.
  VoigtFit will automatically try to identify pixel masks in the ASCII or FITS data, so if such
  automatically retrieved masks are not appropriate for the fitting, they can be ignored by setting
  this keyword in the `data` statement (see :ref:`documentation`).

New in version 0.11.6:

  Improved FITS data import of a large variety of file formats, FITS Tables, Multi Extension FITS files,
  and IRAF format.
  Updated the `systemic` and `data` statements of the parameter file definitions (see :ref:`documentation`).


New in version 0.11.5:

  New atomic data for lines of hydrogen Lyman series. New damping constants for missing levels (n > 6)
  have been calculated using data from Jitrik & Bunge (2004).
  Small bugs related to the Python 3 migration have been fixed.


New in version 0.11.4:

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
