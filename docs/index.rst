
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
  I also thank Johan Fynbo, Christina Thöne, Luca Izzo, Antonio de Ugarte-Postigo,
  Bo Milvang-Jensen, and Lise Christensen for their help in extensively testing
  the software and for their constructive feedback and ideas for new features.



Recent updates
--------------

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
