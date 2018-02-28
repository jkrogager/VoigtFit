
.. include:: voigtfit_logo.rst

.. _install:

Installation
============

VoigtFit is currently only written and tested for Python 2.7;
However, the code is currently being ported to Python 3.6 -- Stay tuned!

Dependencies
------------

VoigtFit depends on matplotlib_, numpy_, scipy_, h5py_, astropy_, and lmfit_. You
can install these using your favorite Python package manager such as
`pip <https://pip.pypa.io/en/stable/installing/>`_ or
`conda <http://conda.pydata.org/docs/>`_.


Using pip
---------

The easiest way to install the most recent stable version of ``VoigtFit`` is
using `pip <http://www.pip-installer.org/>`_:

.. code-block:: bash

    %] pip install VoigtFit


From source
-----------

Alternatively, you can get the source by downloading a `tarball
<https://github.com/jkrogager/VoigtFit/tarball/master>`_ or cloning `the git
repository <https://github.com/jkrogager/VoigtFit>`_:

.. code-block:: bash

    %] git clone https://github.com/jkrogager/VoigtFit.git

Once you've downloaded the source, you can navigate to the root
directory and run:

.. code-block:: bash

    %] python setup.py install




.. _numpy: http://www.numpy.org/
.. _scipy: https://scipy.org/
.. _matplotlib: https://matplotlib.org/
.. _lmfit: https://lmfit.github.io/lmfit-py/
.. _astropy: http://www.astropy.org/
.. _h5py: http://www.h5py.org/


Test the installation
---------------------

If the installation went smoothly, you should be able to run VoigtFit from the terminal
by excecuting the following command:

.. code-block:: bash

    %] VoigtFit

Running the program without any input will create an empty parameter file template.
This way you can always set up a fresh parameter file when starting a new project.
Moreover, as the program grows and more features are implemented, a comment about
such new features will appear automatically in the parameter file template.
This way you can stay updated on what is possible within the parameter file language.

To run the program with a given input file, simply excecute the command:

.. code-block:: bash

    %] VoigtFit  input.pars

Where input.pars is the name of your parameter file.
