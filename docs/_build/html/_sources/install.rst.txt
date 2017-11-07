.. _install:

Installation
============

VoigtFit is currently only written and tested for Python 2.7; However, the code is currently being ported to Python 3.6 -- Stay tuned!

Dependencies
------------

VoigtFit depends on matplotlib_, numpy_, scipy_, h5py_, astropy_, and lmfit_. You
can install these using your favorite Python package manager such as `pip <https://pip.pypa.io/en/stable/installing/>`_ or
`conda <http://conda.pydata.org/docs/>`_.

Using pip
---------

The easiest way to install the most recent stable version of ``VoigtFit`` is
using `pip <http://www.pip-installer.org/>`_:

.. code-block:: bash

    pip install VoigtFit


From source
-----------

Alternatively, you can get the source by downloading a `tarball
<https://github.com/jkrogager/VoigtFit/tarball/master>`_ or cloning `the git
repository <https://github.com/jkrogager/VoigtFit>`_:

.. code-block:: bash

    git clone https://github.com/jkrogager/VoigtFit.git

Once you've downloaded the source, you can navigate to the root
directory and run:

.. code-block:: bash

    python setup.py install




.. _numpy: http://www.numpy.org/
.. _scipy: https://scipy.org/
.. _matplotlib: https://matplotlib.org/
.. _lmfit: https://lmfit.github.io/lmfit-py/
.. _astropy: http://www.astropy.org/
.. _h5py: http://www.h5py.org/


Setting up the environment
--------------------------

First up, you can check which shell type you are using by typing ``echo $0`` in your terminal.
In order to run ``VoigtFit`` from any location, you should add the directory which contains VoigtFit to your ``PYTHONPATH`` environment variable:

in bash:

.. code-block:: bash

    export PYTHONPATH='${PYTHONPATH}:/path/to/folder/with/VoigtFit'

or in csh:

.. code-block:: csh

    setenv PYTHONPATH $PYTHONPATH\:/path/to/folder/with/VoigtFit


The software needs two data files in order to run. These are distributed along with the software under the directory ``static/``.
The code needs the absolute paths to these two files. The easiest way is to locate the absolute path to these files and create
system variables which the software will recognize.

Add the ``VFITDATA`` environment variable. This variable points to the absolute path of the ``static`` directory:

in bash:

.. code-block:: bash

    export VFITDATA='/path/to/VoigtFit/static'

or in csh:

.. code-block:: csh

    setenv VFITDATA path/to/VoigtFit/static

