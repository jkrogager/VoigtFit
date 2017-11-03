.. _install:

Installation
============

Dependencies
------------

VoigtFit depends on ``matplotlib``, ``numpy``, ``scipy``, ``h5py``, ``astropy``, and ``lmfit``. You
can install these using your favorite Python package manager and I would
recommend `conda <http://conda.pydata.org/docs/>`_ if you don't already have
an opinion.

Using pip
---------

The easiest way to install the most recent stable version of ``VoigtFit`` is
with `pip <http://www.pip-installer.org/>`_:

.. code-block:: bash

    pip install VoigtFit


From source
-----------

Alternatively, you can get the source by downloading a `tarball
<https://github.com/jkrogager/VoigtFit/tarball/master>`_ or cloning `the git
repository <https://github.com/jkrogager/VoigtFit>`_:

.. code-block:: bash

    git clone https://github.com/jkrogager/VoigtFit.git

Once you've downloaded the source, you can navigate into the root source
directory and run:

.. code-block:: bash

    python setup.py install

