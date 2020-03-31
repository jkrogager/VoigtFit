
========
VoigtFit
========

Absorption line fitting implemented in Python.

If you use this software, please cite my paper on `arXiv <http://arxiv.org/abs/1803.01187>`_.
Please let me know that you're using VoigtFit by filling in this `short form <https://forms.gle/exPEsrPoyfB4Us7w9>`_.
This way I can keep you updated with critical updates.


Installation
============

Dependencies
------------

Python version 2.7 or >3.6 (only tested on 3.6 and 3.7 so far).

VoigtFit depends on ``matplotlib``, ``numpy``, ``scipy``, ``h5py``, ``astropy``, ``lmfit``, and ``numba``.
You can install these using your favorite Python package manager such as
`conda <http://conda.pydata.org/docs/>`_ or pip_.

Using pip
---------

The easiest way to install the most recent stable version of ``VoigtFit`` is
using pip_::

    pip install VoigtFit

|

If you encounter the following AttributeError when attempting to install via pip:

  AttributeError: 'NoneType' object has no attribute 'splitlines'

Try running pip with the ``--upgrade`` option::

    pip install --upgrade VoigtFit


From source
-----------

Alternatively, you can get the latest version of the source by cloning the git
repository::

    git clone https://github.com/jkrogager/VoigtFit.git

Once you've downloaded the source, you can navigate into the root source
directory and run::

    python setup.py install


If you encounter any problems, do not hesitate to raise an issue here.


Further documentation and how to use it
---------------------------------------

Check out the documentation_ for instructions of installation and use.

.. _pip: http://www.pip-installer.org/
.. _documentation: http://VoigtFit.readthedocs.io
