
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

Python version >3.6 (tested on 3.6 - 3.9 so far).

VoigtFit depends on ``matplotlib``, ``numpy==1.20.3``, ``scipy``, ``h5py``, ``astropy``, ``lmfit``, and ``numba==0.55.0``.
You can install these using your favorite Python package manager such as
`conda <http://conda.pydata.org/docs/>`_ or pip_.

If you encounter issues with the installation of ``h5py`` make sure that you have the HDF5 library installed. If you use Homebrew as a package manager you can try this fix [following `stackoverflow <https://stackoverflow.com/questions/66741778/how-to-install-h5py-needed-for-keras-on-macos-with-m1>`_]::

    brew install hdf5
    export HDF5_DIR="$(brew --prefix hdf5)"
    pip install --no-binary=h5py h5py

If you encounter issues with the installation of ``numba`` (happened for me on Mac M1 architecture), try the following::

    arch -arm64 brew install llvm@11
    LLVM_CONFIG="/opt/homebrew/Cellar/llvm@11/11.1.0_4/bin/llvm-config" arch -arm64 pip install llvmlite

before installing `numba`.


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
