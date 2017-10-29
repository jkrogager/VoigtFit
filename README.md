# Python requirements

Python version 2.7

 ## Standard Packages
- Numpy
- Matplotlib
- Scipy


 ## LmFit
https://lmfit.github.io/lmfit-py/

If you have pip installed, you can install lmfit with:

    %] pip install lmfit

or you can download the source kit, unpack it and install with:

    %] python setup.py install

 ## H5py
http://www.h5py.org/

If you have pip installed, you can install h5py with:

    %] pip install h5py

Or follow the instruction on the webpage.


# Setting up the environment:

Add the ATOMPATH environment variable. This variable points to the location of the atomic data file 'atomdata_updated.dat' located in the 'static' directory.

in bash:

    export ATOMPATH='path/to/atomdata_updated.dat'

or in csh:

    setenv ATOMPATH path/to/atomdata_updated.dat


Add the SOLARPATH environment variable. This variable points to the location of the Solar abundance data file (from Asplund et al. 2009), 'Asplund2009.dat', located in the 'static' directory.

in bash:

    export SOLARPATH='path/to/Asplund2009.dat'

or in csh:

    setenv SOLARPATH path/to/Asplund2009.dat


### To Do

 - Show sky-lines/telluric in mask
