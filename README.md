# Python requirements

Python version 2.7 (Python 3 is not supported yet)

 ## Standard Packages
- Numpy
- Matplotlib
- Scipy


 ## PyFITS / Astropy
https://pypi.python.org/pypi/pyfits/3.3
http://www.astropy.org/

PyFITS can be installed individually, but the package is now mainly provided through the astropy
package. Either one of these packages will work. These are available on pip:

    %] pip install pyfits
    %] pip install astropy

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

First up, you can check which shell type you are using by typing 'echo $0' in your terminal.
In order to run VoigtFit from any location, you should add the directory which contains VoigtFit to your PYTHONPATH environment variable:

in bash:
    export PYTHONPATH='${PYTHONPATH}:/path/to/folder/with/VoigtFit'

or in csh:
    setenv PYTHONPATH $PYTHONPATH\:/path/to/folder/with/VoigtFit


The software needs two data files in order to run. These are distributed along with the software under the directory 'static/'. The code needs the absolute paths to these two files. The easiest way is to locate the absolute path to these files and create system variables which the software will recognize.

Add the VFITDATA environment variable. This variable points to the absolute path of the atomic data file ('atomdata_updated.dat') and the solar abundance file ('Asplund2009.dat') located in the 'static' directory:

in bash:

    export VFITDATA='/path/to/VoigtFit/static'

or in csh:

    setenv VFITDATA path/to/VoigtFit/static


  An example:

    If you are using bash shell and the code for VoigtFit is located in the folder
    '/home/user/coding/VoigtFit', then the path to the atomdata_updated.dat and
    Asplund2009.dat files will be:

    export VFITDATA='/home/user/coding/VoigtFit/static/'


### To Do

 - Show sky-lines/telluric in mask
