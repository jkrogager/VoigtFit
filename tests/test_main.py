import os
from pathlib import Path

import VoigtFit as vfit


here = Path(os.path.dirname(os.path.abspath(__file__)))
test_data_dir = here.parent / 'test_data'
os.chdir(test_data_dir)
# PARFILE = os.path.join(str(test_data_dir), 'test_input_noint.pars')
PARFILE = 'test_input_noint.pars'

class InputArgs:
    def __init__(self, parfile, force=True, verbose=False, version=False):
        self.input = parfile
        self.f = force
        self.version = version
        self.v = verbose


output_files = ['testdata_noint.cont',
                'testdata_noint.hdf5',
                'testdata_noint.out',
                'testdata_noint.reg']

def test_input_parfile():
    args = InputArgs(PARFILE)
    vfit.run_voigtfit(args, testing=True)
    for fname in output_files:
        assert os.path.exists(fname)
        os.remove(fname)
