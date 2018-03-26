"""
This module contains definitions of line-complexes which should be defined
simulatneously. Data in this module are purely included for ease of use.

"This work has made use of the VALD database, operated at Uppsala University,
the Institute of Astronomy RAS in Moscow, and the University of Vienna."

    Ryabchikova T., Piskunov, N., Kurucz, R.L., et al.,
        Physics Scripta, vol 90, issue 5, article id. 054005 (2015), (VALD-3)
    Kupka F., Ryabchikova T.A., Piskunov N.E., Stempels H.C., Weiss W.W., 2000,
        Baltic Astronomy, vol. 9, 590-594 (2000), (VALD-2)
    Kupka F., Piskunov N.E., Ryabchikova T.A., Stempels H.C., Weiss W.W.,
        A&AS 138, 119-133 (1999), (VALD-2)
    Ryabchikova T.A. Piskunov N.E., Stempels H.C., Kupka F., Weiss W.W.
        Proc. of the 6th International Colloquium on Atomic Spectra and Oscillator Strengths,
        Victoria BC, Canada, 1998, Physica Scripta T83, 162-173 (1999), (VALD-2)
    Piskunov N.E., Kupka F., Ryabchikova T.A., Weiss W.W., Jeffery C.S.,
        A&AS 112, 525 (1995) (VALD-1)
"""
__author__ = 'Jens-Kristian Krogager'
import pickle
from os.path import dirname, abspath


def merge_two_dicts(default, x):
    """Merge the keys of dictionary `x` into dictionary `default`. """
    z = default.copy()
    z.update(x)
    return z


root_path = dirname(abspath(__file__))
CI_label_file = root_path + '/static/C_full_labels.txt'
CI_file = root_path + '/static/C_complexes.dict.pickle'

fine_structure_complexes = dict()
CI_complex = pickle.load(open(CI_file))
# Load more complexes here and merge them with the existing dictionary:
fine_structure_complexes = merge_two_dicts(fine_structure_complexes,
                                           CI_complex)
# Or define the complexes by hand, e.g.:
# fine_structure_complexes[main_tag] = [dependent_tags]

full_labels = dict()
with open(CI_label_file) as labels:
    all_lines = labels.readlines()

for line in all_lines:
    if line.strip()[0] == '#':
        continue
    else:
        tag, label = line.strip().split('\t')
        full_labels[tag.strip()] = label
