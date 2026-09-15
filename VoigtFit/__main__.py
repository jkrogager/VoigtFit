
from VoigtFit.main import main, __version__
from VoigtFit.lines import show_transitions
from VoigtFit.grouping import group_components_from_file

from astropy.table import Table
from argparse import ArgumentParser


def print_credits():
    print(r"")
    print(r"       VoigtFit  %s                    " % __version__)
    print(r"      JK Krogager 2018                 ")
    print(r"  ____  _           ___________________")
    print(r"      \/ \  _/\    /                   ")
    print(r"          \/   \  / oigtFit            ")
    print(r"                \/                     ")
    print(r"")


def print_linelist():
    print_credits()

    parser = ArgumentParser(description="VoigtFit Line List Tool.")
    parser.add_argument("ion", type=str, nargs='+',
                        help="List of ions to query (ex: CI CII CIV)")
    parser.add_argument("-l", "--lower", type=float, default=0.,
                        help="The lower limit on the rest-frame wavelength of the transition.")
    parser.add_argument("-u", "--upper", type=float, default=1.e4,
                        help="The upper limit on the rest-frame wavelength of the transition.")
    parser.add_argument("-f", "--flim", type=float, default=0.,
                        help="Return transitions whose oscillator strength is larger than flim.")
    parser.add_argument("--fine", action="store_true",
                        help="Include fine-structure transistions for the given ion if set.")

    args = parser.parse_args()

    lines = []
    for ion in args.ion:
        lines += show_transitions(ion=ion, lower=args.lower, upper=args.upper,
                                  fine_lines=args.fine, flim=args.flim)
    if len(lines) == 0:
        print(f"No lines found for ions: {', '.join(args.ion)}")
        return

    tab = Table(rows=lines, names=['tag', 'ion', 'l0', 'f', 'gamma', 'mass'])
    tab['l0'].format = '%9.3f'
    tab['f'].format = '%.2e'
    tab['gamma'].format = '%.3e'
    tab.remove_column('ion')
    tab.pprint(max_lines=-1)


def run_grouping():
    print_credits()

    parser = ArgumentParser(description="VoigtFit Component Grouping Tool")
    parser.add_argument("filename", type=str,
                        help="Filename of the .hdf5 DataSet from VoigtFit")
    parser.add_argument("ions", type=str, nargs='*',
                        help="List of ions to include in the grouping, ex: FeII SiII ZnII")
    parser.add_argument("--plot", action="store_true",
                        help="Plot the optical depth profile?")

    args = parser.parse_args()
    if len(args.ions) == 0:
        ions = None
    else:
        ions = args.ions
    group_components_from_file(args.filename, ions=ions, plot=args.plot)


if __name__ == '__main__':
    main()
