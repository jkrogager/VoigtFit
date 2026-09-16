import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from collections import defaultdict

from VoigtFit import load_dataset
from VoigtFit.lines import Line
from VoigtFit.voigt import evaluate_optical_depth as calctau
from VoigtFit.components import (find_peaks_in_tau,
                                 sum_logN_per_group,
                                 group_component_velocities)


def group_components_from_file(dataset_filename, ions=None, p=0.01, plot=True):
    """
    Group components in velocity space based on a VoigtFit DataSet
    loaded from file. The column densities for each ion are summed
    according to the groups in optical depth. The results are written
    to a CSV table based on the dataset.name attribute.
    For more information, see :func:`VoigtFit.grouping.group_components`
    """
    ds = load_dataset(dataset_filename)
    results = group_components(ds, ions=ions, plot=plot)
    output_filename = f"logN_grouped_{ds.name}.csv"
    results.round(4)
    results.write(output_filename, overwrite=True, format='ascii.csv',
                  comment='# ')
    print(f"Wrote results to table: {output_filename}")
    return results


def group_components(ds, ions=None, p=0.01, plot=True):
    """
    Group components in velocity space based on a VoigtFit DataSet.
    The grouping is done based on peaks in the optical depth profile
    from the best fit. Only "prominent" peaks are considered
    separate groups. All components are then matched to its closest
    group in velocity space.

    Parameters
    ----------
    ds : :class:`VoigtFit.dataset.Dataset`
        DataSet object which has been fitted.

    ions : list[str]
        List of ions in the dataset to consider for the grouping.
        Use all ions by default. Prompts the user if more than one
        ionization state is present in the dataset.

    plot : bool, default = True
        If True, plot the optical depth profile and the component groups.
        The plot is saved to the "dataset.name-groupings.pdf"

    Returns
    -------
    results : astropy.table.Table
        A table of total column densities per group for all ions including
        lower and upper 1-sigma uncertainties estimated as the 16th and 84th
        percentiles of the total column density distribution assuming Gaussian
        uncertainties from the fit.
        Example:
            velocity  logN_FeII  logN_FeII_l68  logN_FeII_u68
            -9.43     15.242     +0.052         -0.052
            -4.29     13.919     +0.312         -0.308
            +2.57     14.167     +0.398         -0.390
            +11.14    14.723     +0.144         -0.142
            +29.14    14.708     +0.464         -0.533
    """

    z_sys = ds.redshift

    vmin = np.min([reg.velspan for reg in ds.regions])
    vmax = np.max([reg.velspan for reg in ds.regions])
    N = np.mean([len(reg.wl) for reg in ds.regions])

    vel = np.linspace(vmin, vmax, int(N*2))

    # Create mean optical depth profile:
    tau_all = []
    if ions is None:
        ions = list(ds.components.keys())

    # Check ionization states:
    ion_states = set()
    for ion in ions:
        if ion[1].islower():
            ion_states.add(ion[2:])
        else:
            ion_states.add(ion[1:])

    if len(ion_states) > 1:
        print("WARNING - Mixed ionization states detected:")
        print(f"{ion_states}")
        print(f"Ions in dataset: {ions}")
        print("Are you sure you want to procede? (Y/n)")
        answer = str(input())
        if answer.lower() in ['', 'y', 'yes']:
            pass
        else:
            return None

    ions_to_remove = []
    for ion in ions:
        if not ds.has_ion(ion):
            print(f"Skipping undefined ion: {ion}")
            ions_to_remove.append(ion)
            continue

        for line_tag in ds.lines_of_ion(ion):
            line = ds.lines[line_tag]
            reg, = ds.find_line(line_tag)
            v = reg.get_velocity(z_sys, line_tag)
            t = calctau(reg.wl, ds.best_fit, [line])
            norm_tau = np.nansum(t)
            if norm_tau == 0:
                continue
            t = t / norm_tau
            tau_all.append(np.interp(vel, v, t, left=0, right=0))

    for ion in ions_to_remove:
        ions.remove(ion)

    tau = np.nanmedian(tau_all, axis=0)
    peak_vel = find_peaks_in_tau(vel, tau, p=p)

    if plot:
        plt.plot(vel, tau, 'k', lw=1.0)
        plt.xlabel("Relative velocity (km/s)")
        plt.ylabel("Normalized optical depth, $\\tau\\, / \\int \\tau {\\rm d}v$")
        for t in tau_all:
            plt.plot(vel, t, lw=0.5, alpha=0.5, color='0.7')
        plt.title(ds.name + f" : {ions}")
        plt.tight_layout()

    # Loop over ions:
    results_str = {}
    results = Table()
    results['velocity'] = peak_vel
    grouped_velocities = defaultdict(list)
    print("")
    print("Total column densities in groups:")
    print("---------------------------------")
    update_groups = True
    for ion in ions:
        components = ds.components[ion]
        comp_vel = np.array([(comp.z - z_sys)/(z_sys + 1) * 299792 for comp in components])
        logN = np.array([par.value for key, par in ds.best_fit.items()
                        if 'logN' in key and key.endswith(ion)])
        logN_err = np.array([par.stderr for key, par in ds.best_fit.items()
                            if 'logN' in key and key.endswith(ion)])
        good = np.isfinite(logN_err) & (logN_err < 1)
        groups = group_component_velocities(comp_vel[good], peak_vel)
        logN_tot, l68, u68 = sum_logN_per_group(logN[good], logN_err[good], groups)
        total_str = ""
        for i, v0 in enumerate(peak_vel):
            total_str += "%+7.2f : %.3f +%.3f -%.3f\n" % (v0, logN_tot[i], u68[i], l68[i])
            if not update_groups:
                continue
            comp_color = plt.cm.gist_rainbow(i / (len(peak_vel)-1))
            for v_i in comp_vel[good][groups[i]]:
                plt.axvline(v_i, color=comp_color, ls='-', lw=1.5)
                grouped_velocities[f"{v0:+8.2f}"].append(f"{v_i:.2f}")
        results[f'logN_{ion}'] = logN_tot
        results[f'logN_{ion}_l68'] = l68
        results[f'logN_{ion}_u68'] = u68
        results_str[ion] = total_str
        print(ion)
        print(total_str)
        print("")
        update_groups = False

    results.meta['comments'] = [
            'Velocity in units of km/s',
            'Column densities in units of 1/cm^2',
    ]

    print("Velocity grouping (in km/s):")
    for num, (key, vals) in enumerate(grouped_velocities.items(), 1):
        print(f" {num}. {key}: {', '.join(vals)}")
    print("")

    if plot:
        figure_name = f"{ds.name}-groupings.pdf"
        plt.savefig(figure_name)
        print(f"Saved figure: {figure_name}")

    return results
