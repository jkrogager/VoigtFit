# -*- coding: UTF-8 -*-

__author__ = 'Jens-Kristian Krogager'
import re


def clean_line(line):
    """Remove comments and parentheses from the input line."""
    # strip comments:
    comment_begin = line.find('#')
    line = line[:comment_begin]
    # remove parentheses:
    line = line.replace('[', '').replace(']', '')
    line = line.replace('(', '').replace(')', '')
    return line


def parse_parameters(fname):
    """Parse parameters from input file."""
    parameters = dict()
    parameters['logNHI'] = None
    parameters['norm_method'] = 'linear'
    parameters['show_total'] = False
    parameters['plot'] = False
    parameters['resolution'] = list()
    parameters['save'] = False
    parameters['cheb_order'] = 1
    parameters['systemic'] = [None, 'none']
    parameters['clear_mask'] = False
    parameters['velspan'] = 500.
    parameters['snr'] = None
    parameters['output_pars'] = list()
    parameters['options'] = list()
    parameters['fit_options'] = {'rebin': 1, 'method': 'leastsq'}
    parameters['fix_velocity'] = False
    parameters['norm_view'] = 'wave'
    parameters['mask_view'] = 'wave'
    parameters['interactive_view'] = 'wave'
    par_file = open(fname)
    data = list()
    components = list()
    components_to_copy = list()
    components_to_delete = list()
    interactive_components = list()
    lines = list()
    fine_lines = list()
    molecules = dict()
    # fine_lines = list()
    thermal_model = list()

    for line in par_file.readlines():
        if line[0] == '#':
            continue

        elif 'data' in line and 'name' not in line and 'save' not in line:
            line = clean_line(line)
            pars = line.split()
            # get first two values:
            filename = pars[1]
            filename = filename.replace("'", "")
            filename = filename.replace('"', "")
            resolution = pars[2]
            if "'" in resolution or '"' in resolution:
                # resolution is a string => lsf-file:
                resolution = resolution.replace('"', "")
                resolution = resolution.replace("'", "")
            else:
                resolution = float(resolution)

            nsub = None
            if 'nsub=' in line:
                # find value of nsub:
                for item in pars[2:]:
                    if 'nsub=' in item:
                        nsub = int(item.split('=')[1])
                if nsub is None:
                    print("Syntax Error in line:")
                    print(line)
                    nsub = 1
            else:
                nsub = 1

            # search for 'norm' and 'air':
            norm = line.find('norm') > 0
            air = line.find('air') > 0
            airORvac = 'air' if air else 'vac'
            data.append([filename, resolution, norm, airORvac, nsub])

        elif 'lines' in line and 'save' not in line and 'fine' not in line:
            velspan = None
            # strip comments:
            comment_begin = line.find('#')
            line = line[:comment_begin].strip()
            if 'velspan' in line:
                idx = line.find('velspan')
                value = line[idx:].split('=')[1]
                velspan = float(value)

                linelist = line.split()
                linelist = linelist[1:-1]
                all_lines = [[l, velspan] for l in linelist]

            else:
                linelist = line.split()[1:]
                all_lines = [[l, velspan] for l in linelist]

            lines += all_lines

        elif 'fine-lines' in line:
            velspan = None
            line = clean_line(line)
            if 'velspan' in line:
                idx = line.find('velspan')
                value = line[idx:].split('=')[1]
                velspan = float(value)
                line = line[:idx]

            pars = line.split()
            ground_state = pars[1]
            if len(pars) > 2:
                levels = pars[2:]
            else:
                levels = None

            fine_lines.append([ground_state, levels, velspan])

        elif 'molecule' in line:
            velspan = None
            Jmax = 1
            # Ex.  add molecule CO AX(1-0), AX(0-0) [J=1 velspan=150]
            # strip comments:
            comment_begin = line.find('#')
            line = line[:comment_begin].strip()
            for item in line.split():
                if '=' in item:
                    parname, parval = item.split('=')
                    if parname.strip().upper() == 'J':
                        Jmax = int(parval)
                    elif 'velspan' in parname.lower():
                        velspan = float(parval)
                    idx = line.find(item)
                    if idx > 0:
                        line = line[:idx]

            if 'CO' in line:
                CO_begin = line.find('CO')
                band_string = line[CO_begin:].replace(',', '')
                bands = band_string.split()[1:]
                if 'CO' in molecules.keys():
                    for band in bands:
                        molecules['CO'] += [band, Jmax, velspan]
                else:
                    molecules['CO'] = list()
                    for band in bands:
                        molecules['CO'] += [band, Jmax, velspan]

            elif 'H2' in line:
                H2_begin = line.find('H2')
                band_string = line[H2_begin:].replace(',', '')
                bands = band_string.split()[1:]
                if 'H2' in molecules.keys():
                    for band in bands:
                        molecules['H2'] += [band, Jmax, velspan]
                else:
                    molecules['H2'] = list()
                    for band in bands:
                        molecules['H2'] += [band, Jmax, velspan]

            else:
                print "\n [ERROR] - Could not detect any molecular species to add!\n"

        elif 'component' in line and 'copy' not in line and 'delete' not in line:
            # strip comments:
            comment_begin = line.find('#')
            line = line[:comment_begin].strip()
            # remove parentheses:
            line = line.replace('[', '').replace(']', '')
            line = line.replace('(', '').replace(')', '')
            parlist = line.split()[1:]
            # parlist = ['FeII', 'z=2.2453', 'b=12.4', 'logN=14.3']
            ion = parlist[0]
            var_z, var_b, var_N = True, True, True
            tie_z, tie_b, tie_N = None, None, None
            if '=' in line:
                for num, val in enumerate(parlist[1:]):
                    if 'z=' in val and '_' not in val:
                        par, value = val.split('=')
                        z = float(value)
                    elif 'b=' in val and '_' not in val:
                        par, value = val.split('=')
                        b = float(value)
                    elif 'logN=' in val:
                        par, value = val.split('=')
                        logN = float(value)
                    elif 'var_z=' in val:
                        par, value = val.split('=')
                        if value.lower() == 'false':
                            var_z = False
                        elif value.lower() == 'true':
                            var_z = True
                        else:
                            var_z = bool(value)
                    elif 'var_b=' in val:
                        par, value = val.split('=')
                        if value.lower() == 'false':
                            var_b = False
                        elif value.lower() == 'true':
                            var_b = True
                        else:
                            var_b = bool(value)
                    elif 'var_N=' in val:
                        par, value = val.split('=')
                        if value.lower() == 'false':
                            var_N = False
                        elif value.lower() == 'true':
                            var_N = True
                        else:
                            var_N = bool(value)
                    elif 'tie_z=' in val:
                        par, value = val.split('=')
                        tie_z = value
                    elif 'tie_b=' in val:
                        par, value = val.split('=')
                        tie_b = value
                    elif 'tie_N=' in val:
                        par, value = val.split('=')
                        tie_N = value
                    elif '=' not in val:
                        if num == 0:
                            z = float(val)
                        elif num == 1:
                            b = float(val)
                        elif num == 2:
                            logN = float(val)

            else:
                z = float(parlist[1])
                b = float(parlist[2])
                logN = float(parlist[3])

            if 'velocity' in line.lower():
                vel = True
            else:
                vel = False

            if 'thermal' in line.lower():
                thermal = True
            else:
                thermal = False

            components.append([ion, z, b, logN, var_z, var_b, var_N, tie_z, tie_b, tie_N, vel, thermal])

        elif 'copy' in line:
            # strip comments:
            comment_begin = line.find('#')
            line = line[:comment_begin].strip()
            # remove parentheses:
            line = line.replace('[', '').replace(']', '')
            line = line.replace('(', '').replace(')', '')
            # find ion:
            to = line.find('to')
            if to > 0:
                ion = line[to:].split()[1]
            # find anchor:
            idx = line.find('from')
            if idx > 0:
                anchor = line[idx:].split()[1]

            logN_scale = 0.
            ref_comp = 0
            if 'scale' in line:
                numbers = re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", line)
                if len(numbers) == 2:
                    logN_scale = float(numbers[0])
                    ref_comp = int(numbers[1])

            tie_z, tie_b = True, True
            if 'tie_z' in line:
                idx = line.find('tie_z=')
                value = line[idx:].split()[0].split('=')[1]
                if value.lower() == 'false':
                    tie_z = False
                else:
                    tie_z = True

            if 'tie_b' in line:
                idx = line.find('tie_b=')
                value = line[idx:].split()[0].split('=')[1]
                if value.lower() == 'false':
                    tie_b = False
                else:
                    tie_b = True

            components_to_copy.append([ion, anchor, logN_scale, ref_comp, tie_z, tie_b])

        elif 'delete' in line:
            # strip comments:
            comment_begin = line.find('#')
            line = line[:comment_begin].strip()
            # find ion:
            idx = line.find('from')
            if idx > 0:
                ion = line[idx:].split()[1]
            else:
                ion = line.split()[-1]

            number = re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", line)
            if len(number) == 1:
                comp = int(number[0])

            components_to_delete.append([ion, comp])

        elif 'interact' in line and 'save' not in line and 'view' not in line:
            # strip comments:
            comment_begin = line.find('#')
            line = line[:comment_begin].strip()
            line = line.replace(',', '')
            par_list = line.split()[1:]
            interactive_components += par_list

        elif 'name' in line:
            comment_begin = line.find('#')
            line = line[:comment_begin].strip()
            parameters['name'] = line.split(':')[-1].strip()

        elif 'z_sys' in line:
            comment_begin = line.find('#')
            line = line[:comment_begin].strip()
            if ':' in line:
                parameters['z_sys'] = float(line.split(':')[-1].strip())
            elif '=' in line:
                parameters['z_sys'] = float(line.split('=')[-1].strip())
            else:
                parameters['z_sys'] = float(line.split()[-1].strip())

        elif 'norm_method' in line:
            comment_begin = line.find('#')
            line = line[:comment_begin].strip()
            line = line.replace("'", "")
            parameters['norm_method'] = line.split(':')[-1].strip()

        elif 'clear mask' in line.lower():
            parameters['clear_mask'] = True

        elif ('mask' in line and
              'name' not in line and
              'save' not in line and
              'nomask' not in line and
              'view' not in line):
            comment_begin = line.find('#')
            line = line[:comment_begin].strip()
            line = line.replace(',', '')
            items = line.split()[1:]
            if 'mask' in parameters.keys():
                parameters['mask'] += items
            else:
                parameters['mask'] = items

        elif 'resolution' in line and 'name' not in line and 'save' not in line:
            comment_begin = line.find('#')
            line = line[:comment_begin].strip()
            items = line.split()
            if len(items) == 3 and items[0] == 'resolution':
                res = float(items[1])
                line = items[2]
            elif len(items) == 2 and items[0] == 'resolution':
                res = float(items[1])
                line = None
            parameters['resolution'].append([res, line])

        elif 'metallicity' in line and 'name' not in line and 'save' not in line:
            numbers = re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", line)
            if len(numbers) == 2:
                logNHI = [float(n) for n in numbers]
            elif len(numbers) == 1:
                logNHI = [float(numbers[0]), 0.1]
            else:
                print " Error - In order to print metallicities you must give log(NHI)."
            parameters['logNHI'] = logNHI

        elif 'fit-options' in line and 'name' not in line and 'save' not in line:
            comment_begin = line.find('#')
            line = line[:comment_begin].strip()
            items = line.split()[1:]
            # here you can add keywords that will be passed to the fit and minimizer from lmfit.
            # ex: rebin, ftol, xtol, method, etc.
            fit_keywords = dict()
            for item in items:
                key, val = item.split('=')
                if "'" in val or '"' in val:
                    val = val.replace('"', '')
                    val = val.replace("'", '')
                    fit_keywords[key] = val
                elif val.lower() == 'true':
                    fit_keywords[key] = True
                elif val.lower() == 'false':
                    fit_keywords[key] = False
                elif '.' in val:
                    fit_keywords[key] = float(val)
                else:
                    fit_keywords[key] = int(val)
            parameters['fit_options'] = fit_keywords

        elif 'output' in line and 'name' not in line and 'save' not in line:
            comment_begin = line.find('#')
            line = line[:comment_begin].strip()
            items = line.split()[1:]
            # here you can add keywords like 'velocity' to print velocities instead of redshift
            # 'individual-regions' saves individual regions to separate files
            parameters['output_pars'] = items

        elif 'save' in line and 'name' not in line:
            parameters['save'] = True
            # strip comments:
            comment_begin = line.find('#')
            line = line[:comment_begin].strip()
            if 'filename' in line:
                filename_begin = line.find('filename')
                line = line[filename_begin:].strip()
                if '=' in line:
                    filename = line.split('=')[1]
                elif ':' in line:
                    filename = line.split(':')[1]
                else:
                    filename = line.split()[1]
            else:
                filename = None
            parameters['filename'] = filename

        elif 'total' in line and 'name' not in line and 'save' not in line:
            parameters['show_total'] = True

        elif 'signal-to-noise' in line and 'name' not in line and 'save' not in line:
            comment_begin = line.find('#')
            line = line[:comment_begin].strip()
            if '=' in line:
                snr = line.split('=')[1]
            elif ' ' in line:
                snr = line.split()[1]
            elif ':' in line:
                snr = line.split(':')[1]
            parameters['snr'] = float(snr)

        elif ('velspan' in line and 'lines' not in line and
              'molecules' not in line and 'save' not in line):
            # strip comments:
            comment_begin = line.find('#')
            line = line[:comment_begin].strip()
            if '=' in line:
                velspan = line.split('=')[1]
            elif ':' in line:
                velspan = line.split(':')[1]
            else:
                velspan = line.split()[1]
            parameters['velspan'] = float(velspan)

        elif 'C_order' in line and 'name' not in line and 'save' not in line:
            # strip comments:
            comment_begin = line.find('#')
            line = line[:comment_begin].strip()
            if '=' in line:
                order = line.split('=')[1]
            elif ':' in line:
                order = line.split(':')[1]
            else:
                order = line.split()[1]
            parameters['cheb_order'] = int(order)

        elif 'cheb_order' in line and 'name' not in line and 'save' not in line:
            # strip comments:
            comment_begin = line.find('#')
            line = line[:comment_begin].strip()
            order = line.split('=')[1]
            parameters['cheb_order'] = int(order)

        elif 'systemic' in line and 'name' not in line and 'save' not in line:
            # strip comments:
            comment_begin = line.find('#')
            line = line[:comment_begin].strip()
            # remove parentheses
            line = line.replace('[', '').replace(']', '')
            line = line.replace('(', '').replace(')', '')
            mode = line.split('=')[1]
            if ',' in mode:
                # num, ion mode:
                num, ion = mode.split(',')
                parameters['systemic'] = [int(num), ion]
            else:
                # either none or auto:
                if "'" in mode:
                    mode = mode.replace("'", '')
                    parameters['systemic'] = [None, mode]

        elif 'reset' in line and 'name' not in line and 'save' not in line:
            comment_begin = line.find('#')
            line = line[:comment_begin].strip()
            line = line.replace(',', '')
            items = line.split()[1:]
            if 'reset' in parameters.keys():
                parameters['reset'] += items
            else:
                parameters['reset'] = items

        elif 'load' in line and 'name' not in line and 'save' not in line:
            comment_begin = line.find('#')
            line = line[:comment_begin].strip()
            line = line.replace('"', '')
            line = line.replace("'", '')
            filenames = line.split()[1:]
            if 'load' in parameters.keys():
                parameters['load'] += filenames
            else:
                parameters['load'] = filenames

        elif 'thermal model' in line.lower():
            comment_begin = line.find('#')
            line = line[:comment_begin].strip()
            line = line.replace('"', '')
            line = line.replace("'", '')
            ions = list()
            pars = line.split()
            fix_T = False
            fix_turb = False
            T_init = None
            turb_init = None
            for par in pars[2:]:
                if 'T=' in par:
                    T_init = float(par.split('=')[1])
                elif 'turb=' in par:
                    turb_init = float(par.split('=')[1])
                elif 'fix-T' in par:
                    fix_T = True
                elif 'fix-turb' in par:
                    fix_turb = True
                else:
                    ions.append(par)

            if T_init is None:
                print(" Invalid Thermal Model!")
                print("You should give an initial temperature, e.g., T=500")

            if turb_init is None:
                print(" Invalid Thermal Model!")
                print("You should give an initial turbulent broadening, e.g., turb=5")

            thermal_model = [ions, T_init, turb_init, fix_T, fix_turb]

        elif 'fix-velocity' in line.lower():
            parameters['fix_velocity'] = True

        elif 'norm_view' in line.lower():
            key, value = line.split(':')
            if key.strip().lower() == 'norm_view':
                parameters['norm_view'] = value.strip().lower()

        elif 'mask_view' in line.lower():
            line = clean_line(line)
            key, value = line.split(':')
            if key.strip().lower() == 'mask_view':
                parameters['mask_view'] = value.strip().lower()

        elif 'interactive_view' in line.lower():
            line = clean_line(line)
            key, value = line.split(':')
            if key.strip().lower() == 'interactive_view':
                parameters['interactive_view'] = value.strip().lower()

        else:
            pass

    par_file.close()
    parameters['data'] = data
    parameters['lines'] = lines
    parameters['fine-lines'] = fine_lines
    parameters['molecules'] = molecules
    parameters['components'] = components
    parameters['components_to_copy'] = components_to_copy
    parameters['components_to_delete'] = components_to_delete
    parameters['thermal_model'] = thermal_model
    parameters['interactive'] = interactive_components

    return parameters
