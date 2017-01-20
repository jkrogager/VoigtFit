import re


def parse_parameters(fname):
    parameters = dict()
    parameters['logNHI'] = None
    parameters['norm_method'] = 'linear'
    parameters['nomask'] = False
    par_file = open(fname)
    data = list()
    components = list()
    components_to_copy = list()
    components_to_delete = list()
    lines = list()

    for line in par_file.readlines():
        if line[0] == '#':
            pass

        elif 'data' in line and 'name' not in line:
            # strip comments:
            comment_begin = line.find('#')
            line = line[:comment_begin]
            # remove parentheses:
            line = line.replace('[', '').replace(']', '')
            line = line.replace('(', '').replace(')', '')
            pars = line.split()
            # get first two values:
            filename = pars[1]
            filename = filename.replace("'", "")
            filename = filename.replace('"', "")
            resolution = float(pars[2])
            # search for 'norm' and 'air':
            norm = line.find('norm') > 0
            air = line.find('air') > 0
            airORvac = 'air' if air else 'vac'
            data.append([filename, resolution, norm, airORvac])

        elif 'lines' in line:
            velspan = 500.
            # strip comments:
            comment_begin = line.find('#')
            line = line[:comment_begin].strip()
            # remove parentheses:
            if 'span' in line:
                idx = line.find('span')
                value = line[idx:].split('=')[1]
                if ',' in value:
                    value = value.replace('[', '').replace(']', '')
                    value = value.replace('(', '').replace(')', '')
                    v1, v2 = value.split(',')
                    velspan = [float(v1), float(v2)]
                else:
                    velspan = float(value)

                linelist = line.split()
                linelist = linelist[1:-1]
                all_lines = [[l, velspan] for l in linelist]

            else:
                linelist = line.split()[1:]
                all_lines = [[l, velspan] for l in linelist]

            lines += all_lines

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
            tie_z, tie_b = None, None
            if '=' in line:
                for val in parlist[1:]:
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
                        var_z = bool(value)
                    elif 'var_b=' in val:
                        par, value = val.split('=')
                        var_b = bool(value)
                    elif 'var_N=' in val:
                        par, value = val.split('=')
                        var_N = bool(value)
                    elif 'tie_z=' in val:
                        par, value = val.split('=')
                        tie_z = value
                    elif 'tie_b=' in val:
                        par, value = val.split('=')
                        tie_b = value

            else:
                z = float(parlist[1])
                b = float(parlist[2])
                logN = float(parlist[3])

            components.append([ion, z, b, logN, var_z, var_b, var_N, tie_z, tie_b])

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

        elif 'name' in line:
            parameters['name'] = line.split(':')[-1].strip()

        elif 'z_sys' in line:
            parameters['z_sys'] = float(line.split(':')[-1].strip())

        elif 'norm_method' in line:
            parameters['norm_method'] = line.split(':')[-1].strip()

        elif 'nomask' in line and 'name' not in line:
            parameters['nomask'] = True

        elif 'metallicity' in line and 'name' not in line:
            numbers = re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", line)
            if len(numbers) == 2:
                logNHI = [float(n) for n in numbers]
            elif len(numbers) == 1:
                logNHI = [float(numbers[0]), 0.1]
            else:
                print " Error - In order to print metallicities you must give log(NHI)."
            parameters['logNHI'] = logNHI

        else:
            pass

    par_file.close()
    parameters['data'] = data
    parameters['lines'] = lines
    parameters['components'] = components
    parameters['components_to_copy'] = components_to_copy
    parameters['components_to_delete'] = components_to_delete

    return parameters
