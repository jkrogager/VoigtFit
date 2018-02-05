import numpy as np
"""
This module contains definitions of line-complexes which should be defined simulatneously.
Data in this module are purely included for ease of use.
"""
__author__ = 'Jens-Kristian Krogager'

fine_structure_complexes = dict()

CI_full_labels = {
    'CI_1656': '${\\rm C\,\i}\ ^3{\\rm P} \\rightarrow 2s^22p3s\ ^3{\\rm P}\ (\\lambda1656)$',
    'CI_1560': '${\\rm C\,\i}\ ^3{\\rm P} \\rightarrow 2s2p^3\ ^3{\\rm D}\ (\\lambda1560)$',
    'CI_1328': '${\\rm C\,\i}\ ^3{\\rm P} \\rightarrow 2s2p^3\ ^3{\\rm P}\ (\\lambda1328)$',
    'CI_1280': '${\\rm C\,\i}\ ^3{\\rm P} \\rightarrow 2s^22p4s\ ^3{\\rm P}\ (\\lambda1280)$',
    'CI_1277': '${\\rm C\,\i}\ ^3{\\rm P} \\rightarrow 2s^22s3d\ ^3{\\rm D}\ (\\lambda1277)$',
    'CI_1276': '${\\rm C\,\i}\ ^3{\\rm P} \\rightarrow 2s^22p4s\ ^1{\\rm P}\ (\\lambda1276)$'
}

CI_labels = {'CI_1656': '${\\rm CI\ \\lambda1656}$',
             'CI_1560': '${\\rm CI\ \\lambda1650}$',
             'CI_1328': '${\\rm CI\ \\lambda1328}$',
             'CI_1280': '${\\rm CI\ \\lambda1280}$',
             'CI_1277': '${\\rm CI\ \\lambda1277}$',
             'CI_1276': '${\\rm CI\ \\lambda1276}$'}

# - CI 1656 complex
fine_structure_complexes['CI_1656'] = ['CIa_1656',
                                       'CI_1656',
                                       'CIb_1657',
                                       'CIa_1657',
                                       'CIa_1657.9',
                                       'CIb_1658']

# - CI 1560 complex
fine_structure_complexes['CI_1560'] = ['CI_1560',
                                       'CIa_1560',
                                       'CIa_1560.7',
                                       'CIb_1561',
                                       'CIb_1561.3',
                                       'CIb_1561.4']

# - CI 1328 complex
fine_structure_complexes['CI_1328'] = ['CI_1328',
                                       'CIa_1329',
                                       'CIa_1329.1',
                                       'CIa_1329.12',
                                       'CIb_1329',
                                       'CIb_1329.6']

# - CI 1280 complex
fine_structure_complexes['CI_1280'] = ['CIa_1279',
                                       'CI_1280',
                                       'CIb_1280',
                                       'CIa_1280',
                                       'CIa_1280.5',
                                       'CIb_1280.8']

# - CI 1277 complex
fine_structure_complexes['CI_1277'] = ['CI_1277',
                                       'CIa_1277',
                                       'CIa_1277.5',
                                       'CIb_1277',
                                       'CIb_1277.7',
                                       'CIb_1277.9',
                                       'CIb_1279.2',
                                       'CIb_1279.5']

# - CI 1276 complex
fine_structure_complexes['CI_1276'] = ['CI_1276',
                                       'CIa_1276',
                                       'CIb_1277.2']

# ---
CO_full_labels = {
    'AX(0-0)': "${\\rm CO\ A}^1\\Pi(0) \\leftarrow {\\rm X}^1\\Sigma^+(\\nu=0)$",
    'AX(1-0)': "${\\rm CO\ A}^1\\Pi(1) \\leftarrow {\\rm X}^1\\Sigma^+(\\nu=0)$",
    'AX(2-0)': "${\\rm CO\ A}^1\\Pi(2) \\leftarrow {\\rm X}^1\\Sigma^+(\\nu=0)$",
    'AX(3-0)': "${\\rm CO\ A}^1\\Pi(3) \\leftarrow {\\rm X}^1\\Sigma^+(\\nu=0)$",
    'AX(4-0)': "${\\rm CO\ A}^1\\Pi(4) \\leftarrow {\\rm X}^1\\Sigma^+(\\nu=0)$",
    'AX(5-0)': "${\\rm CO\ A}^1\\Pi(5) \\leftarrow {\\rm X}^1\\Sigma^+(\\nu=0)$",
    'AX(6-0)': "${\\rm CO\ A}^1\\Pi(6) \\leftarrow {\\rm X}^1\\Sigma^+(\\nu=0)$",
    'AX(7-0)': "${\\rm CO\ A}^1\\Pi(7) \\leftarrow {\\rm X}^1\\Sigma^+(\\nu=0)$",
    'AX(8-0)': "${\\rm CO\ A}^1\\Pi(8) \\leftarrow {\\rm X}^1\\Sigma^+(\\nu=0)$",
    'AX(9-0)': "${\\rm CO\ A}^1\\Pi(9) \\leftarrow {\\rm X}^1\\Sigma^+(\\nu=0)$",
    'AX(10-0)': "${\\rm CO\ A}^1\\Pi(10) \\leftarrow {\\rm X}^1\\Sigma^+(\\nu=0)$",
    'AX(11-0)': "${\\rm CO\ A}^1\\Pi(11) \\leftarrow {\\rm X}^1\\Sigma^+(\\nu=0)$",
    'CX(0-0)': "${\\rm CO\ C}^1\\Sigma(0) \\leftarrow {\\rm X}^1\\Sigma^+(\\nu=0)$",
    'dX(5-0)': "${\\rm CO\ d}^3\\Delta(5) \\leftarrow {\\rm X}^1\\Sigma^+(\\nu=0)$",
    'eX(1-0)': "${\\rm CO\ e}^3\\Sigma^-(1) \\leftarrow {\\rm X}^1\\Sigma^+(\\nu=0)$"
}

CO_labels = {'COJ0_1544.44': 'AX(0-0)',
             'COJ0_1509.74': 'AX(1-0)',
             'COJ0_1477.56': 'AX(2-0)',
             'COJ0_1447.35': 'AX(3-0)',
             'COJ0_1419.04': 'AX(4-0)',
             'COJ0_1392.52': 'AX(5-0)',
             'COJ0_1367.62': 'AX(6-0)',
             'COJ0_1344.18': 'AX(7-0)',
             'COJ0_1322.15': 'AX(8-0)',
             'COJ0_1301.40': 'AX(9-0)',
             'COJ0_1281.86': 'AX(10-0)',
             'COJ0_1263.43': 'AX(11-0)',
             'COJ0_1087.86': 'CX(0-0)',
             'COJ0_1510.34': 'dX(5-0)',
             'COJ0_1543.17': 'eX(1-0)',
             'COJ0_1543.00': 'eX(1-0)'
             }

CO = {
    # Nu=0:
    'AX(0-0)': [['COJ0_1544.44'],  # J=0
                ['COJ1_1544.54', 'COJ1_1544.38'],  # J=1
                ['COJ2_1544.72', 'COJ2_1544.57', 'COJ2_1544.34'],  # J=2
                ['COJ3_1544.84', 'COJ3_1544.61', 'COJ3_1544.31'],  # J=3
                ['COJ4_1544.98', 'COJ4_1544.68', 'COJ4_1544.30'],  # J=4
                ['COJ5_1545.14', 'COJ5_1544.76', 'COJ5_1544.31']],   # J=5
    # Nu=1:
    'AX(1-0)': [['COJ0_1509.74'],
                ['COJ1_1509.83', 'COJ1_1509.69'],
                ['COJ2_1510.01', 'COJ2_1509.87', 'COJ2_1509.66'],
                ['COJ3_1510.13', 'COJ3_1509.92', 'COJ3_1509.64'],
                ['COJ4_1510.27', 'COJ4_1509.99', 'COJ4_1509.64']],
    # Nu=2:
    'AX(2-0)': [['COJ0_1477.56'],
                ['COJ1_1477.64', 'COJ1_1477.51'],
                ['COJ2_1477.81', 'COJ2_1477.68', 'COJ2_1477.47'],
                ['COJ3_1477.93', 'COJ3_1477.72', 'COJ3_1477.45'],
                ['COJ4_1478.06', 'COJ4_1477.79', 'COJ4_1477.45']],
    # Nu=3:
    'AX(3-0)': [['COJ0_1447.35'],
                ['COJ1_1447.43', 'COJ1_1447.30'],
                ['COJ2_1447.59', 'COJ2_1447.46', 'COJ2_1447.27'],
                ['COJ3_1447.70', 'COJ3_1447.51', 'COJ3_1447.25'],
                ['COJ4_1447.83', 'COJ4_1447.58', 'COJ4_1447.25']],
    # Nu=4:
    'AX(4-0)': [['COJ0_1419.04'],
                ['COJ1_1419.12', 'COJ1_1419.00'],
                ['COJ2_1419.27', 'COJ2_1419.15', 'COJ2_1418.97'],
                ['COJ3_1419.38', 'COJ3_1419.20', 'COJ3_1418.96'],
                ['COJ4_1419.51', 'COJ4_1419.27', 'COJ4_1418.97']],
    # Nu=5:
    'AX(5-0)': [['COJ0_1392.52'],
                ['COJ1_1392.60', 'COJ1_1392.48'],
                ['COJ2_1392.74', 'COJ2_1392.63', 'COJ2_1392.46'],
                ['COJ3_1392.85', 'COJ3_1392.68', 'COJ3_1392.45'],
                ['COJ4_1392.98', 'COJ4_1392.75', 'COJ4_1392.46']],
    # Nu=6:
    'AX(6-0)': [['COJ0_1367.62'],
                ['COJ1_1367.69', 'COJ1_1367.58'],
                ['COJ2_1367.83', 'COJ2_1367.73', 'COJ2_1367.56'],
                ['COJ3_1367.94', 'COJ3_1367.78', 'COJ3_1367.56'],
                ['COJ4_1368.07', 'COJ4_1367.85', 'COJ4_1367.58'],
                ['COJ5_1368.21', 'COJ5_1367.94', 'COJ5_1367.61']],
    # Nu=7:
    'AX(7-0)': [['COJ0_1344.18'],
                ['COJ1_1344.25', 'COJ1_1344.15'],
                ['COJ2_1344.39', 'COJ2_1344.29', 'COJ2_1344.13'],
                ['COJ3_1344.49', 'COJ3_1344.34', 'COJ3_1344.13'],
                ['COJ4_1344.62', 'COJ4_1344.41', 'COJ4_1344.15'],
                ['COJ5_1344.76', 'COJ5_1344.49', 'COJ5_1344.18']],
    # Nu=8:
    'AX(8-0)': [['COJ0_1322.15'],
                ['COJ1_1322.21', 'COJ1_1322.11'],
                ['COJ2_1322.35', 'COJ2_1322.25', 'COJ2_1322.10'],
                ['COJ3_1322.45', 'COJ3_1322.30', 'COJ3_1322.10'],
                ['COJ4_1322.57', 'COJ4_1322.37', 'COJ4_1322.13'],
                ['COJ5_1322.71', 'COJ5_1322.46', 'COJ5_1322.17']],
    # Nu=9:
    'AX(9-0)': [['COJ0_1301.40'],
                ['COJ1_1301.46', 'COJ1_1301.37'],
                ['COJ2_1301.59', 'COJ2_1301.50', 'COJ2_1301.36'],
                ['COJ3_1301.70', 'COJ3_1301.55', 'COJ3_1301.37'],
                ['COJ4_1301.82', 'COJ4_1301.63', 'COJ4_1301.39'],
                ['COJ5_1301.95', 'COJ5_1301.72', 'COJ5_1301.43']],
    # Nu=10:
    'AX(10-0)': [['COJ0_1281.86'],
                 ['COJ1_1281.92', 'COJ1_1281.83'],
                 ['COJ2_1282.05', 'COJ2_1281.96', 'COJ2_1281.83'],
                 ['COJ3_1282.15', 'COJ3_1282.02', 'COJ3_1281.84'],
                 ['COJ4_1282.27', 'COJ4_1282.09', 'COJ4_1281.84'],
                 ['COJ5_1282.40', 'COJ5_1282.18', 'COJ5_1281.91']],
    # Nu=11:
    'AX(11-0)': [['COJ0_1263.43'],
                 ['COJ1_1263.49', 'COJ1_1263.40'],
                 ['COJ2_1263.61', 'COJ2_1263.53', 'COJ2_1263.40'],
                 ['COJ3_1263.71', 'COJ3_1263.58', 'COJ3_1263.41'],
                 ['COJ4_1263.83', 'COJ4_1263.66', 'COJ4_1263.44'],
                 ['COJ5_1263.96', 'COJ5_1263.75', 'COJ5_1263.49']],

    'eX(1-0)': [['COJ0_1543.17', 'COJ0_1543.00'],
                ['COJ1_1543.20', 'COJ1_1542.91', 'COJ1_1543.17'],
                ['COJ2_1543.26', 'COJ2_1543.44', 'COJ2_1542.85', 'COJ2_1543.27', 'COJ2_1543.23'],
                ['COJ3_1543.35', 'COJ3_1543.66', 'COJ3_1542.83', 'COJ3_1543.37', 'COJ3_1543.33'],
                ['COJ4_1543.48', 'COJ4_1543.90', 'COJ4_1542.83', 'COJ4_1543.49', 'COJ4_1543.45']],

    'CX(0-0)': [['COJ0_1087.86'],
                ['COJ1_1087.95', 'COJ1_1087.82'],
                ['COJ2_1088.00', 'COJ2_1087.77'],
                ['COJ3_1088.04', 'COJ3_1087.72'],
                ['COJ4_1088.09', 'COJ4_1087.67']],

    'dX(5-0)': [['COJ0_1510.34'],
                ['COJ1_1510.42', 'COJ1_1510.30'],
                ['COJ2_1510.60', 'COJ2_1510.48', 'COJ2_1510.29'],
                ['COJ3_1510.74', 'COJ3_1510.56', 'COJ3_1510.31'],
                ['COJ4_1510.91', 'COJ4_1510.66', 'COJ4_1510.36']]
}


# --- Rotatinal Constants in units of cm^-1
#     E = hc * B * J(J + 1)
rotational_constant = {'H2': 60.853,
                       'CO': 1.9313,
                       'HD': 45.655}

hc = 1.2398e-4         # eV.cm
k_B = 8.6173e-5        # eV/K


def population_of_level(element, T, J):
    """
    Calculate the population of the Jth level relative to the J=0 level.
    The distribution is assumed to be an isothermal Boltzmann distribution:

    n(J) \propto g(J) e^(-E(J) / kT)
    """
    if element not in rotational_constant.keys():
        print " Element is not in database! "
        print " All elements in database are: " + ", ".join(rotational_constant.keys())
        return -1
    else:
        # convert rotational constant to units of Kelvin:
        B = rotational_constant[element]
        B = B * hc / k_B
        n_J = (2*J + 1) * np.exp(-B*J*(J+1)/T)
        return n_J
