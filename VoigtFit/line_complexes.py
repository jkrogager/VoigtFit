"""
This module contains definitions of line-complexes which should be defined
simulatneously. Data in this module are purely included for ease of use.
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
