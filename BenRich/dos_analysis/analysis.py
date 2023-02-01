import user_fns as fns
import os
import numpy as np

bottom_atoms = [
    3, 7, 11, 15, 19, 23, 27, 31, 35
]
top_atoms = [
    2, 6, 10, 14, 18, 22, 26, 30, 34
]
surface_atoms = bottom_atoms + top_atoms

all_atoms = list(np.arange(1, 37, dtype=int))

# Ranges of where density appears in No_bias calculations
ranges = [
    [-4.3, -4.275],
    [-2.74, -2.7],
    [-0.35, -0.2]
]

path = 'C:\\Users\\User\\Desktop\\backup[s\\1-19-2023\\img_intrx\\calcs\\surfs'
os.chdir(path)

file_Z12_None = 'Cu_Ap2-5_Bp2-5_Cp1-5_Z12\\No_bias\\dosUp'
file_Z20_None = 'Cu_Ap2-5_Bp2-5_Cp1-5_Z20\\No_bias\\dosUp'
file_Z28_None = 'Cu_Ap2-5_Bp2-5_Cp1-5_Z28\\No_bias\\dosUp'
file_Z12_0V = 'Cu_Ap2-5_Bp2-5_Cp1-5_Z12\\No_bias\\dosUp'
file_Z20_0V = 'Cu_Ap2-5_Bp2-5_Cp1-5_Z20\\No_bias\\dosUp'
file_Z28_0V = 'Cu_Ap2-5_Bp2-5_Cp1-5_Z28\\No_bias\\dosUp'

paths = [
    file_Z12_None,
    file_Z20_None,
    file_Z28_None,
    file_Z12_0V,
    file_Z20_0V,
    file_Z28_0V,
]

data, headers = fns.get_data(paths)


"""
    :param criteria_sets: Will use shape to dictate how many figures to create, along with how many data sets to plot
    on each graph
            criteria_sets[i] = [fname, [specs], xlims]

            specs = [[crit_00, crit_01, ..., crit_0n, graph_title],
                     [crit_10, crit_11, ..., crit_1n, graph_title],
                     ...
                     [crit_m0, crit_m1, ..., crit_mn, graph_title]]
            crit_ij = [legend_name, run_idx, orbitals, atom_ids]
"""

crit_1 = ['Tot DOS, No bias', 0, ['Total'], all_atoms]
crit_2 = ['Tot DOS, 0V', 3, ['Total'], all_atoms]

crit_3 = ['Tot DOS, No bias', 1, ['Total'], all_atoms]
crit_4 = ['Tot DOS, 0V', 4, ['Total'], all_atoms]

crit_5 = ['Tot DOS, No bias', 2, ['Total'], all_atoms]
crit_6 = ['Tot DOS, 0V', 5, ['Total'], all_atoms]

specs_1 = [[crit_1, crit_2, 'Z = 12A'],
           [crit_3, crit_4, 'Z = 20A'],
           [crit_5, crit_6, 'Z = 28A']]

criteria_1 = ['tot_up', #fname
              specs_1,#specs
              ranges[-1]]

criteria_sets = [
    criteria_1
]

save_dir = 'C:\\Users\\User\\PycharmProjects\\Musgrave_scripts\\BenRich\\dos_analysis\\local_data'

fns.plot_criteria_sets_2(data, headers, criteria_sets, save_dir)

