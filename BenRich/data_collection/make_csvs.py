# This script creates two csvs for copy-pasting into our backup google doc
#   (one for energies, one for reference paths)
# User needs to change the variables "calcs_path" and "surf_name" to match the surf they want to collect data for
#   and the path to the "calcs" dir where the calculations were conducted.
# To run this script, simply copy paste this into a make_csvs.py on a machine with your calc data, change the
#   necessary variables, and run it (just `python make_csvs.py`)
# Created CSVs will be saved to your calcs directory (what you give for calcs_path)
# As per gc_manager convention, ads site directories that start with "_" (ie calcs/adsorbed/Fe/0.00V/_01)
#   will be ignored.

import numpy as np
import pandas as pd
from os import listdir, remove
from os.path import isdir, join as opj, exists as ope

##################### USER INPUTS (Change these before running) #####################
calcs_path = "D:\\scratch_backup\\perl\\amm_bin\\calcs"
surf_name = "HfP_110"
#####################################################################################


### REFERENCE LISTS (Make sure these align with our google sheet before running) ###
biases = ["0.00V", "-0.50V"]
mols_order = ["N2", "N2H", "NNH2", "NHNH2", "NH2NH2", "N", "NH", "NH2", "NH3"]
#####################################################################################

def get_ecomp_path(path):
    ecomp_path = ""
    found = False
    if ope(opj(path, "Ecomponents")):
        ecomp_path = opj(path, "Ecomponents")
        found = True
    else:
        contents = listdir(path)
        for f in contents:
            sub_dir = opj(path, f)
            if isdir(sub_dir) and ope(opj(sub_dir, "Ecomponents")):
                ecomp_path = opj(sub_dir, "Ecomponents")
                found = True
    return ecomp_path, found


def get_nrg(calc_dir, nrg_key):
    ecomp, found = get_ecomp_path(calc_dir)
    nrg = 0
    if found:
        with open(ecomp) as f:
            for line in f:
                if "=" in line:
                    key = line.split("=")[0].strip()
                    val = line.split("=")[1].strip()
                    if key == nrg_key:
                        nrg = val
    else:
        print(f"No Ecomponents found within {calc_dir}")
    return nrg

def get_lowest_ads_data(ads_dir, nrg_key):
    sites = listdir(ads_dir)
    nrgs = []
    paths = []
    for site in sites:
        if not site[0] == "_":
            site_dir = opj(ads_dir, site)
            if isdir(site_dir):
                nrgs.append(get_nrg(site_dir, nrg_key))
                paths.append(str(site_dir))
    min_nrg = min(nrgs)
    min_path = paths[nrgs.index(min_nrg)]
    return min_nrg, min_path

def save_mins_to_csvs(calcs_path, surf_name):
    s1 = len(biases) + 1
    s2 = len(mols_order) + 1
    nrgs_array = np.zeros([s1, s2], dtype = float)
    paths_array = np.array([np.array(["."*500 for k in range(s2)]) for l in range(s1)])
    surfs_dir = opj(calcs_path, opj("surfs", surf_name))
    adsorbed_dir = opj(calcs_path, opj("adsorbed", surf_name))
    nrgs_array[0, 0] = get_nrg(opj(surfs_dir, "No_bias"), "F")
    paths_array[0, 0] = opj(surfs_dir, "No_bias")
    for i, bias in enumerate(biases):
        nrgs_array[i + 1, 0] = get_nrg(opj(surfs_dir, bias), "G")
        paths_array[i + 1, 0] = opj(surfs_dir, bias)
        for j, mol in enumerate(mols_order):
            ads_dir = opj(adsorbed_dir, opj(mol, bias))
            min_nrg, min_path = get_lowest_ads_data(ads_dir, "G")
            nrgs_array[i + 1, j + 1] = min_nrg
            paths_array[i + 1, j + 1] = min_path
    for j, mol in enumerate(mols_order):
        ads_dir = opj(adsorbed_dir, opj(mol, "No_bias"))
        min_nrg, min_path = get_lowest_ads_data(ads_dir, "F")
        print(min_path)
        nrgs_array[0, j + 1] = min_nrg
        paths_array[0, j + 1] = min_path
    print(paths_array)
    nrgs_fname = opj(calcs_path, f"{surf_name}_nrgs.csv")
    paths_fname = opj(calcs_path, f"{surf_name}_paths.csv")
    for p in [nrgs_fname, paths_fname]:
        if ope(p):
            remove(p)
    nrgs = pd.DataFrame(nrgs_array)
    paths = pd.DataFrame(paths_array)
    nrgs.to_csv(nrgs_fname, index=False)
    paths.to_csv(paths_fname, index=False)


save_mins_to_csvs(calcs_path, surf_name)
