from os import listdir, getcwd, chdir
from os.path import isdir, join as opj, exists as ope
from shutil import copy as cp
from subprocess import run


aidx_key = "atom number"
el_key = "atomic symbol"
charge_key = "net_charge"

# def parse_ddec_output(calc_dir):
#     charges_data_dict = parse_ddec6_net_atomic_charges(calc_dir)


def get_net_atomic_charges_data_dict(calc_dir):
    data_dict = parse_ddec6_net_atomic_charges_get_data_dict(calc_dir)
    data_dict = format_data_dict_data_types(data_dict)
    return data_dict

def format_data_dict_data_types(data_dict):
    for akey in data_dict:
        for hv in data_dict[akey]:
            if "atom number" in hv:
                data_dict[akey][hv] = int(data_dict[akey][hv])
            elif "atomic symbol" in hv:
                pass
            else:
                data_dict[akey][hv] = float(data_dict[akey][hv])
    return data_dict


def parse_ddec6_net_atomic_charges_get_data_dict(calc_dir):
    rep_key_1 = "three eigenvalues of traceless quadrupole moment tensor"
    rep_key_2 = "Rsquared"
    header_1, data_1_lines, header_2, data_2_lines = parse_ddec6_net_atomic_charges_get_lines(calc_dir)
    header_1 = [v.strip() for v in header_1.split(",")]
    header_2 = [v.strip() for v in header_2.split(",")]
    data_1_lines = [data_1_line.split() for data_1_line in data_1_lines]
    data_2_lines = [data_2_line.split() for data_2_line in data_2_lines]
    if rep_key_1 in header_1[-1]:
        header_1[-1] = "Q-eig-1"
        for i in range(2):
            header_1.append(f"Q-eig-{i+2}")
    if rep_key_2 in header_2[-1]:
        header_2[-1] = "R2"
    data_dict = {}
    for d1l in data_1_lines:
        atom_key = d1l[0]
        data_dict = append_data_dict(data_dict, header_1, d1l, atom_key)
    for d2l in data_2_lines:
        atom_key = d2l[0]
        data_dict = append_data_dict(data_dict, header_2, d2l, atom_key)
    return data_dict

def append_data_dict(data_dict, header, split_line, atom_key):
    if not atom_key in data_dict:
        data_dict[atom_key] = {}
    for i, v in enumerate(header):
        data_dict[atom_key][v] = split_line[i]
    return data_dict


def parse_ddec6_net_atomic_charges_get_lines(calc_dir):
    fname = opj(calc_dir, "DDEC6_even_tempered_net_atomic_charges.xyz")
    look_key_1 = "The following XYZ coordinates are in angstroms. The atomic dipoles and quadrupoles are in atomic units."
    reading_1 = False
    i1 = 0
    header_1 = None
    data_1_lines = []
    look_key_2 = "The sperically averaged electron density of each atom fit to a function of the form exp(a - br) for r >=rmin_cloud_penetration"
    reading_2 = False
    i2 = 0
    header_2 = None
    data_2_lines = []
    with open(fname, "r") as f:
        for i, line in enumerate(f):
            if look_key_1 in line:
                reading_1 = True
                i1 = i
            elif look_key_2 in line:
                reading_2 = True
                reading_1 = False
                i2 = i
            elif reading_1:
                if i == i1 + 1:
                    header_1 = line
                elif len(line.split()) == 0:
                    reading_1 = False
                else:
                    data_1_lines.append(line)
            elif reading_2:
                if i == i2 + 1:
                    header_2 = line
                elif len(line.split()) == 0:
                    reading_2 = False
                else:
                    data_2_lines.append(line)
    return header_1, data_1_lines, header_2, data_2_lines

