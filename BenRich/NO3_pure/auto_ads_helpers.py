from os import listdir, mkdir
from ase.io import read as _read, write as _write
read = lambda path: _read(path, format="gaussian-in")
write = lambda path, atoms: _write(path, atoms, format="gaussian-in")
from os.path import join as opj, exists as ope, isdir, basename
import matplotlib.pyplot as plt
import numpy as np
from shutil import copy as cp
from copy import deepcopy, copy

cval_dict = {
    "H": 1.0,
    "N2": 1.5,
    "NO3": 1.5
}



default_xy_vec = np.array([1.0, 0.0, 0.0])
ref_xvec = np.array([1.0, 0.0, 0.0])
ref_yvec = np.array([0.0, 1.0, 0.0])
ref_zvec = np.array([0.0, 0.0, 1.0])

step_vec_key = "step vec"
rotate_xy_vec_key = "rotate xy vec"
start_xyz_key = "start xyz"
bad_struc_key = "bad struc func"

def get_min_vec(atoms, i1, i2, ec=False):
    p1 = atoms.positions[i1]
    _p2 = atoms.positions[i2]
    av = atoms.cell[0]
    bv = atoms.cell[1]
    xr = [0]
    if ec:
        xr = [-1, 0, 1]
    dists = []
    vecs = []
    for a in xr:
        for b in xr:
            p2 = _p2 + a*av + b*bv
            vec = p2 - p1
            vecs.append(vec)
            dists.append(np.linalg.norm(vec))
            # print(f"{i1} {i2} {vec}")
    mindx = dists.index(np.min(dists))
    return vecs[mindx], dists[mindx]

def get_min_posn(atoms, i_ref, i, ec=False):
    p0 = atoms.positions[i_ref]
    pvec, _ = get_min_vec(atoms, i_ref, i, ec=ec)
    return p0 + pvec

def too_close(ads_atoms, mol_idcs, cutoff=1.0):
    for mol_idx in mol_idcs:
        for j in range(len(ads_atoms)):
            if not j in mol_idcs:
                mvec, mdist = get_min_vec(ads_atoms, mol_idx, j, ec=True)
                if mdist < cutoff:
                    # print(f"{j} too close ({mdist})")
                    return True
    return False

def get_mean_posn(atoms, idcs, ec=False):
    posns = [atoms.positions[idcs[0]]]
    for j in range(len(idcs)-1):
        dvec, _ = get_min_vec(atoms, idcs[0], idcs[j+1], ec=ec)
        posns.append(posns[0]+dvec)
    mean_posn = np.zeros(3)
    for p in posns:
        mean_posn += p
    mean_posn *= (1/len(idcs))
    return mean_posn

def get_min_surf_aidcs_dist(surf_atoms, ads_idcs, ec):
    min_dist = 100
    vec = None
    for i, idx1 in enumerate(ads_idcs):
        for j, idx2 in enumerate(ads_idcs[i+1:]):
            _vec, _dist = get_min_vec(surf_atoms, idx1, idx2, ec=ec)
            if _dist < min_dist:
                min_dist = _dist
                vec = _vec
    return min_dist

def get_xy_vec(surf_atoms, ads_idcs, ec=False):
    max_dist = 0
    vec = None
    for i, idx1 in enumerate(ads_idcs):
        for j, idx2 in enumerate(ads_idcs[i+1:]):
            _vec, _dist = get_min_vec(surf_atoms, idx1, idx2, ec=ec)
            if _dist > max_dist:
                max_dist = _dist
                vec = _vec
    vec *= (1/np.linalg.norm(vec))
    return vec

def get_bridge_step_vec(xy_vec, surf_atoms):
    e3 = surf_atoms.cell[2] * (1/np.linalg.norm(surf_atoms.cell[2]))
    bridge_vec = np.cross(xy_vec, np.cross(e3, xy_vec))
    if bridge_vec[2] < 0:
        bridge_vec *= -1
    bridge_vec *= (1/np.linalg.norm(bridge_vec))
    return bridge_vec

def get_normal_vec3(surf_atoms, i1, i2, i3, ec=False):
    posns = [surf_atoms.positions[i1], get_min_posn(surf_atoms, i1, i2, ec=ec), get_min_posn(surf_atoms, i1, i3, ec=ec)]
    vec3 = np.cross(posns[1] - posns[0], posns[2] - posns[0])
    if vec3[2] < 0:
        vec3 *= (-1)
    vec3 *= 1/np.linalg.norm(vec3)
    return vec3

def get_normal_vec(surf_atoms, ads_idcs, ec):
    sum_norm_vec = np.zeros(3)
    for i, i1 in enumerate(ads_idcs):
        for _j, i2 in enumerate(ads_idcs[i+1:]):
            j = i + 1 + _j
            for _k, i3 in enumerate(ads_idcs[j+1:]):
                k = j + 1 + _k
                sum_norm_vec += get_normal_vec3(surf_atoms, i1, i2, i3, ec=ec)
    sum_norm_vec *= (1/np.linalg.norm(sum_norm_vec))
    return sum_norm_vec

def get_ads_instructions_1a(surf_atoms, ads_idcs, mol):
    start_xyz = surf_atoms.positions[ads_idcs[0]]
    step_vec = surf_atoms.cell[2]
    step_vec *= 1/np.linalg.norm(step_vec)
    start_xyz += step_vec*0.01
    cval = cval_dict[mol]
    ads_inst_dict = {
        start_xyz_key: start_xyz,
        step_vec_key: step_vec,
        bad_struc_key: lambda atoms, idcs: too_close(atoms, idcs, cutoff=cval),
        rotate_xy_vec_key: default_xy_vec
    }
    return ads_inst_dict

def get_ads_instructions_2a(surf_atoms, ads_idcs, ec, mol):
    if mol == "H":
        cut_val = get_min_surf_aidcs_dist(surf_atoms, ads_idcs, ec) * (1.8/3)
    else:
        cut_val = cval_dict[mol]
    start_xyz = get_mean_posn(surf_atoms, ads_idcs, ec=ec)
    xy_vec = get_xy_vec(surf_atoms, ads_idcs, ec=ec)
    step_vec = get_bridge_step_vec(xy_vec, surf_atoms)
    step_vec *= 1/np.linalg.norm(step_vec)
    start_xyz += 0.3 * step_vec
    ads_inst_dict = {
        start_xyz_key: start_xyz,
        step_vec_key: step_vec,
        bad_struc_key: lambda atoms, idcs: too_close(atoms, idcs, cutoff=cut_val),
        rotate_xy_vec_key: xy_vec
    }
    return ads_inst_dict

def get_ads_instructions_ma(surf_atoms, ads_idcs, ec, mol):
    if mol == "H":
        cut_val = get_min_surf_aidcs_dist(surf_atoms, ads_idcs, ec) * (2./3)
    else:
        cut_val = cval_dict[mol]
    start_xyz = get_mean_posn(surf_atoms, ads_idcs, ec=ec)
    step_vec = get_normal_vec(surf_atoms, ads_idcs, ec)
    step_vec *= (1/np.linalg.norm(step_vec))
    start_xyz *= 0.1 * step_vec
    ads_inst_dict = {
        start_xyz_key: start_xyz,
        step_vec_key: step_vec,
        bad_struc_key: lambda atoms, idcs: too_close(atoms, idcs, cutoff=cut_val),
        rotate_xy_vec_key: get_xy_vec(surf_atoms, ads_idcs, ec=ec)
    }
    return ads_inst_dict

def get_ads_instructions(surf_atoms, ads_idcs, mol, ec=False):
    if len(ads_idcs) == 1:
        return get_ads_instructions_1a(surf_atoms, ads_idcs, mol)
    elif len(ads_idcs) == 2:
        return get_ads_instructions_2a(surf_atoms, ads_idcs, ec, mol)
    elif len(ads_idcs) >= 3:
        return get_ads_instructions_ma(surf_atoms, ads_idcs, ec, mol)

# def rotate_xy(mol_atoms, vec, surf_atoms):
#     defacto_x = surf_atoms.cell[0] + surf_atoms.cell[1]
#     defacto_x *= (1/np.linalg.norm(defacto_x))
#     defacto_z = surf_atoms.cell[2] * (1/np.linalg.norm(surf_atoms.cell[2]))
#     theta = np.arccos(np.dot(vec, defacto_x))
#     theta *= (180/np.pi) * (-1)
#     try:
#         mol_atoms.rotate(theta, defacto_z, center=mol_atoms.get_center_of_mass(), rotate_cell=False)
#     except:
#         pass
#     return mol_atoms

def get_defacto_z(surf_atoms):
    defacto_z = surf_atoms.cell[2] * (1/np.linalg.norm(surf_atoms.cell[2]))
    return defacto_z

def get_defacto_x(surf_atoms):
    defacto_x = surf_atoms.cell[0] + surf_atoms.cell[1]
    defacto_x *= (1/np.linalg.norm(defacto_x))
    defacto_x = proj_out_defacto_z(surf_atoms, defacto_x)
    defacto_x *= (1/np.linalg.norm(defacto_x))
    return defacto_x

def proj_out_defacto_z(surf_atoms, vec, norm=True):
    new_vec = deepcopy(vec)
    defacto_z = get_defacto_z(surf_atoms)
    new_vec -= defacto_z*(np.dot(new_vec, defacto_z))
    if norm:
        new_vec *= (1/np.linalg.norm(new_vec))
    return new_vec

def proj_out_defacto_x(surf_atoms, vec, norm=True):
    new_vec = deepcopy(vec)
    defacto_x = get_defacto_x(surf_atoms)
    new_vec -= defacto_x*(np.dot(new_vec, defacto_x))
    if norm:
        new_vec *= (1/np.linalg.norm(new_vec))
    return new_vec


def rotate_xy(mol_atoms, vec, surf_atoms):
    defacto_x = get_defacto_x(surf_atoms)
    vec = proj_out_defacto_z(surf_atoms, vec, norm=True)
    defacto_z = get_defacto_z(surf_atoms)
    theta = np.arccos(np.dot(vec, defacto_x))
    theta *= (180/np.pi) * (-1)
    mol_atoms.rotate(theta, defacto_z, center=mol_atoms.get_center_of_mass(), rotate_cell=False)
    return mol_atoms

def rotate_z(mol_atoms, xy_vec, step_vec, surf_atoms):
    use_step_vec = proj_out_defacto_x(surf_atoms, step_vec, norm=True)
    use_xy_vec = proj_out_defacto_z(surf_atoms, xy_vec, norm=True)
    defacto_z = get_defacto_z(surf_atoms)
    costheta = np.dot(use_step_vec, defacto_z)
    theta = np.arccos(costheta)
    theta *= (180/np.pi) * (-1)
    axis = np.cross(use_xy_vec, defacto_z)
    axis *= (1/np.linalg.norm(axis))
    mol_atoms.rotate(theta, axis, center=mol_atoms.get_center_of_mass(), rotate_cell=False)
    return mol_atoms

ref_atoms_x_path = "D:\\scratch_backup\\perl\\NO3_pure\\ref_structs\\H2\\x_ref.gjf"

def align_mol_to_cell(mol_atoms, surf_atoms):
    defacto_z = get_defacto_z(surf_atoms)
    costheta = np.dot(ref_zvec, defacto_z)
    theta = np.arccos(costheta)
    theta *= (180/np.pi) * (-1)
    mol_atoms.rotate(theta, ref_yvec, center=mol_atoms.get_center_of_mass(), rotate_cell=False)
    defacto_x = get_defacto_x(surf_atoms)
    defacto_x *= (1/np.linalg.norm(defacto_x))
    costheta = np.dot(ref_yvec, defacto_x)
    theta = np.arccos(costheta)
    theta *= (180/np.pi) * (-1)
    mol_atoms.rotate(theta, defacto_z, center=mol_atoms.get_center_of_mass(), rotate_cell=False)
    return mol_atoms


def rotate_mol_atoms(mol_atoms, xy_vec, step_vec, surf_atoms_path):
    surf_atoms = read(surf_atoms_path)
    if len(mol_atoms) > 1:
        mol_atoms = align_mol_to_cell(mol_atoms, surf_atoms)
        mol_atoms = rotate_xy(mol_atoms, xy_vec, surf_atoms)
        mol_atoms = rotate_z(mol_atoms, xy_vec, step_vec, surf_atoms)
    return mol_atoms

def place_mol_on_surf_placer(surf_atoms_path, mol_atoms, xyz):
    ads_atoms = read(surf_atoms_path)
    nSurfAtoms = len(surf_atoms)
    mol_idcs = []
    for i, atom in enumerate(mol_atoms):
        ads_atoms.append(atom)
        mol_atom_idx = nSurfAtoms + i
        ads_atoms.positions[mol_atom_idx] += xyz
        mol_idcs.append(mol_atom_idx)
    return ads_atoms, mol_idcs

def place_mol_on_surf_stepper(surf_atoms_path, mol_atoms, start_xyz, step_vec, bad_struc_func, step_size=0.1):
    for i in range(int(5./step_size)):
        xyz = start_xyz + (step_vec*i*step_size)
        ads_atoms, mol_idcs = place_mol_on_surf_placer(surf_atoms_path, mol_atoms, xyz)
        if not bad_struc_func(ads_atoms, mol_idcs):
            return ads_atoms
        else:
            continue
    return None

def get_ads_atoms(surf_atoms_path, mol_atoms_path, ads_instructions):
    step_vec = ads_instructions[step_vec_key]
    xy_vec = ads_instructions[rotate_xy_vec_key]
    mol_atoms = read(mol_atoms_path)
    mol_atoms = rotate_mol_atoms(mol_atoms, xy_vec, step_vec, surf_atoms_path)
    start_xyz = ads_instructions[start_xyz_key]
    bad_struc_func = ads_instructions[bad_struc_key]
    ads_atoms = place_mol_on_surf_stepper(surf_atoms_path, mol_atoms, start_xyz, step_vec, bad_struc_func)
    return ads_atoms