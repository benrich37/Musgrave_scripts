import numpy as np
import sys

sys.path.append("/")
import jdftxfuncs as jfunc
from os.path import join as opj
import matplotlib.pyplot as plt
from ase.io import read
from ase.units import Hartree, Bohr
from ase import Atoms, Atom
from ase.dft.dos import linear_tetrahedron_integration as lti
from numba import jit
from itertools import product
from ase.io import read
from os.path import join as opj, exists as ope, dirname
from os import listdir
import scipy
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings


warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

bond_cutoff = 3
nrg_sym = "G"

def get_kmap(atoms):
    el_counter_dict = {}
    idx_to_key_map = []
    els = atoms.get_chemical_symbols()
    for i, el in enumerate(els):
        if not el in el_counter_dict:
            el_counter_dict[el] = 0
        el_counter_dict[el] += 1
        idx_to_key_map.append(f"{el} #{el_counter_dict[el]}")
    return idx_to_key_map


def atom_idx_to_key_map(atoms):
    return get_kmap(atoms)


def get_min_dist(posn1, posn2, cell, pbc):
    dists = []
    xrs = []
    for x in pbc:
        if x:
            xrs.append([-1, 0, 1])
        else:
            xrs.append([0])
    for a in xrs[0]:
        for b in xrs[1]:
            for c in xrs[2]:
                _p2 = posn2 + (a * cell[0]) + (b * cell[1]) + (c * cell[2])
                dists.append(np.linalg.norm(posn1 - _p2))
    return np.min(dists)


def get_pair_idcs(atoms, pbc, atomic_radii_dict, tol=0.1):
    pairs = []
    els = atoms.get_chemical_symbols()
    posns = atoms.positions
    for i, el1 in enumerate(els):
        for j, el2 in enumerate(els):
            if i > j:
                cutoff = atomic_radii_dict[el1] + atomic_radii_dict[el2]
                if get_min_dist(posns[i], posns[j], atoms.cell, pbc) <= cutoff + tol:
                    pairs.append([i, j])
    return pairs


def norm_projs(proj_kju):
    nStates = np.shape(proj_kju)[0]
    nBands = np.shape(proj_kju)[1]
    nProj = np.shape(proj_kju)[2]
    u_sums = np.zeros(nProj)
    for u in range(nProj):
        for k in range(nStates):
            for j in range(nBands):
                c = abs(proj_kju[k, j, u]) ** 2
                u_sums[u] += c
    for u in range(nProj):
        for k in range(nStates):
            for j in range(nBands):
                proj_kju[k, j, u] *= (1 / np.sqrt(u_sums[u]))
    return proj_kju


def get_jnorms(proj_kju):
    nStates = np.shape(proj_kju)[0]
    nBands = np.shape(proj_kju)[1]
    nProj = np.shape(proj_kju)[2]
    j_sums = np.zeros(nBands)
    for u in range(nProj):
        for k in range(nStates):
            for j in range(nBands):
                c = abs(proj_kju[k, j, u]) ** 2
                j_sums[j] += c
    for j in range(len(j_sums)):
        j_sums[j] = np.sqrt(j_sums[j])
    return j_sums


def get_knorms(proj_kju):
    nStates = np.shape(proj_kju)[0]
    nBands = np.shape(proj_kju)[1]
    nProj = np.shape(proj_kju)[2]
    k_sums = np.zeros(nStates)
    for u in range(nProj):
        for k in range(nStates):
            for j in range(nBands):
                c = abs(proj_kju[k, j, u]) ** 2
                k_sums[k] += c
    for k in range(len(k_sums)):
        k_sums[k] = np.sqrt(k_sums[k])
    return k_sums


def parse_data(root=None, bandfile="bandProjections", kPtsfile="kPts", eigfile="eigenvals", fillingsfile="fillings",
               outfile="out", gamma=False, kidcs=None, print_sig=False):
    # TODO: Return array of strings to indicate which orbital is in which orbital index
    # ^ yeah fat chance nerd
    """
    :param bandfile: Path to BandProjections file (str)
    :param gvecfile: Path to Gvectors file (str)
    :param eigfile: Path to eigenvalues file (str)
    :param guts: Whether to data not directly needed by main functions (Boolean)
    :return:
        - proj: a rank 3 numpy array containing the complex band projection,
                data (<φ_μ|ψ_j> = T_μj) with dimensions (nStates, nBands, nProj)
        - nStates: the number of electronic states (integer)
        - nBands: the number of band functions (integer)
        - nProj: the number of band projections (integer)
        - nOrbsPerAtom: a list containing the number of orbitals considered
                        for each atom in the crystal structure (list(int))
        - wk: A list of weight factors for each k-point (list(float))
        - k_points: A list of k-points (given as 3 floats) for each k-point. (list(list(float))
        - E: nStates by nBands array of KS eigenvalues (np.ndarray(float))
        *- iGarr: A list of numpy arrays for the miller indices of each G-vector used in the
                  expansion of each state (list(np.ndarray(int)))
    :rtype: tuple
    """
    if not root is None:
        bandfile = opj(root, bandfile)
        kPtsfile = opj(root, kPtsfile)
        eigfile = opj(root, eigfile)
        fillingsfile = opj(root, fillingsfile)
        outfile = opj(root, outfile)
    proj_kju, nStates, nBands, nProj, nSpecies, nOrbsPerAtom = jfunc.parse_complex_bandfile(bandfile)
    orbs_dict = jfunc.orbs_idx_dict(outfile, nOrbsPerAtom)
    kfolding = jfunc.get_kfolding(outfile)
    nK = int(np.prod(kfolding))
    nSpin = int(nStates / nK)
    if ope(kPtsfile):
        wk, ks, nStates = jfunc.parse_kptsfile(kPtsfile)
        wk = np.array(wk)
        ks = np.array(ks)
    else:
        ks = np.zeros([nK*nSpin, 3])
        wk = np.ones(nK*nSpin)
        wk *= (1/nK)
    wk = wk.reshape([nSpin, kfolding[0], kfolding[1], kfolding[2]])
    ks = ks.reshape([nSpin, kfolding[0], kfolding[1], kfolding[2], 3])
    E = np.fromfile(eigfile)
    Eshape = [nSpin, kfolding[0], kfolding[1], kfolding[2], nBands]
    E_sabcj = E.reshape(Eshape)
    fillings = np.fromfile(fillingsfile)
    occ_sabcj = fillings.reshape(Eshape)
    # Normalize such that sum(occ_kj) = nelec
    # occ_sabcj *= (1/nK)
    # Normalize such that sum_jk(<u|jk><jk|u>) = 1
    # proj_kju = norm_projs(proj_kju)
    proj_shape = Eshape
    proj_shape.append(nProj)
    proj_flat = proj_kju.flatten()
    proj_sabcju = proj_flat.reshape(proj_shape)
    mu = jfunc.get_mu(outfile)
    if not kidcs is None:
        abc = []
        for i in range(3):
            abc.append([kidcs[i], kidcs[i] + 1])
        proj_sabcju = proj_sabcju[:, abc[0][0]:abc[0][1], abc[1][0]:abc[1][1], abc[2][0]:abc[2][1], :, :]
        E_sabcj = E_sabcj[:, abc[0][0]:abc[0][1], abc[1][0]:abc[1][1], abc[2][0]:abc[2][1], :]
        occ_sabcj = occ_sabcj[:, abc[0][0]:abc[0][1], abc[1][0]:abc[1][1], abc[2][0]:abc[2][1], :]
        wk = np.ones(np.shape(occ_sabcj[:, :, :, :, 0])) * 0.5
        ks = [np.zeros([3])]
    elif gamma:
        abc = []
        for i in range(3):
            kfi = kfolding[i]
            kfi_0 = int(np.ceil(kfi / 2.) - 1)
            kfi_p = 1
            if kfi % 2 == 0:
                kfi_p += 1
            abc.append([kfi_0, kfi_0 + kfi_p])
        proj_sabcju = proj_sabcju[:, abc[0][0]:abc[0][1], abc[1][0]:abc[1][1], abc[2][0]:abc[2][1], :, :]
        E_sabcj = E_sabcj[:, abc[0][0]:abc[0][1], abc[1][0]:abc[1][1], abc[2][0]:abc[2][1], :]
        occ_sabcj = occ_sabcj[:, abc[0][0]:abc[0][1], abc[1][0]:abc[1][1], abc[2][0]:abc[2][1], :]
        wk = np.ones(np.shape(occ_sabcj[:, :, :, :, 0])) * 0.5
        ks = [np.zeros([3])]
    if print_sig:
        print("proj_sabcju, E_sabcj, occ_sabcj, wk, ks, orbs_dict, mu")
    return proj_sabcju, E_sabcj, occ_sabcj, wk, ks, orbs_dict, mu


def parse_complex_bandprojection_old(tokens):
    # This code is unjittable, leave it be for now
    """ Should only be called for data generated by modified JDFTx
    :param tokens: Parsed data from bandProjections file
    :return out: data in the normal numpy complex data format (list(complex))
    """
    out = []
    for i in range(int(len(tokens)/2)):
        repart = tokens[2*i]
        impart = tokens[(2*i) + 1].lstrip("i")
        num = complex(float(repart), float(impart))
        out.append(num)
    return out


def parse_bandfile_old(bandfile):
    """ Parser function for the 'bandProjections' file produced by JDFTx
    :param bandfile: the path to the bandProjections file to parse
    :type bandfile: str
    :return: a tuple containing six elements:
        - proj: a rank 3 numpy array containing the complex band projection,
                data (<φ_μ|ψ_j> = T_μj) with dimensions (nStates, nBands, nProj)
        - nStates: the number of electronic states (integer)
        - nBands: the number of energy bands (integer)
        - nProj: the number of band projections (integer)
        - nSpecies: the number of atomic species (integer)
        - nOrbsPerAtom: a list containing the number of orbitals considered
                        for each atom in the crystal structure
    :rtype: tuple
    """
    with open(bandfile, 'r') as f:
        for iLine, line in enumerate(f):
            tokens = line.split()
            if iLine==0:
                nStates = int(tokens[0])
                nBands = int(tokens[2])
                nProj = int(tokens[4])
                nSpecies = int(tokens[6])
                proj = np.zeros((nStates, nBands, nProj), dtype=complex)
                parser = parse_complex_bandprojection_old
                nOrbsPerAtom = []
            elif iLine>=2:
                if iLine<nSpecies+2:
                    nAtoms = int(tokens[1])
                    nOrbsPerAtom.extend( [int(tokens[2]),] * nAtoms)
                else:
                    iState = (iLine-(nSpecies+2)) // (nBands+1)
                    iBand = (iLine-(nSpecies+2)) - iState*(nBands+1) - 1
                    if iBand>=0 and iState<nStates:
                        proj[iState,iBand]=np.array(parser(tokens))
    f.close()
    return proj, nStates, nBands, nProj, nSpecies, nOrbsPerAtom


def parse_data_old(root=None, bandfile="bandProjections", kPtsfile="kPts", eigfile="eigenvals", fillingsfile="fillings",
               outfile="out", gamma=False, kidcs=None):
    if not root is None:
        bandfile = opj(root, bandfile)
        kPtsfile = opj(root, kPtsfile)
        eigfile = opj(root, eigfile)
        fillingsfile = opj(root, fillingsfile)
        outfile = opj(root, outfile)
    proj_kju, nStates, nBands, nProj, nSpecies, nOrbsPerAtom = parse_bandfile_old(bandfile)
    orbs_dict = jfunc.orbs_idx_dict(outfile, nOrbsPerAtom)
    kfolding = jfunc.get_kfolding(outfile)
    nK = int(np.prod(kfolding))
    nSpin = int(nStates / nK)
    if ope(kPtsfile):
        wk, ks, nStates = jfunc.parse_kptsfile(kPtsfile)
        wk = np.array(wk)
        ks = np.array(ks)
    else:
        ks = np.zeros([nK*nSpin, 3])
        wk = np.ones(nK*nSpin)
        wk *= (1/nK)
    wk = wk.reshape([nSpin, kfolding[0], kfolding[1], kfolding[2]])
    ks = ks.reshape([nSpin, kfolding[0], kfolding[1], kfolding[2], 3])
    E = np.fromfile(eigfile)
    Eshape = [nSpin, kfolding[0], kfolding[1], kfolding[2], nBands]
    E_sabcj = E.reshape(Eshape)
    fillings = np.fromfile(fillingsfile)
    occ_sabcj = fillings.reshape(Eshape)
    # Normalize such that sum(occ_kj) = nelec
    # occ_sabcj *= (1/nK)
    # Normalize such that sum_jk(<u|jk><jk|u>) = 1
    # proj_kju = norm_projs(proj_kju)
    proj_shape = Eshape
    proj_shape.append(nProj)
    proj_flat = proj_kju.flatten()
    proj_sabcju = proj_flat.reshape(proj_shape)
    mu = jfunc.get_mu(outfile)
    if not kidcs is None:
        abc = []
        for i in range(3):
            abc.append([kidcs[i], kidcs[i] + 1])
        proj_sabcju = proj_sabcju[:, abc[0][0]:abc[0][1], abc[1][0]:abc[1][1], abc[2][0]:abc[2][1], :, :]
        E_sabcj = E_sabcj[:, abc[0][0]:abc[0][1], abc[1][0]:abc[1][1], abc[2][0]:abc[2][1], :]
        occ_sabcj = occ_sabcj[:, abc[0][0]:abc[0][1], abc[1][0]:abc[1][1], abc[2][0]:abc[2][1], :]
        wk = np.ones(np.shape(occ_sabcj[:, :, :, :, 0])) * 0.5
        ks = [np.zeros([3])]
    elif gamma:
        abc = []
        for i in range(3):
            kfi = kfolding[i]
            kfi_0 = int(np.ceil(kfi / 2.) - 1)
            kfi_p = 1
            if kfi % 2 == 0:
                kfi_p += 1
            abc.append([kfi_0, kfi_0 + kfi_p])
        proj_sabcju = proj_sabcju[:, abc[0][0]:abc[0][1], abc[1][0]:abc[1][1], abc[2][0]:abc[2][1], :, :]
        E_sabcj = E_sabcj[:, abc[0][0]:abc[0][1], abc[1][0]:abc[1][1], abc[2][0]:abc[2][1], :]
        occ_sabcj = occ_sabcj[:, abc[0][0]:abc[0][1], abc[1][0]:abc[1][1], abc[2][0]:abc[2][1], :]
        wk = np.ones(np.shape(occ_sabcj[:, :, :, :, 0])) * 0.5
        ks = [np.zeros([3])]
    return proj_sabcju, E_sabcj, occ_sabcj, wk, ks, orbs_dict, mu


def get_el_orb_u_dict(path, atoms, orbs_dict, aidcs):
    els = [atoms.get_chemical_symbols()[i] for i in aidcs]
    kmap = atom_idx_to_key_map(atoms)
    labels_dict = jfunc.get_atom_orb_labels_dict(path)
    el_orbs_dict = {}
    for i, el in enumerate(els):
        if not el in el_orbs_dict:
            el_orbs_dict[el] = {}
        for ui, u in enumerate(orbs_dict[kmap[aidcs[i]]]):
            orb = labels_dict[el][ui]
            if not orb in el_orbs_dict[el]:
                el_orbs_dict[el][orb] = []
            el_orbs_dict[el][orb].append(u)
    return el_orbs_dict

def get_start_lines(outfname, add_end=False):
    start_lines = []
    for i, line in enumerate(open(outfname)):
        if "JDFTx 1." in line or "Input parsed successfully" in line:
            start_lines.append(i)
        end_line = i
    if add_end:
        start_lines.append(end_line)
    return start_lines

def get_atoms_from_outfile_data(names, posns, R, charges=None, E=0, momenta=None):
    atoms = Atoms()
    posns *= Bohr
    R = R.T*Bohr
    atoms.cell = R
    if charges is None:
        charges = np.zeros(len(names))
    if momenta is None:
        momenta = np.zeros([len(names), 3])
    for i in range(len(names)):
        atoms.append(Atom(names[i], posns[i], charge=charges[i], momentum=momenta[i]))
    atoms.E = E
    return atoms

def get_start_line(outfname):
    start_lines = get_start_lines(outfname, add_end=False)
    return start_lines[-1]

def get_input_coord_vars_from_outfile(outfname):
    start_line = get_start_line(outfname)
    names = []
    posns = []
    R = np.zeros([3,3])
    lat_row = 0
    active_lattice = False
    with open(outfname) as f:
        for i, line in enumerate(f):
            if i > start_line:
                tokens = line.split()
                if len(tokens) > 0:
                    if tokens[0] == "ion":
                        names.append(tokens[1])
                        posns.append(np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])]))
                    elif tokens[0] == "lattice":
                        active_lattice = True
                    elif active_lattice:
                        if lat_row < 3:
                            R[lat_row, :] = [float(x) for x in tokens[:3]]
                            lat_row += 1
                        else:
                            active_lattice = False
                    elif "Initializing the Grid" in line:
                        break
    return names, posns, R

def get_atoms_list_from_out_reset_vars(nAtoms=100, _def=100):
    R = np.zeros([3, 3])
    posns = []
    names = []
    chargeDir = {}
    active_lattice = False
    lat_row = 0
    active_posns = False
    log_vars = False
    coords = None
    new_posn = False
    active_lowdin = False
    idxMap = {}
    j = 0
    E = 0
    if nAtoms is None:
        nAtoms = _def
    charges = np.zeros(nAtoms, dtype=float)
    forces = []
    active_forces = False
    coords_forces = None
    return R, posns, names, chargeDir, active_posns, active_lowdin, active_lattice, posns, coords, idxMap, j, lat_row, \
        new_posn, log_vars, E, charges, forces, active_forces, coords_forces

def get_atoms_list_from_out_slice(outfile, i_start, i_end):
    charge_key = "oxidation-state"
    opts = []
    nAtoms = None
    R, posns, names, chargeDir, active_posns, active_lowdin, active_lattice, posns, coords, idxMap, j, lat_row, \
        new_posn, log_vars, E, charges, forces, active_forces, coords_forces = get_atoms_list_from_out_reset_vars()
    for i, line in enumerate(open(outfile)):
        if i > i_start and i < i_end:
            if new_posn:
                if "Lowdin population analysis " in line:
                    active_lowdin = True
                elif "R =" in line:
                    active_lattice = True
                elif "# Forces in" in line:
                    active_forces = True
                    coords_forces = line.split()[3]
                elif line.find('# Ionic positions in') >= 0:
                    coords = line.split()[4]
                    active_posns = True
                elif active_lattice:
                    if lat_row < 3:
                        R[lat_row, :] = [float(x) for x in line.split()[1:-1]]
                        lat_row += 1
                    else:
                        active_lattice = False
                        lat_row = 0
                elif active_posns:
                    tokens = line.split()
                    if len(tokens) and tokens[0] == 'ion':
                        names.append(tokens[1])
                        posns.append(np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])]))
                        if tokens[1] not in idxMap:
                            idxMap[tokens[1]] = []
                        idxMap[tokens[1]].append(j)
                        j += 1
                    else:
                        posns = np.array(posns)
                        active_posns = False
                        nAtoms = len(names)
                        if len(charges) < nAtoms:
                            charges = np.zeros(nAtoms)
                ##########
                elif active_forces:
                    tokens = line.split()
                    if len(tokens) and tokens[0] == 'force':
                        forces.append(np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])]))
                    else:
                        forces = np.array(forces)
                        active_forces = False
                ##########
                elif "Minimize: Iter:" in line:
                    if "F: " in line:
                        E = float(line[line.index("F: "):].split(' ')[1])
                    elif "G: " in line:
                        E = float(line[line.index("G: "):].split(' ')[1])
                elif active_lowdin:
                    if charge_key in line:
                        look = line.rstrip('\n')[line.index(charge_key):].split(' ')
                        symbol = str(look[1])
                        line_charges = [float(val) for val in look[2:]]
                        chargeDir[symbol] = line_charges
                        for atom in list(chargeDir.keys()):
                            for k, idx in enumerate(idxMap[atom]):
                                charges[idx] += chargeDir[atom][k]
                    elif "#" not in line:
                        active_lowdin = False
                        log_vars = True
                elif log_vars:
                    if np.sum(R) == 0.0:
                        R = get_input_coord_vars_from_outfile(outfile)[2]
                    if coords != 'cartesian':
                        posns = np.dot(posns, R)
                    if len(forces) == 0:
                        forces = np.zeros([nAtoms, 3])
                    if coords_forces.lower() != 'cartesian':
                        forces = np.dot(forces, R)
                    opts.append(get_atoms_from_outfile_data(names, posns, R, charges=charges, E=E, momenta=forces))
                    R, posns, names, chargeDir, active_posns, active_lowdin, active_lattice, posns, coords, idxMap, j, lat_row, \
                        new_posn, log_vars, E, charges, forces, active_forces, coords_forces = get_atoms_list_from_out_reset_vars(
                        nAtoms=nAtoms)
            elif "Computing DFT-D3 correction:" in line:
                new_posn = True
    return opts

def get_atoms_list_from_out(outfile):
    start_lines = get_start_lines(outfile, add_end=True)
    for i in range(len(start_lines) - 1):
        i_start = start_lines[::-1][i+1]
        i_end = start_lines[::-1][i]
        atoms_list = get_atoms_list_from_out_slice(outfile, i_start, i_end)
        if type(atoms_list) is list:
            if len(atoms_list):
                return atoms_list
    erstr = "Failed getting atoms list from out file"
    raise ValueError(erstr)
def get_atoms_from_out(outfile):
    atoms_list = get_atoms_list_from_out(outfile)
    return atoms_list[-1]

def get_atoms(path):
    if ope(opj(path, "CONTCAR.gjf")):
        atoms = read(opj(path, "CONTCAR.gjf"), format="gaussian-in")
    elif ope(opj(path, "CONTCAR")):
        atoms = read(opj(path, "CONTCAR"), format="vasp")
    else:
        atoms = get_atoms_from_out(opj(path, "out"))
    return atoms


def get_ads_idcs(atoms):
    ads_idcs = []
    for i, sym in enumerate(atoms.get_chemical_symbols()):
        if sym in ["N", "H"]:
            ads_idcs.append(i)
    return ads_idcs


def get_ads_center(atoms, ads_idcs):
    ads_center = np.zeros(3)
    for idx in ads_idcs:
        ads_center += atoms.positions[idx]
    ads_center *= (1 / len(ads_idcs))
    return ads_center


def is_bonded(dist, sym):
    if dist < bond_cutoff:
        return True
    return False


def get_dist(i, j, atoms):
    posns = atoms.positions
    pi = posns[i]
    pj = posns[j]
    dists = []
    for a in [-1, 0, 1]:
        for b in [-1, 0, 1]:
            for c in [-1, 0, 1]:
                _pj = pj + (atoms.cell[0] * a) + (atoms.cell[1] * b) + (atoms.cell[2] * c)
                dists.append(np.linalg.norm(pi - _pj))
    return np.min(dists)


def get_surf_aidcs(path):
    atoms = get_atoms(path)
    syms = atoms.get_chemical_symbols()
    ads_idcs = get_ads_idcs(atoms)
    ads_center = get_ads_center(atoms, ads_idcs)
    posns = atoms.positions
    ads_posns = [posns[idx] for idx in ads_idcs]
    dists = [np.min([get_dist(i, j, atoms) for i in ads_idcs]) for j in range(len(syms))]
    surf_aidcs = []
    for i, dist in enumerate(dists):
        if (not i in ads_idcs) and (is_bonded(dist, syms[i])):
            surf_aidcs.append(i)
    return surf_aidcs


def get_nrg(path, nrg_key=nrg_sym):
    ecomp = opj(path, "Ecomponents")
    with open(ecomp, "r") as f:
        for line in f:
            if "=" in line:
                key = line.split("=")[0].strip()
                if nrg_key in key:
                    val = float(line.strip().split("=")[1].strip())
                    return val


def collect_sites_data(bias_dir):
    sites_data = []
    sites = listdir(bias_dir)
    for site in sites:
        if (not site[0] == "_") and (not "_" + site in sites):
            path = opj(bias_dir, site)
            if ope(opj(path, "ion_opt")):
                path = opj(path, "ion_opt")
            if ope(opj(path, "Ecomponents")):
                nrg = get_nrg(path)
                aidcs = get_surf_aidcs(path)
                sites_data.append([nrg, aidcs, site])
            else:
                print(f"Check {path}")
    return sites_data


def site_in_site_list(site_list, sidcs):
    for site in site_list:
        if is_same_site(site, sidcs):
            return True
    return False


def get_site_index(site_list, sidcs):
    for i, site in enumerate(site_list):
        if is_same_site(site, sidcs):
            return i


def is_same_site(sidcs1, sids2):
    if len(sidcs1) == len(sids2):
        for i in range(len(sidcs1)):
            if not sidcs1[i] == sids2[i]:
                return False
            return True
    return False


def get_us_aidcs(aidcs, orbs_dict, kmap, orb_bool_func=None, path=None, atoms=None):
    us = []
    if orb_bool_func is None:
        for aidx in aidcs:
            us += orbs_dict[kmap[aidx]]
    else:
        el_orbs_dict = get_el_orb_u_dict(path, atoms, orbs_dict, aidcs)
        for el in el_orbs_dict:
            for orb in el_orbs_dict[el]:
                if orb_bool_func(orb):
                    us += el_orbs_dict[el][orb]
    return us


def get_weights_aidcs(_aidcs, proj_sabcju=None, data=None, orbs_dict=None, path=None, kmap=None, orb_bool_func=None,
                      atoms=None):
    aidcs = [aidx - 1 for aidx in _aidcs]
    if proj_sabcju is None:
        if data is None:
            if path is None:
                raise ValueError("need more")
            else:
                data = parse_data(root=path)
        proj_sabcju = data[0]
    if orbs_dict is None:
        if data is None:
            if path is None:
                raise ValueError("need more")
            else:
                data = parse_data(root=path)
        orbs_dict = data[5]
    if kmap is None:
        if path is None:
            raise ValueError("need more")
        else:
            atoms = get_atoms(path)
            kmap = atom_idx_to_key_map(atoms)
    us = []
    if orb_bool_func is None:
        for aidx in aidcs:
            us += orbs_dict[kmap[aidx]]
    else:
        el_orbs_dict = get_el_orb_u_dict(path, atoms, orbs_dict, aidcs)
        for el in el_orbs_dict:
            for orb in el_orbs_dict[el]:
                if orb_bool_func(orb):
                    us += el_orbs_dict[el][orb]
    weights = np.zeros(np.shape(proj_sabcju)[:-1])
    for u in us:
        weights += np.abs(proj_sabcju[:, :, :, :, :, u])
    return weights


def gauss(x, mu, sig):
    return np.exp(-((x - mu) ** 2) / sig)

def gsmear(xs, ys, sig, wid=50):
    output = np.zeros(np.shape(xs))
    wid2 = wid * 2
    for i in range(len(xs) - wid2):
        idx = i + wid
        output[i:i + wid2] += gauss(xs[i:i + wid2], xs[idx], sig) * ys[idx]
    return output


def get_dE(E_sabcj, spandiv=4):
    shape = np.shape(E_sabcj)
    nSpin = shape[0]
    nKa = shape[1]
    nKb = shape[2]
    nKc = shape[3]
    nBands = shape[4]
    by_band = []
    for s in range(nSpin):
        for j in range(nBands):
            vals = []
            for a in range(nKa):
                for b in range(nKb):
                    for c in range(nKc):
                        vals.append(E_sabcj[s, a, b, c, j])
            by_band.append(vals)
    spanEs = []
    for vgroup in by_band:
        idcs = np.argsort(vgroup)
        vals = [vgroup[idx] for idx in idcs]
        spanE = vals[-1] - vals[0]
        spanEs.append(spanE)
        mindE = 100
        for i in range(len(vals) - 1):
            mindE = min(mindE, vals[i + 1] - vals[i])
    return min(spanEs) / spandiv


def get_Erange(E_sabcj, dE=None, spread=100, spandiv=4):
    if dE is None:
        dE = get_dE(E_sabcj, spandiv=spandiv)
    Erange = np.arange(np.min(E_sabcj) - dE * spread, np.max(E_sabcj) + dE * spread, dE)
    return Erange


def get_dosUp_dosDn(cell, E_sabcj, Erange, weights=None, do_gsmear=False, sig=0.1, wid=50):
    if weights is None:
        weights = [None, None]
    dosUp = lti(cell, E_sabcj[0], Erange, weights=weights[0])
    dosDn = lti(cell, E_sabcj[1], Erange, weights=weights[1])
    if do_gsmear:
        dosUp = gsmear(Erange, dosUp, sig, wid=wid)
        dosDn = gsmear(Erange, dosDn, sig, wid=wid)
    return dosUp, dosDn


def get_dosTot_dosPol(cell, E_sabcj, Erange, weights=None):
    dosUp, dosDn = get_dosUp_dosDn(cell, E_sabcj, Erange, weights=weights)
    dosTot = dosUp + dosDn
    dosPol = abs(dosUp - dosDn)
    return dosTot, dosPol

def plot_pdos_plotter2(ax, dosUp, dosDn, Erange, mu, xcut=None, ybounds=None):
    ax.plot(dosUp, Erange, label=r"$\rho(\epsilon)_{\alpha}$", c="black", linestyle="solid")
    ax.plot((-1) * dosDn, Erange, label=r"$\rho(\epsilon)_{\beta}$", c="black", linestyle="solid")
    ax.axhline(y=mu, c="red")
    if not xcut is None:
        ax.set_xlim(-1 * xcut, xcut)
    if ybounds is None:
        ybounds = [Erange[0], Erange[-1]]
    ax.set_ylim(ybounds[0], ybounds[1])

def plot_pdos_ax(ax, weights, E_sabcj, Erange, cell, mu, xcut, yb=None, dosUp=None, dosDn=None):
    if (dosUp is None) or (dosDn is None):
        dosUp, dosDn = get_dosUp_dosDn(cell, E_sabcj, Erange, weights)
    plot_pdos_plotter2(ax, dosUp, dosDn, Erange, mu, xcut, yb)


def plot_pdos_aidcs(calc_paths, aidcs, xcut=None, ybs=None, Erange=None, do_gsmear=False, sig=0.1, spandiv=64, wid=50,
                    spread=50, orb_bool_func=None, savedir=None, title=None):
    ncols = len(calc_paths)
    do_ybs = True
    if ybs is None:
        ybs = [None]
        do_ybs = False
    nrows = len(ybs)
    datas = []
    for i, calc_path in enumerate(calc_paths):
        datas.append(parse_data(root=calc_path))
    if do_ybs:
        for i in range(len(ybs)):
            if ybs[i][0] is None:
                ybs[i][0] = np.min(datas[0][1])
            if ybs[i][1] is None:
                ybs[i][1] = np.max(datas[0][1])
    if Erange is None:
        Erange = get_Erange(datas[0][1], spread=spread, spandiv=spandiv)
    if do_ybs:
        hrs = []
        for yb in ybs:
            hrs.append(abs(yb[0] - yb[1]))
    else:
        hrs = [1]
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=not do_ybs, figsize=(5 * nrows, 10),
                            height_ratios=hrs)
    for i, calc_path in enumerate(calc_paths):
        data = datas[i]
        proj_sabcju, E_sabcj, occ_sabcj, wk, ks, orbs_dict, mu = data
        atoms = get_atoms(calc_path)
        weights = get_weights_aidcs(aidcs, data=data, path=calc_path, orb_bool_func=orb_bool_func)
        dosUp, dosDn = get_dosUp_dosDn(atoms.cell, E_sabcj, Erange, weights, do_gsmear=do_gsmear, sig=sig, wid=wid)
        if xcut is None:
            xcut = max(np.max(dosUp), np.max(dosDn))
        for j, yb in enumerate(ybs):
            if do_ybs:
                ax = axs[j, i]
            else:
                ax = axs[i]
            # ax1 = axs[2*i, j]
            # ax2 = axs[1+(2*i), j]
            plot_pdos_ax(ax, weights, E_sabcj, Erange, atoms.cell, mu, xcut, yb=yb, dosUp=dosUp, dosDn=dosDn)


@jit(nopython=True)
def get_P_uvjsabc_jit(proj_sabcju, P_uvjsabc, nProj, nBands, nKa, nKb, nKc, nSpin):
    for u in range(nProj):
        for v in range(nProj):
            for j in range(nBands):
                for a in range(nKa):
                    for b in range(nKb):
                        for c in range(nKc):
                            for s in range(nSpin):
                                t1 = proj_sabcju[s, a, b, c, j, u]
                                t2 = proj_sabcju[s, a, b, c, j, v]
                                P_uvjsabc[u, v, j, s, a, b, c] = np.conj(t1) * t2
    return P_uvjsabc


def get_P_uvjsabc(proj_sabcju):
    shape = np.shape(proj_sabcju)
    nSpin = shape[0]
    nKa = shape[1]
    nKb = shape[2]
    nKc = shape[3]
    nBands = shape[4]
    nProj = shape[5]
    P_uvjsabc = np.zeros([nProj, nProj, nBands, nSpin, nKa, nKb, nKc], dtype=complex)
    P_uvjsabc = get_P_uvjsabc_jit(proj_sabcju, P_uvjsabc, nProj, nBands, nKa, nKb, nKc, nSpin)
    return np.real(P_uvjsabc)

@jit(nopython=True)
def get_P_uvjsabc_bare_min_jit(proj_sabcju, P_uvjsabc, nProj, nBands, nKa, nKb, nKc, nSpin, orbs_u, orbs_v):
    for u in orbs_u:
        for v in orbs_v:
            for j in range(nBands):
                for a in range(nKa):
                    for b in range(nKb):
                        for c in range(nKc):
                            for s in range(nSpin):
                                t1 = proj_sabcju[s, a, b, c, j, u]
                                t2 = proj_sabcju[s, a, b, c, j, v]
                                P_uvjsabc[u, v, j, s, a, b, c] += np.real(np.conj(t1) * t2)
    return P_uvjsabc

# @jit(nopython=True)
# def get_P_uvjsabc_bare_min_jit(proj_sabcju, proj_sabcju_star, P_uvjsabc, nProj, nBands, nKa, nKb, nKc, nSpin, orbs):
#     for u in orbs:
#         for v in orbs:
#             for j in range(nBands):
#                 for a in range(nKa):
#                     for b in range(nKb):
#                         for c in range(nKc):
#                             for s in range(nSpin):
#                                 t1 = proj_sabcju[s, a, b, c, j, u]
#                                 t2 = proj_sabcju_star[s, a, b, c, j, v]
#                                 P_uvjsabc[u, v, j, s, a, b, c] += np.real(t1 * t2)
#     return P_uvjsabc

def get_P_uvjsabc_bare_min(proj_sabcju, orbs_u, orbs_v):
    shape = np.shape(proj_sabcju)
    nSpin = shape[0]
    nKa = shape[1]
    nKb = shape[2]
    nKc = shape[3]
    nBands = shape[4]
    nProj = shape[5]
    P_uvjsabc = np.zeros([nProj, nProj, nBands, nSpin, nKa, nKb, nKc], dtype=np.float32)
    P_uvjsabc = get_P_uvjsabc_bare_min_jit(proj_sabcju, P_uvjsabc, nProj, nBands, nKa, nKb, nKc, nSpin, orbs_u, orbs_v)
    return np.real(P_uvjsabc)


@jit(nopython=True)
def get_H_uvsabc_jit(H_uvsabc, P_uvjsabc, E_sabcj, nProj, nBands, nKa, nKb, nKc, nSpin):
    for u in range(nProj):
        for v in range(nProj):
            for j in range(nBands):
                for s in range(nSpin):
                    for a in range(nKa):
                        for b in range(nKb):
                            for c in range(nKc):
                                H_uvsabc[u, v, s, a, b, c] += P_uvjsabc[u, v, j, s, a, b, c] * E_sabcj[s, a, b, c, j]
    return H_uvsabc


def get_H_uvsabc(P_uvjsabc, E_sabcj):
    shape = np.shape(P_uvjsabc)
    nProj = shape[0]
    nBands = shape[2]
    nSpin = shape[3]
    nKa = shape[4]
    nKb = shape[5]
    nKc = shape[6]
    H_uvsabc = np.zeros([nProj, nProj, nSpin, nKa, nKb, nKc], dtype=complex)
    return get_H_uvsabc_jit(H_uvsabc, P_uvjsabc, E_sabcj, nProj, nBands, nKa, nKb, nKc, nSpin)


@jit(nopython=True)
def get_H_uvsabc_bare_min_jit(H_uvsabc, P_uvjsabc, E_sabcj, nProj, nBands, nKa, nKb, nKc, nSpin, orbs_u, orbs_v):
    for u in orbs_u:
        for v in orbs_v:
            for j in range(nBands):
                for s in range(nSpin):
                    for a in range(nKa):
                        for b in range(nKb):
                            for c in range(nKc):
                                H_uvsabc[u, v, s, a, b, c] += P_uvjsabc[u, v, j, s, a, b, c] * E_sabcj[s, a, b, c, j]
    return H_uvsabc


def get_H_uvsabc_bare_min(P_uvjsabc, E_sabcj, orbs_u, orbs_v):
    shape = np.shape(P_uvjsabc)
    nProj = shape[0]
    nBands = shape[2]
    nSpin = shape[3]
    nKa = shape[4]
    nKb = shape[5]
    nKc = shape[6]
    H_uvsabc = np.zeros([nProj, nProj, nSpin, nKa, nKb, nKc], dtype=float)
    return get_H_uvsabc_bare_min_jit(H_uvsabc, P_uvjsabc, E_sabcj, nProj, nBands, nKa, nKb, nKc, nSpin, orbs_u, orbs_v)


@jit(nopython=True)
def get_pCOHP_sabcj_jit(nSpin, nKa, nKb, nKc, nBands, orbs_u, orbs_v, P_uvjsabc, H_uvsabc, wk_sabc, pCOHP_sabcj):
    for s in range(nSpin):
        for a in range(nKa):
            for b in range(nKb):
                for c in range(nKc):
                    for j in range(nBands):
                        uv_sum = 0
                        for u in orbs_u:
                            for v in orbs_v:
                                p1 = P_uvjsabc[u, v, j, s, a, b, c]
                                p2 = H_uvsabc[u, v, s, a, b, c]
                                p3 = wk_sabc[s, a, b, c]
                                uv_sum += np.real(p1 * p2) * p3
                        pCOHP_sabcj[s, a, b, c, j] += uv_sum
    return pCOHP_sabcj

@jit(nopython=True)
def get_pCOHP_sabcj_wweights_jit(nSpin, nKa, nKb, nKc, nBands, orbs_u, orbs_v, P_uvjsabc, H_uvsabc, wk_sabc, pCOHP_sabcj):
    for s in range(nSpin):
        for a in range(nKa):
            for b in range(nKb):
                for c in range(nKc):
                    for j in range(nBands):
                        uv_sum = 0
                        for u in orbs_u:
                            for v in orbs_v:
                                p1 = P_uvjsabc[u, v, j, s, a, b, c]
                                p2 = H_uvsabc[u, v, s, a, b, c]
                                p3 = wk_sabc[s, a, b, c]
                                p4 = abs(P_uvjsabc[u, u, j, s, a, b, c])
                                uv_sum += np.real(p1 * p2) * p3 * p4
                        pCOHP_sabcj[s, a, b, c, j] += uv_sum
    return pCOHP_sabcj


def get_pCOHP_sabcj(P_uvjsabc, H_uvsabc, orbs_u, orbs_v, wk_sabc=None):
    shape = np.shape(P_uvjsabc)
    nBands = shape[2]
    nSpin = shape[3]
    nKa = shape[4]
    nKb = shape[5]
    nKc = shape[6]
    pCOHP_sabcj = np.zeros([nSpin, nKa, nKb, nKc, nBands])
    if wk_sabc is None:
        wk_sabc = np.ones([nSpin, nKa, nKb, nKc])
    return get_pCOHP_sabcj_jit(nSpin, nKa, nKb, nKc, nBands, orbs_u, orbs_v, P_uvjsabc, H_uvsabc, wk_sabc, pCOHP_sabcj)

def get_pCOHP_sabcj_wweights(P_uvjsabc, H_uvsabc, orbs_u, orbs_v, wk_sabc=None):
    shape = np.shape(P_uvjsabc)
    nBands = shape[2]
    nSpin = shape[3]
    nKa = shape[4]
    nKb = shape[5]
    nKc = shape[6]
    pCOHP_sabcj = np.zeros([nSpin, nKa, nKb, nKc, nBands])
    if wk_sabc is None:
        wk_sabc = np.ones([nSpin, nKa, nKb, nKc])
    return get_pCOHP_sabcj_wweights_jit(nSpin, nKa, nKb, nKc, nBands, orbs_u, orbs_v, P_uvjsabc, H_uvsabc, wk_sabc, pCOHP_sabcj)


def get_pCOHP_tetr_weights_wweights(aidcs1, aidcs2, orbs_dict, kmap, P_uvjsabc, H_uvsabc, path=None, atoms=None, orb_func1=None,
                           orb_func2=None):
    us = get_us_aidcs(aidcs1, orbs_dict, kmap, orb_bool_func=orb_func1, path=path, atoms=atoms)
    vs = get_us_aidcs(aidcs2, orbs_dict, kmap, orb_bool_func=orb_func2, path=path, atoms=atoms)
    weights = get_pCOHP_sabcj_wweights(P_uvjsabc, H_uvsabc, us, vs)
    return weights

def get_pCOHP_tetr_weights(us, vs, orbs_dict, kmap, P_uvjsabc, H_uvsabc, path=None, atoms=None, orb_func1=None,
                           orb_func2=None):
    # us = get_us_aidcs(aidcs1, orbs_dict, kmap, orb_bool_func=orb_func1, path=path, atoms=atoms)
    # vs = get_us_aidcs(aidcs2, orbs_dict, kmap, orb_bool_func=orb_func2, path=path, atoms=atoms)
    weights = get_pCOHP_sabcj(P_uvjsabc, H_uvsabc, us, vs)
    return weights

def get_pCOHP_tetr_weights_worse(aidcs1, aidcs2, orbs_dict, kmap, P_uvjsabc, H_uvsabc, path=None, atoms=None, orb_func1=None,
                           orb_func2=None):
    us = get_us_aidcs(aidcs1, orbs_dict, kmap, orb_bool_func=orb_func1, path=path, atoms=atoms)
    vs = get_us_aidcs(aidcs2, orbs_dict, kmap, orb_bool_func=orb_func2, path=path, atoms=atoms)
    weights = get_pCOHP_sabcj(P_uvjsabc, H_uvsabc, us, vs)
    return weights

@jit(nopython=True)
def gauss(x, mu, sig):
    return np.exp(-((x - mu) ** 2) / sig)

@jit(nopython=True)
def get_hacky_cohp_jit(Erange, eflat, wflat, cflat, sig):
    for i in range(len(eflat)):
        cflat += gauss(Erange, eflat[i], sig)*wflat[i]
    return cflat


def get_hacky_cohp_helper(Erange, E_sabcj, weights_sabcj, sig):
    wup = weights_sabcj[0].flatten()
    wdn = weights_sabcj[1].flatten()
    eup = E_sabcj[0].flatten()
    edn = E_sabcj[1].flatten()
    cup = np.zeros(np.shape(Erange))
    cdn = np.zeros(np.shape(Erange))
    cup = get_hacky_cohp_jit(Erange, eup, wup, cup, sig)
    cdn = get_hacky_cohp_jit(Erange, edn, wdn, cdn, sig)
    return cup, cdn

import time

def get_hacky_cohp(idcs1, idcs2, path, data=None, res=0.01, sig=0.00001, orbs1=None, orbs2=None, Erange=None):
    start = time.time()
    if data is None:
        data = parse_data(root=path)
    proj_sabcju, E_sabcj, occ_sabcj, wk, ks, orbs_dict, mu = data
    atoms = get_atoms(path)
    kmap = get_kmap(atoms)
    if Erange is None:
        Erange = np.arange(np.min(E_sabcj)-(10*res), np.max(E_sabcj)+(10*res), res)
    orb_idcs = [[],[]]
    orbs_pulls = [orbs1, orbs2]
    for i, set in enumerate([idcs1, idcs2]):
        orbs_pull = orbs_pulls[i]
        if not orbs_pull is None:
            el_orb_u_dict = get_el_orb_u_dict(path, atoms, orbs_dict, set)
            for el in el_orb_u_dict:
                for orb in el_orb_u_dict[el]:
                    if type(orbs_pull) is list:
                        for orbi in orbs_pull:
                            if orbi in orb:
                                orb_idcs[i] += el_orb_u_dict[el][orb]
                                break
                    else:
                        if orbs_pull in orb:
                            orb_idcs[i] += el_orb_u_dict[el][orb]
        else:
            for idx in set:
                orb_idcs[i] += orbs_dict[kmap[idx]]
    orbs = orb_idcs[0] + orb_idcs[1]
    # end = time.time()
    # print(f"setup: {end - start}")
    # start = time.time()
    P_uvjsabc = get_P_uvjsabc_bare_min(proj_sabcju, orb_idcs[0], orb_idcs[1])
    # end = time.time()
    # print(f"P_uvjsabc: {end - start}")
    # start = time.time()
    H_uvsabc = get_H_uvsabc_bare_min(P_uvjsabc, E_sabcj, orb_idcs[0], orb_idcs[1])
    # end = time.time()
    # print(f"H_uvsabc: {end - start}")
    # start = time.time()
    # weights_sabcj = get_pCOHP_tetr_weights_worse(idcs1, idcs2, orbs_dict, kmap, P_uvjsabc, H_uvsabc, path=path, atoms=atoms)
    weights_sabcj = get_pCOHP_tetr_weights(orb_idcs[0], orb_idcs[1], orbs_dict, kmap, P_uvjsabc, H_uvsabc, path=path, atoms=atoms)
    # end = time.time()
    # print(f"weights_sabcj: {end - start}")
    # start = time.time()
    cup, cdn = get_hacky_cohp_helper(Erange, E_sabcj, weights_sabcj, sig)
    # end = time.time()
    # print(f"genning plot: {end - start}")
    return Erange, cup, cdn, weights_sabcj, E_sabcj

# def get_hacky_cohp(idcs1, idcs2, path, data=None, res=0.01, sig=0.00001, orbs1=None, orbs2=None, Erange=None):
#     start = time.time()
#     if data is None:
#         data = parse_data(root=path)
#     proj_sabcju, E_sabcj, occ_sabcj, wk, ks, orbs_dict, mu = data
#     atoms = get_atoms(path)
#     kmap = get_kmap(atoms)
#     if Erange is None:
#         Erange = np.arange(np.min(E_sabcj)-(10*res), np.max(E_sabcj)+(10*res), res)
#     orb_idcs = [[],[]]
#     orbs_pulls = [orbs1, orbs2]
#     for i, set in enumerate([idcs1, idcs2]):
#         orbs_pull = orbs_pulls[i]
#         if not orbs_pull is None:
#             el_orb_u_dict = get_el_orb_u_dict(path, atoms, orbs_dict, set)
#             for el in el_orb_u_dict:
#                 for orb in el_orb_u_dict[el]:
#                     if type(orbs_pull) is list:
#                         for orbi in orbs_pull:
#                             if orbi in orb:
#                                 orb_idcs[i] += el_orb_u_dict[el][orb]
#                                 break
#                     else:
#                         if orbs_pull in orb:
#                             orb_idcs[i] += el_orb_u_dict[el][orb]
#         else:
#             for idx in set:
#                 orb_idcs[i] += orbs_dict[kmap[idx]]
#     orbs = orb_idcs[0] + orb_idcs[1]
#     end = time.time()
#     print(f"setup: {end - start}")
#     start = time.time()
#     P_uvjsabc = get_P_uvjsabc_bare_min(proj_sabcju, orbs)
#     end = time.time()
#     print(f"P_uvjsabc: {end - start}")
#     start = time.time()
#     H_uvsabc = get_H_uvsabc_bare_min(P_uvjsabc, E_sabcj, orbs)
#     end = time.time()
#     print(f"H_uvsabc: {end - start}")
#     start = time.time()
#     weights_sabcj = get_pCOHP_tetr_weights(idcs1, idcs2, orbs_dict, kmap, P_uvjsabc, H_uvsabc, path=path, atoms=atoms)
#     end = time.time()
#     print(f"weights_sabcj: {end - start}")
#     start = time.time()
#     cup, cdn = get_hacky_cohp_helper(Erange, E_sabcj, weights_sabcj, sig)
#     end = time.time()
#     print(f"genning plot: {end - start}")
#     return Erange, cup, cdn, weights_sabcj, E_sabcj

def get_hacky_cohp_wweights(idcs1, idcs2, path, data=None, res=0.01, sig=0.00001, orbs1=None, orbs2=None, Erange=None):
    "Weights idcs1"
    if data is None:
        data = parse_data(root=path)
    proj_sabcju, E_sabcj, occ_sabcj, wk, ks, orbs_dict, mu = data
    atoms = get_atoms(path)
    kmap = get_kmap(atoms)
    if Erange is None:
        Erange = np.arange(np.min(E_sabcj)-(10*res), np.max(E_sabcj)+(10*res), res)
    orb_idcs = get_orb_idcs(idcs1, idcs2, path, orbs_dict, atoms, orbs1=orbs1, orbs2=orbs2)
    orbs = orb_idcs[0] + orb_idcs[1]
    P_uvjsabc = get_P_uvjsabc_bare_min(proj_sabcju, orbs)
    H_uvsabc = get_H_uvsabc_bare_min(P_uvjsabc, E_sabcj, orbs)
    weights_sabcj = get_pCOHP_tetr_weights_wweights(idcs1, idcs2, orbs_dict, kmap, P_uvjsabc, H_uvsabc, path=path, atoms=atoms)
    cup, cdn = get_hacky_cohp_helper(Erange, E_sabcj, weights_sabcj, sig)
    return Erange, cup, cdn, weights_sabcj, E_sabcj


def get_pdos_weights_sabcj(idcs, path, data, orb_bool_func):
    if data is None:
        data = parse_data(root=path)
    proj_sabcju, E_sabcj, occ_sabcj, wk, ks, orbs_dict, mu = data
    atoms = get_atoms(path)
    kmap = get_kmap(atoms)
    orbs = []
    if not orb_bool_func is None:
        el_orb_u_dict = get_el_orb_u_dict(path, atoms, orbs_dict, idcs)
        for el in el_orb_u_dict:
            for orb in el_orb_u_dict[el]:
                if orb_bool_func(orb):
                    orbs += el_orb_u_dict[el][orb]
    else:
        for idx in idcs:
            orbs += orbs_dict[kmap[idx]]
    weights_sabcj = np.zeros(np.shape(E_sabcj))
    for orb in orbs:
        weights_sabcj += np.abs(proj_sabcju[:,:,:,:,:,orb])
    return weights_sabcj

def get_hacky_pdos(idcs, path, data=None, res=0.01, sig=0.00001, orb_bool_func=None, Erange=None):
    if data is None:
        data = parse_data(root=path)
    proj_sabcju, E_sabcj, occ_sabcj, wk, ks, orbs_dict, mu = data
    weights_sabcj = get_pdos_weights_sabcj(idcs, path, data, orb_bool_func)
    if Erange is None:
        Erange = np.arange(np.min(E_sabcj)-(10*res), np.max(E_sabcj)+(10*res), res)
    cup, cdn = get_hacky_cohp_helper(Erange, E_sabcj, weights_sabcj, sig)
    return Erange, cup, cdn, weights_sabcj, E_sabcj


def hacky_plot_handler(xs, ys, ax, label, linestyle="solid", c=None, offy=0, spef_norm=False, norm_val=0, ev=True, evy=False, label_peaks=True, difcut=2, add_scatter=False):
    pxs = xs
    if ev:
        pxs = xs*Hartree
    ysp = ys
    if evy:
        ysp = ys*Hartree
    if type(ax) is np.ndarray:
        for axp in ax:
            axp.plot(pxs, ysp-offy, label=label, linestyle=linestyle, c=c)
            if add_scatter:
                axp.scatter(pxs, ysp-offy, c=c)
    else:
        ax.plot(pxs, ysp-offy, label=label, linestyle=linestyle, c=c)
        if add_scatter:
            ax.scatter(pxs, ysp-offy, c=c)
        if label_peaks:
            peaks, _ = scipy.signal.find_peaks(ys, height=0.1)
            ax.plot(pxs[peaks], (ysp-offy)[peaks], "x", c=c)
            difs = np.diff(pxs[peaks])
            for i, dif in enumerate(difs):
                if abs(dif) < difcut:
                    ax.text(pxs[peaks[i]], (ysp-offy)[peaks[i]], f"{dif:.2f}")
            peaks, _ = scipy.signal.find_peaks(ys*-1, height=0.1)
            ax.plot(pxs[peaks], (ysp-offy)[peaks], "x", c=c)
            difs = np.diff(pxs[peaks])
            for i, dif in enumerate(difs):
                if abs(dif) < difcut:
                    ax.text(pxs[peaks[i]], (ysp-offy)[peaks[i]], f"{dif:.2f}")
        if spef_norm:
            ax.text(max(pxs)+0.1, -offy, f"{norm_val:.3e}")

# def hacky_plot_handler(xs, ys, ax, label, linestyle="solid", c=None, offy=0, spef_norm=False, norm_val=0, ev=True, label_peaks=True, difcut=2):
#     pxs = xs
#     if ev:
#         pxs = xs*Hartree
#     if type(ax) is np.ndarray:
#         for axp in ax:
#             axp.plot(pxs, ys-offy, label=label, linestyle=linestyle, c=c)
#     else:
#         ax.plot(pxs, ys-offy, label=label, linestyle=linestyle, c=c)
#         if label_peaks:
#             peaks, _ = scipy.signal.find_peaks(ys, height=0.1)
#             ax.plot(pxs[peaks], (ys-offy)[peaks], "x", c=c)
#             difs = np.diff(pxs[peaks])
#             for i, dif in enumerate(difs):
#                 if abs(dif) < difcut:
#                     ax.text(pxs[peaks[i]], (ys-offy)[peaks[i]], f"{dif:.2f}")
#             peaks, _ = scipy.signal.find_peaks(ys*-1, height=0.1)
#             ax.plot(pxs[peaks], (ys-offy)[peaks], "x", c=c)
#             difs = np.diff(pxs[peaks])
#             for i, dif in enumerate(difs):
#                 if abs(dif) < difcut:
#                     ax.text(pxs[peaks[i]], (ys-offy)[peaks[i]], f"{dif:.2f}")
#         if spef_norm:
#             ax.text(max(pxs)+0.1, -offy, f"{norm_val:.3e}")
def plot_hacky_dos(ax, path, data=None, res=0.001, sig=0.0001, label=None, norm=True, linestyle="solid", c=None, offy=0, ev=True, label_peaks=False, spef_norm=False, Erange=None, norm_to_mu=False):
    Erange, cup, cdn, weights_sabcj, E_sabcj = get_hacky_pdos(list(range(len(get_atoms(path)))), path, data=data, res=res, sig=sig, orb_bool_func=None, Erange=Erange)
    if norm_to_mu:
        mu = data[-1]
        if ev:
            mu *= Hartree
        Erange -= mu
    if norm:
        scale = max(np.max(cup), np.max(cdn))
        cup *= 1/scale
        cdn *= 1/scale
    else:
        spef_norm = False
    hacky_plot_handler(Erange, cup, ax, label, c=c, linestyle=linestyle, offy=offy, ev=ev, label_peaks=label_peaks, spef_norm=spef_norm)
    if type(label) is str:
        label = "_" + label
    hacky_plot_handler(Erange, (-1)*cdn, ax, label, c=c, linestyle=linestyle, offy=offy, ev=ev, label_peaks=label_peaks)

def plot_hacky_pdos(ax, idcs, path, data=None, res=0.001, sig=0.0001, label=None, norm=True, orbs=None, linestyle="solid", c=None, offy=0, ev=True, label_peaks=False, spef_norm=False, Erange=None, norm_to_mu=False):
    orb_bool_func = None
    if not orbs is None:
        if type(orbs) is list:
            orb_bool_func = lambda s: True in [o in s for o in orbs]
        else:
            orb_bool_func = lambda s: orbs in s
    Erange, cup, cdn, weights_sabcj, E_sabcj = get_hacky_pdos(idcs, path, data=data, res=res, sig=sig, orb_bool_func=orb_bool_func, Erange=Erange)
    if norm_to_mu:
        mu = data[-1]
        # if ev:
        #     mu *= Hartree
        Erange -= mu
    if norm:
        scale = max(np.max(cup), np.max(cdn))
        cup *= 1/scale
        cdn *= 1/scale
    else:
        spef_norm = False
    hacky_plot_handler(Erange, cup, ax, label, c=c, linestyle=linestyle, offy=offy, ev=ev, label_peaks=label_peaks, spef_norm=spef_norm)
    if type(label) is str:
        label = "_" + label
    hacky_plot_handler(Erange, (-1)*cdn, ax, label, c=c, linestyle=linestyle, offy=offy, ev=ev, label_peaks=label_peaks)

def plot_hacky_poldos(ax, idcs, path, data=None, res=0.001, sig=0.0001, label=None, norm=True, orbs=None, linestyle="solid", c=None, offy=0, ev=True, label_peaks=False, spef_norm=False):
    orb_bool_func = None
    if not orbs is None:
        if type(orbs) is list:
            orb_bool_func = lambda s: True in [o in s for o in orbs]
        else:
            orb_bool_func = lambda s: orbs in s
    Erange, cup, cdn, weights_sabcj, E_sabcj = get_hacky_pdos(idcs, path, data=data, res=res, sig=sig, orb_bool_func=orb_bool_func)
    ctot = cup+cdn
    cpol = abs(cup-cdn)
    if norm:
        scale = max(np.max(ctot), np.max(cpol))
        ctot *= 1/scale
        cpol *= 1/scale
    else:
        spef_norm = False
    hacky_plot_handler(Erange, ctot, ax, label, c=c, linestyle=linestyle, offy=offy, ev=ev, label_peaks=label_peaks, spef_norm=spef_norm)
    if type(label) is str:
        label = "_" + label
    hacky_plot_handler(Erange, (-1)*cpol, ax, label, c=c, linestyle=linestyle, offy=offy, ev=ev, label_peaks=label_peaks, spef_norm=spef_norm)
    ax.axhline(y=-offy, color="black")


def plot_hacky_cohp_wweights(ax, idcs1, idcs2, path, data=None, res=0.001, sig=0.0001, label=None, norm=True, linestyle="solid", c=None, orbs1=None, orbs2=None, offy=0, spef_norm=False, ev=True, label_peaks=False, difcut=2, Erange=None, refline=False):
    Erange, cup1, cdn1, weights_sabcj, E_sabcj = get_hacky_cohp_wweights(idcs1, idcs2, path, data=data, res=res, sig=sig, orbs1=orbs1, orbs2=orbs2, Erange=Erange)
    cplot1 = cup1+cdn1
    norm_val = 0
    if norm:
        norm_val1 = np.max(abs(cplot1))
        cplot1 *= (1/norm_val1)
    else:
        spef_norm = False
    hacky_plot_handler(Erange, cplot1, ax, label, linestyle="dashed", c=c, offy=offy, spef_norm=spef_norm, norm_val=norm_val, ev=ev, label_peaks=label_peaks, difcut=difcut)
    if refline:
        ax.axhline(y=-offy, c="black", lw=0.3)

def plot_hacky_cohp(ax, idcs1, idcs2, path, data=None, res=0.001, sig=0.0001, label=None, norm=True, linestyle="solid", c=None, norm_to_mu=False, orbs1=None, orbs2=None, offy=0, spef_norm=False, ev=True, label_peaks=False, difcut=2, Erange=None, refline=False):
    Erange, cup, cdn, weights_sabcj, E_sabcj = get_hacky_cohp(idcs1, idcs2, path, data=data, res=res, sig=sig, orbs1=orbs1, orbs2=orbs2, Erange=Erange)
    cplot = cup+cdn
    norm_val = 0
    if norm:
        norm_val = np.max(abs(cplot))
        cplot *= (1/norm_val)
    else:
        spef_norm = False
    if norm_to_mu:
        mu = data[-1]
        Erange -= mu
    hacky_plot_handler(Erange, cplot, ax, label, linestyle=linestyle, c=c, offy=offy, spef_norm=spef_norm, norm_val=norm_val, ev=ev, label_peaks=label_peaks, difcut=difcut)
    if refline:
        ax.axhline(y=-offy, c="black", lw=0.3)

def get_orb_idcs(idcs1, idcs2, path, orbs_dict, atoms, orbs1=None, orbs2=None):
    orb_idcs = [[],[]]
    orbs_pulls = [orbs1, orbs2]
    kmap = get_kmap(atoms)
    for i, set in enumerate([idcs1, idcs2]):
        orbs_pull = orbs_pulls[i]
        if not orbs_pull is None:
            el_orb_u_dict = get_el_orb_u_dict(path, atoms, orbs_dict, set)
            for el in el_orb_u_dict:
                for orb in el_orb_u_dict[el]:
                    if type(orbs_pull) is list:
                        for orbi in orbs_pull:
                            if orbi in orb:
                                orb_idcs[i] += el_orb_u_dict[el][orb]
                                break
                    else:
                        if orbs_pull in orb:
                            orb_idcs[i] += el_orb_u_dict[el][orb]
        else:
            for idx in set:
                orb_idcs[i] += orbs_dict[kmap[idx]]
    return orb_idcs
def get_hacky_cohp_pieces(idcs1, idcs2, path, data=None, orbs1=None, orbs2=None):
    if data is None:
        data = parse_data(root=path)
    proj_sabcju, E_sabcj, occ_sabcj, wk, ks, orbs_dict, mu = data
    atoms = get_atoms(path)
    orb_idcs = get_orb_idcs(idcs1, idcs2, path, orbs_dict, atoms, orbs1=orbs1, orbs2=orbs2)
    orbs = orb_idcs[0] + orb_idcs[1]
    P_uvjsabc = get_P_uvjsabc_bare_min(proj_sabcju, orb_idcs[0], orb_idcs[1])
    H_uvsabc = get_H_uvsabc_bare_min(P_uvjsabc, E_sabcj, orb_idcs[0], orb_idcs[1])
    return E_sabcj, P_uvjsabc, H_uvsabc, orbs_dict, atoms, wk, occ_sabcj


# @jit(nopython=True)
def int_hacky_cohp_pieces_jit(E_sabcj, weights_sabcj, wk_sabc, nSpin, nKa, nKb, nKc, nBands, integs, nrgs):
    for j in range(nBands):
        hold_sum = 0
        for s in range(nSpin):
            for a in range(nKa):
                for b in range(nKb):
                    for c in range(nKc):
                        hold_sum += weights_sabcj[s,a,b,c,j]*wk_sabc[s,a,b,c]
        nrg = np.mean(E_sabcj[:,:,:,:,j])
        integ = integs[-1] + hold_sum
        nrgs.append(nrg)
        integs.append(integ)
    return nrgs, integs[1:]

# @jit(nopython=True)
def int_hacky_cohp_occ_pieces_jit(E_sabcj, weights_sabcj, wk_sabc, occ_sabcj, nSpin, nKa, nKb, nKc, nBands, integs, nrgs):
    for j in range(nBands):
        hold_sum = 0
        for s in range(nSpin):
            for a in range(nKa):
                for b in range(nKb):
                    for c in range(nKc):
                        hold_sum += weights_sabcj[s,a,b,c,j]*wk_sabc[s,a,b,c]*occ_sabcj[s,a,b,c,j]
        nrg = np.mean(E_sabcj[:, :, :, :, j])
        integ = integs[-1] + hold_sum
        nrgs.append(nrg)
        integs.append(integ)
    return nrgs, integs[1:]

def int_hacky_cohp_occ_pieces(E_sabcj, weights_sabcj, wk_sabc, occ_sabcj):
    nrgs = []
    integs = [0]
    shape = np.shape(E_sabcj)
    nSpin = shape[0]
    nKa = shape[1]
    nKb = shape[2]
    nKc = shape[3]
    nBands = shape[4]
    # nrgs = np.zeros(nBands)
    # integs = np.zeros(nBands)
    nrgs, integs = int_hacky_cohp_occ_pieces_jit(E_sabcj, weights_sabcj, wk_sabc, occ_sabcj, nSpin, nKa, nKb, nKc, nBands, integs, nrgs)
    return nrgs, integs


def int_hacky_cohp_pieces(E_sabcj, weights_sabcj, wk_sabc):
    nrgs = []
    integs = [0]
    shape = np.shape(E_sabcj)
    nSpin = shape[0]
    nKa = shape[1]
    nKb = shape[2]
    nKc = shape[3]
    nBands = shape[4]
    nrgs, integs = int_hacky_cohp_pieces_jit(E_sabcj, weights_sabcj, wk_sabc, nSpin, nKa, nKb, nKc, nBands, integs, nrgs)
    return nrgs, integs




def change_wk_shape(wk_sabc, nBands):
    shape = np.shape(wk_sabc)
    nSpin = shape[0]
    nKa = shape[1]
    nKb = shape[2]
    nKc = shape[3]
    wk_sabcj = np.zeros([nSpin, nKa, nKb, nKc, nBands])
    for s in range(nSpin):
        for a in range(nKa):
            for b in range(nKb):
                for c in range(nKc):
                    wk_sabcj[s,a,b,c,:] += wk_sabc[s,a,b,c]
    return wk_sabcj

@jit(nopython=True)
def int_hacky_cohp_pieces_flat_jit(E_sabcj_f, weights_sabcj_f, wk_sabcj_f, idcs, integs, nrgs):
    for i, idx in enumerate(idcs):
        nrg = E_sabcj_f[idx]
        integ = integs[i-1] + weights_sabcj_f[idx]*wk_sabcj_f[idx]
        nrgs[i] += nrg
        integs[i] += integ
    return nrgs, integs
    #     nrgs.append(nrg)
    #     integs.append(integ)
    # return nrgs, integs[1:]

def int_hacky_cohp_pieces_flat(E_sabcj, weights_sabcj, wk_sabc):
    shape = np.shape(E_sabcj)
    nSpin = shape[0]
    nKa = shape[1]
    nKb = shape[2]
    nKc = shape[3]
    nBands = shape[4]
    E_sabcj_f = E_sabcj.flatten()
    weights_sabcj_f = weights_sabcj.flatten()
    wk_sabcj = change_wk_shape(wk_sabc, nBands)
    wk_sabcj_f = wk_sabcj.flatten()
    idcs = np.argsort(E_sabcj_f)
    # nrgs = []
    # integs = [0]
    nrgs = np.zeros(len(idcs))
    integs = np.zeros(len(idcs))
    nrgs, integs = int_hacky_cohp_pieces_flat_jit(E_sabcj_f, weights_sabcj_f, wk_sabcj_f, idcs, integs, nrgs)
    return nrgs, integs
def get_hacky_icohp(idcs1, idcs2, path, data=None,  orbs1=None, orbs2=None, use_occs=False):
    if use_occs:
        print("Occs not implemented anymore")
    E_sabcj, P_uvjsabc, H_uvsabc, orbs_dict, atoms, wk, occ_sabcj = get_hacky_cohp_pieces(idcs1, idcs2, path, data=data, orbs1=orbs1, orbs2=orbs2)
    orb_idcs = get_orb_idcs(idcs1, idcs2, path, orbs_dict, atoms, orbs1=orbs1, orbs2=orbs2)
    weights_sabcj = get_pCOHP_tetr_weights(orb_idcs[0], orb_idcs[1], orbs_dict, get_kmap(atoms), P_uvjsabc, H_uvsabc, path=path, atoms=atoms)
    nrgs, ipcohp = int_hacky_cohp_pieces_flat(E_sabcj, weights_sabcj, wk)
    return np.array(nrgs), np.array(ipcohp)

def icohp_plot_hack(_nrgs, _ipcohp, fake_split=0.000001):
    nrgs =[_nrgs[0]]
    ipcohp = [_ipcohp[0]]
    for i, nrg in enumerate(_nrgs[1:]):
        nrgs.append(nrg-fake_split)
        ipcohp.append(ipcohp[-1])
        nrgs.append(nrg)
        ipcohp.append(_ipcohp[i+1])
    return np.array(nrgs), np.array(ipcohp)

def plot_hacky_icohp(ax, idcs1, idcs2, path, data=None, label=None, linestyle="solid", c=None, orbs1=None, orbs2=None, offy=0, evx=True, evy=True, refline=False, add_scatter=True, use_occs=False):
    nrgs, ipcohp = get_hacky_icohp(idcs1, idcs2, path, data=data, orbs1=orbs1, orbs2=orbs2, use_occs=use_occs)
    nrgs, ipcohp = icohp_plot_hack(nrgs, ipcohp)
    hacky_plot_handler(nrgs, ipcohp, ax, label, linestyle=linestyle, c=c, offy=offy, ev=evx, evy=evy, label_peaks=False, add_scatter=add_scatter)
    if refline:
        ax.axhline(y=-offy, c="black", lw=0.3)

v0 = 4.66
def bias_to_mu(bias_str):
    voltage = float(bias_str.rstrip("V")) + v0
    mu = - voltage / Hartree
    return mu

@jit(nopython=True)
def get_just_icohp_helper_jit(occ_sabcj, weights_sabcj, wk_sabc, nSpin, nKa, nKb, nKc, nBands, icohp):
    for s in range(nSpin):
        for a in range(nKa):
            for b in range(nKb):
                for c in range(nKc):
                    for j in range(nBands):
                        icohp += occ_sabcj[s,a,b,c,j]*weights_sabcj[s,a,b,c,j]*wk_sabc[s,a,b,c]
    return icohp

def get_just_icohp_helper(occ_sabcj, weights_sabcj, wk):
    shape = np.shape(occ_sabcj)
    nSpin = shape[0]
    nKa = shape[1]
    nKb = shape[2]
    nKc = shape[3]
    nBands = shape[4]
    icohp = 0
    icohp = get_just_icohp_helper_jit(occ_sabcj, weights_sabcj, wk, nSpin, nKa, nKb, nKc, nBands, icohp)
    # div = np.prod([nKa, nKb, nKc])
    # icohp *= (1/div)
    return icohp


def get_just_icohp(idcs1, idcs2, path, data=None, orbs1=None, orbs2=None):
    if data is None:
        data = parse_data(root=path)
    E_sabcj, P_uvjsabc, H_uvsabc, orbs_dict, atoms, wk, occ_sabcj = get_hacky_cohp_pieces(idcs1, idcs2, path, data=data, orbs1=orbs1, orbs2=orbs2)
    orb_idcs = get_orb_idcs(idcs1, idcs2, path, orbs_dict, atoms, orbs1=orbs1, orbs2=orbs2)
    kmap = get_kmap(atoms)
    weights_sabcj = get_pCOHP_tetr_weights(orb_idcs[0], orb_idcs[1], orbs_dict, kmap, P_uvjsabc, H_uvsabc, path=path, atoms=atoms)
    icohp = get_just_icohp_helper(occ_sabcj, weights_sabcj, wk)
    return icohp

def get_mu(outfile):
    return jfunc.get_mu(outfile)