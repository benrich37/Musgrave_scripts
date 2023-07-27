"""
Use me on a JDFTx out file running an ionic/lattice minimization
(pairs well with monitoring a lattice optimization ran through https://github.com/benrich37/perlStuff/blob/master/opt.py)
"""


import sys
file = sys.argv[1]
import numpy as np
from ase.units import Bohr
from ase import Atoms, Atom


def opt_spacer(i, nSteps):
    dump_str = "\n GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad\n"
    dump_str += f"\n Step number   {i+1}\n"
    if i == nSteps:
        dump_str += " Optimization completed.\n"
        dump_str += "    -- Stationary point found.\n"
    dump_str += "\n GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad\n"
    return dump_str

def scf_str(atoms, e_conv=(1/27.211397)):
    got_it = False
    try:
        E = atoms.get_potential_energy()
        got_it = True
    except:
        pass
    try:
        E = atoms.E
        got_it = True
    except:
        pass
    if not got_it:
        E = 0
    return f"\n SCF Done:  E =  {0*e_conv}\n\n"

def log_input_orientation(atoms, do_cell=False):
    dump_str = "                          Input orientation:                          \n"
    dump_str += " ---------------------------------------------------------------------\n"
    dump_str += " Center     Atomic      Atomic             Coordinates (Angstroms)\n"
    dump_str += " Number     Number       Type             X           Y           Z\n"
    dump_str += " ---------------------------------------------------------------------\n"
    at_ns = atoms.get_atomic_numbers()
    at_posns = atoms.positions
    nAtoms = len(at_ns)
    for i in range(nAtoms):
        dump_str += f" {i+1} {at_ns[i]} 0 "
        for j in range(3):
            dump_str += f"{at_posns[i][j]} "
        dump_str += "\n"
    if do_cell:
        cell = atoms.cell
        for i in range(3):
            dump_str += f"{i + nAtoms + 1} -2 0 "
            for j in range(3):
                dump_str += f"{cell[i][j]} "
            dump_str += "\n"
    dump_str += " ---------------------------------------------------------------------\n"
    return dump_str

def get_charges(atoms):
    es = []
    charges = None
    try:
        charges = atoms.get_charges()
    except Exception as e:
        es.append(e)
        pass
    if charges is None:
        try:
            charges = atoms.charges
        except Exception as e:
            es.append(e)
            pass
    if charges is None:
        try:
            charges = atoms.arrays["initial_charges"]
        except Exception as e:
            es.append(e)
            print(es)
            assert False
    return charges

def log_charges(atoms):
    try:
        charges = get_charges(atoms)
        nAtoms = len(atoms.positions)
        symbols = atoms.get_chemical_symbols()
    except:
        return " "
    dump_str = " **********************************************************************\n\n"
    dump_str += "            Population analysis using the SCF Density.\n\n"
    dump_str = " **********************************************************************\n\n Mulliken charges:\n    1\n"
    for i in range(nAtoms):
        dump_str += f"{int(i+1)} {symbols[i]} {charges[i]} \n"
    dump_str += f" Sum of Mulliken charges = {np.sum(charges)}\n"
    return dump_str

def get_do_cell(pbc):
    return np.sum(pbc) > 0


def get_start_line(outfname):
    start = 0
    for i, line in enumerate(open(outfname)):
        if "JDFTx 1." in line:
            start = i
    return start


def get_atoms_from_outfile_data(names, posns, R, charges=None, E=0):
    atoms = Atoms()
    posns *= Bohr
    R = R.T*Bohr
    atoms.cell = R
    if charges is None:
        charges = np.zeros(len(names))
    for i in range(len(names)):
        atoms.append(Atom(names[i], posns[i], charge=charges[i]))
    atoms.E = E
    return atoms


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
    return R, posns, names, chargeDir, active_posns, active_lowdin, active_lattice, posns, coords, idxMap, j, lat_row, \
        new_posn, log_vars, E, charges


def get_atoms_list_from_out(outfile):
    start = get_start_line(outfile)
    charge_key = "oxidation-state"
    opts = []
    nAtoms = None
    R, posns, names, chargeDir, active_posns, active_lowdin, active_lattice, posns, coords, idxMap, j, lat_row, \
        new_posn, log_vars, E, charges = get_atoms_list_from_out_reset_vars()
    for i, line in enumerate(open(outfile)):
        if i > start:
            if new_posn:
                if "Lowdin population analysis " in line:
                    active_lowdin = True
                if "R =" in line:
                    active_lattice = True
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
                        posns=np.array(posns)
                        active_posns = False
                        nAtoms = len(names)
                        if len(charges) < nAtoms:
                            charges=np.zeros(nAtoms)
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
                    if coords != 'cartesian':
                        posns = np.dot(posns[1], R[2])
                    opts.append(get_atoms_from_outfile_data(names, posns, R, charges=charges, E=E))
                    R, posns, names, chargeDir, active_posns, active_lowdin, active_lattice, posns, coords, idxMap, j, lat_row, \
                        new_posn, log_vars, E, charges = get_atoms_list_from_out_reset_vars(nAtoms=nAtoms)
            elif "Computing DFT-D3 correction:" in line:
                new_posn = True
    return opts

def is_done(outfile):
    start_line = get_start_line(outfile)
    done = False
    with open(outfile, "r") as f:
        for i, line in enumerate(f):
            if i > start_line:
                if "Minimize: Iter:" in line:
                    done = False
                elif "Minimize: Converged" in line:
                    done = True
    return done



def out_to_logx_str(outfile, e_conv=(1/27.211397)):
    atoms_list = get_atoms_list_from_out(outfile)
    dump_str = "\n Entering Link 1 \n \n"
    do_cell = get_do_cell(atoms_list[0].cell)
    for i in range(len(atoms_list)):
        dump_str += log_input_orientation(atoms_list[i], do_cell=do_cell)
        dump_str += f"\n SCF Done:  E =  {atoms_list[i].E*e_conv}\n\n"
        dump_str += log_charges(atoms_list[i])
        dump_str += opt_spacer(i, len(atoms_list))
    if is_done(outfile):
        dump_str += log_input_orientation(atoms_list[-1])
        dump_str += " Normal termination of Gaussian 16"
    return dump_str

assert "out" in file
with open(file + ".logx", "w") as f:
    f.write(out_to_logx_str(file))
    f.close()