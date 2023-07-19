"Creates a new vasp-style file where the center of mass is aligned with the center of the cell along the Z-axis"
"Also creates a duplicate _backup of the original just in case"
import sys
file = sys.argv[1]
from ase.io import read, write
import numpy as np
def shift_com(file):
    atoms = read(file, format="vasp")
    write(file + "_backup", atoms, format="vasp")
    com = atoms.get_center_of_mass()
    coc = get_coc(atoms)
    disp_z = np.array([0., 0., coc[2] - com[2]])
    atoms = shift_atoms(atoms, disp_z)
    write(file, atoms, format="vasp")
    return atoms

def get_coc(atoms):
    cell = atoms.get_cell()
    coc = np.zeros(3)
    for v in cell:
        coc += v/2.
    return coc

def shift_atoms(atoms, vector):
    for i in range(len(atoms.positions)):
        atoms.positions[i] += vector
    return atoms
try:
    shift_com(file)
except Exception as e:
    print(e)
    exit()