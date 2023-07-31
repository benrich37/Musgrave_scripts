import sys
file = sys.argv[1]
from ase.io import read, write
from ase.build import sort

format="vasp"
if (".com" in file) or (".gjf") in file:
    format = "gaussian-in"

atoms = read(file, format=format)
atoms = sort(atoms)
write(file, atoms, format=format)