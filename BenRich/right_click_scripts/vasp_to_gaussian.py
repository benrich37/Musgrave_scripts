import sys
file = sys.argv[1]
from ase.io import read, write
try:
    atoms = read(file, format="vasp")
except Exception as e:
    print(e)
    exit()
write(file + ".gjf", atoms, format="gaussian-in")