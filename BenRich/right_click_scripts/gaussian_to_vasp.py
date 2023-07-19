"Takes a gaussian .gjf file, and saves a vasp-style file"
import sys
file = sys.argv[1]
from ase.io import read, write

try:
    atoms = read(file, format="gaussian-in")
except Exception as e:
    print(e)
    exit()
fname_new = file[:file.index('.gjf')]
write(fname_new, atoms, format="vasp")