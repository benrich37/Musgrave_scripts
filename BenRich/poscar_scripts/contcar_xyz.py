#from interp import Poscar
import os
os.chdir('/Users/richb/Desktop')
#salt = Poscar('CONTCAR')


contcar = []

with open('CONTCAR') as file:
    for line in file:
        contcar.append(
            list(filter(lambda x: len(x) > 0, line.rstrip('\n').split(' ')))
        )

rel_posns = contcar[8:]
scales = contcar[2:5]
scale_factors = [float(scales[0][0]), float(scales[1][1]), float(scales[2][2])]

abs_posns = []
for p in rel_posns:
    abs_posns.append([])
    for i in range(3):
        abs_posns[-1].append(float(p[i])*scale_factors[i] - 8.0)

atom_ids = contcar[5]
atom_counts = contcar[6]
print(atom_ids)
print(atom_counts)

last = 0
for i in range(len(atom_counts)):
    for j in range(int(atom_counts[i])):
        abs_posns[last].insert(0, atom_ids[i])
        last += 1

dump_str = ''
for p in abs_posns:
    dump_str += p[0]
    dump_str += '      '
    for i in range(3):
        dump_str += str(p[i + 1])
        dump_str += ' '
    dump_str += '\n'

file = open('test.xyz', 'w')
file.write(dump_str)
file.close()



