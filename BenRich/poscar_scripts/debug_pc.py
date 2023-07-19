from interp import Poscar
import numpy as np
import os


# path = 'C:\\Users\\User\\Desktop\\backup[s\\1-19-2023\\img_intrx\\calcs\\surfs'
path = 'C:\\Users\\User\\Desktop\\backup[s\\1-19-2023\\mnc-pt-pyrid_archive\\1n\\'
os.chdir(path)

cu_z12 = Poscar('Cu_Ap2-5_Bp2-5_Cp1-5_Z12\\No_bias\\POSCAR', pc=True)

space_z_per_layer = abs(float(cu_z12.posns[0][2]) - float(cu_z12.posns[1][2])) * float(cu_z12.lattice[2][2]) * float(cu_z12.lat_scale)
# needed_z = space_z_per_layer * 4
needed_z = 16.0
z_rescale_factor = needed_z / (float(cu_z12.lattice[2][2]) * float(cu_z12.lat_scale))

cu_z12.rescale_unit_cell([1., 1., z_rescale_factor], freeze=True)
cu_z12.dump_new_poscar('local\\POSCAR')

#print(z_rescale_factor)

# cu_tube = Poscar('../gc_backup/calcs/surfs/Cu_Ap1-5m0-5_Bp2-0m0-0_Cp2-0m0-0/POSCAR')
# cu_tube.center_posns()
# # outside = cu_tube.count_by_lambda_booler(lambda abc: cu_tube.outside_cylinder(abc, 2.6, [.5, .5, .5], [0., 1.0, 0.], eval_in_cart=True))
# # for n in outside:
# #     print(n + 1)
# cu_tube.unfreeze_by_lambda_booler(
#     lambda abc: cu_tube.outside_cylinder(abc, 2.6, [.5,.5,.5], [0., 1., 0.], eval_in_cart=True),
#     b = False
# )
# # try_rads = np.arange(0.05, 0.15, 0.02)
# # for r in try_rads:
# #     print(r)
# #     print(len(cu_tube.count_by_lambda_booler(lambda abc: cu_tube.outside_cylinder(abc, r, [.5, .5, .5], [0., 1.0, 0.]))))
#
# cu_tube.dump_new_poscar('dump_new/POSCAR')