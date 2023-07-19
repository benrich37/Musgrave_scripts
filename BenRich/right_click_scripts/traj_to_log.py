"Takes an ASE optimization traj file, and turns it into a fake gaussian .logx output file"
"Note: The energies within the .logx file may not be the true SCF energies found, but the force-consistent energies used in the geometry optimization"
"WARNING: WILL NOT WORK WITH NEB TRAJS"
import sys
file = sys.argv[1]
from ase.io.trajectory import TrajectoryReader

assert ".traj" in file

def traj_to_log_str(traj):
    dump_str = "\n Entering Link 1 \n \n"
    nSteps = len(traj)
    for i in range(nSteps):
        dump_str += log_input_orientation(traj[i])
        dump_str += scf_str(traj[i])
        dump_str += opt_spacer(i, nSteps)
    dump_str += log_input_orientation(traj[-1])
    dump_str += " Normal termination of Gaussian 16"
    return dump_str

def scf_str(atoms):
    return f"\n SCF Done:  E =  {atoms.get_potential_energy()}\n\n"

def opt_spacer(i, nSteps):
    dump_str = "\n GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad\n"
    dump_str += f"\n Step number   {i+1}\n"
    if i == nSteps:
        dump_str += " Optimization completed.\n"
        dump_str += "    -- Stationary point found.\n"
    dump_str += "\n GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad\n"
    return dump_str

def log_input_orientation(atoms):
    dump_str = "                          Input orientation:                          \n"
    dump_str += " ---------------------------------------------------------------------\n"
    dump_str += " Center     Atomic      Atomic             Coordinates (Angstroms)\n"
    dump_str += " Number     Number       Type             X           Y           Z\n"
    dump_str += " ---------------------------------------------------------------------"
    at_ns = atoms.get_atomic_numbers()
    at_posns = atoms.positions
    for i in range(len(at_posns)):
        dump_str += f" {i+1} {at_ns[i]} 0 "
        for j in range(3):
            dump_str += f"{at_posns[i][j]} "
        dump_str += "\n"
    dump_str += " ---------------------------------------------------------------------\n"
    return dump_str

traj = TrajectoryReader(file)
with open(file[:file.index(".traj")] + "_traj.logx", "w") as f:
    f.write(traj_to_log_str(traj))
    f.close()