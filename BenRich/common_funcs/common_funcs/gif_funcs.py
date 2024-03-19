import matplotlib.pyplot as plt
from ase.visualize import view
import numpy as np
from ase.io import read, write
from ase.visualize.plot import plot_atoms
import os
import imageio
from datetime import datetime
from pathlib import Path
from ase.io.trajectory import TrajectoryReader
def my_join(p1, p2, splitter="//"):
    p1 = str(p1)
    p2 = str(p2)
    if (p1[-len(splitter):] != splitter) and (p2[:len(splitter)] != splitter):
        p1 += splitter
    elif (p1[-len(splitter):] == splitter) and (p2[:len(splitter)] == splitter):
        p1 = p1[:-len(splitter)]
    return p1 + p2

def read_f(dir):
    with open(dir + "//Ecomponents") as f:
        for line in f:
            if "F =" in line:
                return float(line.strip().split("=")[1])
def get_dist(i, data_dir, i1=49, i2=50, shift = -1):
    img_dir = my_join(data_dir, str(i))
    atoms = read(my_join(img_dir, "CONTCAR"), format="vasp")
    return np.linalg.norm(atoms.positions[i2 + shift] - atoms.positions[i1 + shift])

def get_start_line(fname):
    start = 0
    with open(fname, 'r') as f:
        for i, line in enumerate(f):
            if "*******" in line:
                start = i
    return start


def get_lowdin_charges(dir):
    atoms = read(os.path.join(dir, "CONTCAR"))
    charges_dir = {}
    indices_dir = {}
    read_f = dir + "//out"
    start = get_start_line(read_f)
    with open(read_f, 'r') as f:
        for i, line in enumerate(f):
            if i > start:
                if "Lowdin population analysis" in line:
                    charges_dir = {}
                if "oxidation-state" in line:
                    lsplit = line.split(" ")
                    ion = lsplit[2].lower()
                    charges_dir[ion] = []
                    for ch in lsplit[3:]:
                        charges_dir[ion].append(float(ch))
    ions = atoms.get_chemical_symbols()
    for i in range(len(ions)):
        if not ions[i].lower() in indices_dir.keys():
            indices_dir[ions[i].lower()] = []
        indices_dir[ions[i].lower()].append(i)
    try:
        assert(len(charges_dir.keys()) == len(indices_dir.keys()))
    except:
        print(dir)
        print(charges_dir.keys())
        print(indices_dir.keys())
    charges = np.zeros(len(ions))
    for a in list(indices_dir.keys()):
        for i, j in enumerate(indices_dir[a]):
            charges[j] += charges_dir[a][i]
    return charges

import colorsys

def float_to_color(float_array):
    min_val = min(float_array)
    max_val = max(float_array)
    norm_array = [(i - min_val) / (max_val - min_val) for i in float_array]
    color_array = [colorsys.hsv_to_rgb(value / 2, 1, 1) for value in norm_array]
    hex_color_array = ['#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255)) for r, g, b in color_array]
    return hex_color_array


def list_integer_dirs(directory, strict = True):
    """Returns a list of directories within the given directory that are named with integers"""
    integer_dirs = []
    for subdir in Path(directory).iterdir():
        if subdir.is_dir() and subdir.name.isdigit():
            if strict:
                if os.path.exists(os.path.join(subdir, "Ecomponents")) and os.path.exists(os.path.join(subdir, "out")):
                    integer_dirs.append(subdir)
            else:
                integer_dirs.append(subdir)
    return integer_dirs

def get_ref_range(data_dir):
    dirs = list_integer_dirs(data_dir)
    ref_range = []
    for d in dirs:
        ref_range.append(int(str(d).split("\\")[-1]))
    return ref_range

def process_rots(rots):
    rots_ref = [0, 0, 0, 0, 0, 0]
    if rots is None:
        rz1 = 80
        rx1 = 30
        ry1 = 0
        rz2 = -10
        rx2 = 15
        ry2 = 0
        rots_ref = [rz1, rx1, ry1, rz2, rx2, ry2]
    else:
        for i in range(len(rots)):
            rots_ref[i] += rots[i]
        if len(rots) < 4:
            rots_ref[3] += rots_ref[0] - 90
    return f'{int(rots_ref[0])}z,{int(rots_ref[1])}x,{int(rots_ref[2])}z', f'{int(rots_ref[3])}z,{int(rots_ref[4])}x,{int(rots_ref[5])}z'


def see_surf(idx, i1, i2, ref_range, atoms_list, fs, data_dir, colors=None, rots=None, ref_x=None, text_str = None, show_max=True, xlabel=None, save=False):
    fig = plt.figure(figsize=(8, 12))
    ax1 = plt.subplot2grid((3,2), (1,0), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid((3,2), (1,1), rowspan=1, colspan=1)
    ax3 = plt.subplot2grid((3,2), (0,0), rowspan=1, colspan=2)
    ax4 = plt.subplot2grid((3,2), (2,0), rowspan=1, colspan=2)
    if ref_x is None:
        ref_x = []
        for i in ref_range:
            ref_x.append(get_dist(i, i1=i1, i2=i2))
    atoms = atoms_list[idx]
    r1, r2 = process_rots(rots)
    if not colors is None:
        plot_atoms(atoms, ax1, rotation=r1, colors=colors[idx])
        plot_atoms(atoms, ax2, rotation=r2, colors=colors[idx])
    else:
        plot_atoms(atoms, ax1, rotation=r1)
        plot_atoms(atoms, ax2, rotation=r2)
    ax1.set_ylim(0, 10)
    ax2.set_ylim(0, 10)
    ax1.set_xlim(5, 15)
    ax2.set_xlim(5, 15)
    ax3.plot(ref_x, fs)
    ax4.plot(range(len(fs)), ref_x)
    ax4.set_xlabel("Step")
    ax3.scatter(ref_x[idx], fs[idx], label="image " + str(idx), color=["C0"])
    ax4.scatter([idx], ref_x[idx], color=["C0"])
    if not xlabel is None:
        ax3.set_xlabel(xlabel)
        ax4.set_ylabel(xlabel)
    else:
        ax3.set_xlabel(f"{atoms.get_chemical_symbols()[i1-1]}({str(i1)}) - {atoms.get_chemical_symbols()[i2-1]}({str(i2)}) dist (Angstrom)")
        ax4.set_ylabel(f"{atoms.get_chemical_symbols()[i1-1]}({str(i1)}) - {atoms.get_chemical_symbols()[i2-1]}({str(i2)}) dist (Angstrom)")
    ax3.ticklabel_format(useOffset=False)
    if show_max:
        max_f = np.max(fs)
        max_idx = fs.index(max_f)
        ax3.scatter(ref_x[max_idx], fs[max_idx], label=f"Barrier: {max_f:.{7}g}", color="red", marker="x")
    ax3.legend()
    if not text_str is None:
        fig.text(0.05, 0.05, text_str, fontsize='xx-small')
    if save:
        fname = data_dir + "pngs//" + str(idx) + ".png"
        plt.savefig(fname)
        plt.close()
        return imageio.v2.imread(fname)


def map_2d_idcs(track_idcs):
    i1 = track_idcs[0]
    i2 = track_idcs[1]
    i3 = track_idcs[-2]
    i4 = track_idcs[-1]
    return i1, i2, i3, i4

def see_surf_iter(iter, idx, track_idcs, data_dir, iter_fs, iters, rots=None, ref_x = None, text_str = None, show_max=True, xlabel=None, ref_bounds = None, interp_contour = False, max_step=0):
    fs = iter_fs
    iter_dir = my_join(data_dir, "iters")
    iter_iter_dir = my_join(iter_dir, str(iter))
    img_iter_iter_dir = my_join(iter_iter_dir, str(idx))
    _2d = len(track_idcs) >= 3
    i1, i2, i3, i4 = map_2d_idcs(track_idcs)
    r1, r2 = process_rots(rots)
    ref_y = fs
    ref_range = range(iter + 1)
    if ref_x is None:
        ref_x = []
        ref_y = []
        if _2d:
            for i in ref_range:
                ref_x.append(get_dist(i, i1=i1, i2=i2, dirr=iter_iter_dir))
                ref_y.append(get_dist(i, i1=i3, i2=i4, dirr=iter_iter_dir))
        else:
            for i in ref_range:
                ref_x.append(get_dist(i, i1=i1, i2=i2, dirr=iter_iter_dir))
    fig = plt.figure(figsize=(8, 12))
    ax1 = plt.subplot2grid((3,2), (1,0), rowspan=1, colspan=1)
    ax2 = plt.subplot2grid((3,2), (1,1), rowspan=1, colspan=1)
    ax3 = plt.subplot2grid((3,2), (0,0), rowspan=1, colspan=2)
    ax4 = plt.subplot2grid((3,2), (2,0), rowspan=1, colspan=2)
    ax5 = ax4.twinx()
    if _2d:
        if interp_contour:
            x, y, z = get_ref_data(iter_dir, track_idcs, iters)
            ax3.tricontour(x, y, z, levels=14, linewidths=0.5, colors='k')
            cntr2 = ax3.tricontourf(x, y, z, levels=14, cmap="RdBu_r")
            fig.colorbar(cntr2, ax=ax3)
        else:
            for i in range(5)[::-1]:
                ival = (i+1)**2
                colorplot = ax3.scatter(ref_x, ref_y, c=fs, alpha=(1/ival), s=50*ival)
            fig.colorbar(colorplot, ax=ax3,format = "{x:.6}", vmin=ref_bounds[-2], vmax=ref_bounds[-1])
    atoms = read(my_join(img_iter_iter_dir, "CONTCAR"), format="vasp")
    plot_atoms(atoms, ax1,  rotation=r1)
    plot_atoms(atoms, ax2,  rotation=r2)
    ax1.set_ylim(0, 10)
    ax2.set_ylim(0, 10)
    ax1.set_xlim(5, 15)
    ax2.set_xlim(5, 15)
    ax4.plot(range(len(ref_x)), ref_x)
    ax5.plot(range(len(ref_x)), fs)
    ax4.plot([idx], ref_x[idx])
    ax3.plot(ref_x, ref_y, color='black')
    ax3.scatter(ref_x[idx], ref_y[idx], label="image " + str(idx))
    if not xlabel is None:
        ax3.set_xlabel(xlabel)
    else:
        ax3.set_xlabel(f"{atoms.get_chemical_symbols()[i1-1]}({str(i1)}) - {atoms.get_chemical_symbols()[i2-1]}({str(i2)}) dist (Angstrom)")
    if _2d:
        ax3.set_ylabel(f"{atoms.get_chemical_symbols()[i3-1]}({str(i3)}) - {atoms.get_chemical_symbols()[i4-1]}({str(i4)}) dist (Angstrom)")
    ax3.ticklabel_format(useOffset=False)
    if show_max:
        max_f = np.max(fs)
        max_idx = fs.index(max_f)
        ax3.scatter(ref_x[max_idx], ref_y[max_idx], label=f"{max_f:.{7}g}", color="red", marker="x")
    if not ref_bounds is None:
        ax3.set_xlim(ref_bounds[0], ref_bounds[1])
        ax3.set_ylim(ref_bounds[2], ref_bounds[3])
        ax4.set_ylim(ref_bounds[0], ref_bounds[1])
        ax4.set_xlim(0, max(max_step, len(fs)))
        ax5.set_ylim(ref_bounds[-2], ref_bounds[-1])
    ax3.legend()
    if not text_str is None:
        fig.text(0.05, 0.05, text_str, fontsize='xx-small')
    if not _2d:
        plt.tight_layout()
    fname = data_dir + "iter_pngs//" + str(iter) + "_" + str(idx) + ".png"
    plt.savefig(fname)
    plt.close()
    return imageio.v2.imread(fname)



def get_ref_data(iter_data_dir, track_idcs, iters_range):
    i1, i2, i3, i4 = map_2d_idcs(track_idcs)
    xvals = []
    yvals = []
    fs = []
    for it in iters_range:
        iter_iter_data_dir = my_join(iter_data_dir, str(it))
        for img in range(it + 1):
            xvals.append(get_dist(img, iter_iter_data_dir, i1=i1, i2=i2))
            yvals.append(get_dist(img, iter_iter_data_dir, i1=i3, i2=i4))
            fs.append(read_f(my_join(iter_iter_data_dir, str(img))))
    return xvals, yvals, fs

def get_ref_bounds(iter_data_dir, track_idcs, iters_range):
    xvals, yvals, fs = get_ref_data(iter_data_dir, track_idcs, iters_range)
    return np.min(xvals), np.max(xvals), np.min(yvals), np.max(yvals), np.min(fs), np.max(fs)

def get_dist_traj(atoms, i1=0, i2=0, shift = -1):
    val = np.linalg.norm(atoms.positions[i1 + shift] - atoms.positions[i2 + shift])
    if np.isclose(val, 0):
        raise ValueError
    return val

def get_en_traj(atoms):
    return atoms.get_potential_energy()

def get_neb_scan_contour_data(neb_opts, track_idcs):
    x, y, z = [], [], []
    for i in range(len(neb_opts)):
        xs, ys, zs = get_traj_ref_data(neb_opts[i], track_idcs)
        x += xs
        y += ys
        z += zs
    return x, y, z

def get_traj_ref_data(opts, track_idcs):
    _i1, _i2, _i3, _i4 = map_2d_idcs(track_idcs)
    x = []
    y = []
    z = []
    for i in range(len(opts)):
        xs, ys, zs = get_traj_opt_ref_data(opts[i], [_i1, _i2, _i3, _i4])
        x += xs
        y += ys
        z += zs
    return x, y, z

def get_traj_opt_ref_data(opts, track_idcs):
    _i1, _i2, _i3, _i4 = map_2d_idcs(track_idcs)
    x = []
    y = []
    z = []
    for atoms in opts:
        try:
            x.append(get_dist_traj(atoms, _i1, _i2))
            y.append(get_dist_traj(atoms, _i3, _i4))
            z.append(get_en_traj(atoms))
        except:
            pass
    return x, y, z

def get_nOpts(traj, nImages):
    nSys = len(traj)
    nOpts = nSys / float(nImages)
    assert np.isclose(nOpts % 1,0.)
    return int(nOpts)

def get_neb_opts(neb_dir):
    rel_dirs = list_integer_dirs(neb_dir, strict=False)
    neb_opts = []
    for dirr in rel_dirs:
        neb_opts.append([])
        neb_traj = my_join(dirr, "neb.traj")
        traj = TrajectoryReader(neb_traj)
        nImgs = len(list_integer_dirs(dirr))
        nOpts = get_nOpts(traj, nImgs)
        for i in range(nOpts):
            neb_opts[-1].append([])
            for j in range(nImgs):
                idx = int(j+(nImgs*i))
                try:
                    neb_opts[-1][-1].append(traj[idx])
                except:
                    neb_opts[-1][-1].append(None)
                    pass
    return neb_opts


def get_opts(traj, nImages):
    nOpts = get_nOpts(traj, nImages)
    opts = []
    for i in range(nOpts):
        opts.append([])
        for j in range(nImages):
            idx = int(j+(nImages*i))
            try:
                opts[-1].append(traj[idx])
            except:
                opts[-1].append(None)
                pass
    return opts

def x_y_labels(atoms, track_idcs):
    i1, i2, i3, i4 = map_2d_idcs(track_idcs)
    xlabel = f"{atoms.get_chemical_symbols()[i1-1]}({str(i1)}) - {atoms.get_chemical_symbols()[i2-1]}({str(i2)}) dist (Angstrom)"
    ylabel = f"{atoms.get_chemical_symbols()[i3-1]}({str(i3)}) - {atoms.get_chemical_symbols()[i4-1]}({str(i4)}) dist (Angstrom)"
    return xlabel, ylabel



def animate_slide(iter_pngs_dir, track_idcs, x, y, z, opts, iOpt, xlabel=None, ylabel=None):
    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((1,1), (0,0), rowspan=1, colspan=1)
    cntr1 = ax1.tricontourf(x, y, z, levels=30, cmap="RdBu_r")
    xs, ys, zs = get_traj_opt_ref_data(opts[iOpt], track_idcs)
    ax1.plot(xs, ys, color="black")
    if not xlabel is None:
        ax1.set_xlabel(xlabel)
    if not ylabel is None:
        ax1.set_ylabel(ylabel)
    fname = my_join(iter_pngs_dir, str(iOpt) + ".png")
    plt.savefig(fname)
    plt.close()
    return imageio.v2.imread(fname)

def animate_neb_traj(neb_traj_dir, track_idcs, duration=100, read_existing=False, xyz=None):
    neb_traj = my_join(neb_traj_dir, "neb.traj")
    traj = TrajectoryReader(neb_traj)
    nImgs = len(list_integer_dirs(neb_traj_dir))
    nOpts = get_nOpts(traj, nImgs)
    print(f"nOPts: {nOpts}")
    assert os.path.exists(neb_traj)
    iter_pngs_dir = my_join(neb_traj_dir, "iter_pngs")
    if not os.path.exists(iter_pngs_dir):
        os.mkdir(iter_pngs_dir)
    if not read_existing:
        atoms = traj[0]
        xlabel, ylabel = x_y_labels(atoms, track_idcs)
        opts = get_opts(traj, nImgs)
        if xyz is None:
            x, y, z = get_traj_ref_data(opts, track_idcs)
            x = x[1:-1]
            y = y[1:-1]
        else:
            x, y, z = xyz[0], xyz[1], xyz[2]
        frames = []
        for iOpt in range(nOpts):
            frames.append(animate_slide(iter_pngs_dir, track_idcs, x, y, z, opts, iOpt, xlabel=xlabel, ylabel=ylabel))
    else:
        frames = []
        for iOpt in range(nOpts):
            fname = my_join(iter_pngs_dir, str(iOpt) + ".png")
            frames.append(imageio.v2.imread(fname))
    imageio.mimsave(my_join(neb_traj_dir,"neb.gif"), frames, duration=duration, loop=0)


def animate_neb_scan_slide(x, y, z, neb_opts, iTraj, iOpt, neb_dir, track_idcs, xlabel=None, ylabel=None, xlims=None, ylims=None, pngs_dir=None):
    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((1,1), (0,0), rowspan=1, colspan=1)
    cntr1 = ax1.tricontourf(x, y, z, levels=30, cmap="RdBu_r")
    xs, ys, zs = get_traj_opt_ref_data(neb_opts[iTraj][iOpt], track_idcs)
    ax1.plot(xs, ys, color="black")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_xlim(xlims[0], xlims[1])
    ax1.set_ylim(ylims[0], ylims[1])
    fname = my_join(pngs_dir, f"{iTraj}_{iOpt}.png")
    plt.savefig(fname)
    plt.close()
    return imageio.v2.imread(fname)


def animate_neb_scan(neb_dir, track_idcs, read_existing = False):
    neb_opts = get_neb_opts(neb_dir)
    xlabel, ylabel = x_y_labels(neb_opts[0][0][0], track_idcs)
    nIters = len(list_integer_dirs(neb_dir, strict=False))
    x, y, z = get_neb_scan_contour_data(neb_opts, track_idcs)
    xlims = [np.min(x), np.max(x)]
    ylims = [np.min(y), np.max(y)]
    pngs_dir = my_join(neb_dir, "iters_pngs")
    if not os.path.exists(pngs_dir):
        os.mkdir(pngs_dir)
    frames = []
    if not read_existing:
        for iTraj in range(nIters):
            x, y, z = get_traj_ref_data(neb_opts[iTraj], track_idcs)
            for iOpt in range(len(neb_opts[iTraj])):
                frames.append(animate_neb_scan_slide(x, y, z, neb_opts, iTraj, iOpt, neb_dir, track_idcs, xlabel=xlabel, ylabel=ylabel, xlims=xlims, ylims=ylims, pngs_dir=pngs_dir))
    return frames