import numpy as np
import os
import matplotlib.pyplot as plt
import math

def get_raw_data(file_path):
    file = open(file_path)
    raw_data = []
    for line in file:
        raw_data.append(line.split('\t'))
    file.close()
    return raw_data

def get_data_dict(raw_data):
    data_dict = {}
    header = raw_data[0]
    for val in header:
        data_dict[val] = []
    for i in range(len(header)):
        for j in range(np.shape(raw_data)[0] - 1):
            data_dict[header[i]].append(raw_data[j + 1][i])
    return data_dict

def transpose_raw_data(data):
    header = data[0]
    rest = data[1:]
    dim = np.shape(rest)
    ret_lists = []
    for i in range(dim[1]):
        ret_lists.append([])
    for i in range(dim[0]):
        for j in range(dim[1]):
            ret_lists[j].append(rest[i][j])
    return header, ret_lists

def collect_runs_data(file_paths):
    raw_data = []
    for f in file_paths:
        raw_data.append(get_raw_data(f))
    data = []
    headers = []
    for d in raw_data:
        tr_data = transpose_raw_data(d)
        data.append(tr_data[1])
        headers.append(tr_data[0])
    return headers, data

def plot_all_dos(data, y_idx, bounds = [0, -1]):
    runs = len(data)
    fig, ax = plt.subplots(runs, 1, sharex='col')
    for i in range(runs):
        ax[i].plot(np.array(data[i][0][bounds[0]:bounds[-1]],
                            dtype=float),
                   np.array(data[i][y_idx][bounds[0]:bounds[-1]],
                            dtype=float)
                   )

def plot_all_dos_many(data, y_idcs, header, bounds = [0, -1], xlims = None):
    runs = len(data)
    plots = len(y_idcs)
    fig, ax = plt.subplots(runs, 1, sharex='col')
    for i in range(runs):
        for j in range(plots):
            ax[i].plot(np.array(data[i][0][bounds[0]:bounds[-1]], dtype=float),
                       np.array(data[i][y_idcs[j]][bounds[0]:bounds[-1]], dtype=float),
                       label = header[y_idcs[j]])
            ax[i].legend()
    if xlims is not None:
        ax[0].set_xlim(xlims[0], xlims[1])
    fig.savefig('C:\\Users\\User\\PycharmProjects\\ASE_Env\\figs\\test_605pm.png')


def check_orbital_and_atom(label, orbs, atom_ids):
    parse = label.split(' ')
    if 'total' in orbs[0].lower():
        yes = False
        for word in parse:
            yes = yes or 'total' in word.lower()
        return yes
    else:
        if 'orbital' in parse:
            orbital = parse[0]
            if orbital in orbs:
                atom_id = int(parse[-1].strip('#'))
                return atom_id in atom_ids
            else:
                return False
        else:
            return False



def bool_on_list(check_list, lambda_bool):
    ret_list = []
    for i in range(len(check_list)):
        if lambda_bool(check_list[i]):
            ret_list.append(i)
    return ret_list

def sum_same_ax_sets(y_sets):
    ys = np.zeros(np.shape(y_sets[0]))
    for y in y_sets:
        ys += np.array(y, dtype=float)
    return ys

def sum_ind_ax_sets(sets):
    """
    :param sets: A list of tuples, where each tuple has [0]-x vals, [1]-y vals
    :return:
    """
    num_sets = len(sets)
    x_min = 1E1000
    x_max = -1E1000
    dx = 1E1000
    for i in range(num_sets):
        x_min = min(x_min, min(sets[i][0]))
        x_max = max(x_max, max(sets[i][0]))
        dx = min(dx, abs(sets[i][0][1] - sets[i][0][0]))
    xs = np.arange(x_min, x_max, step = dx)
    ys = list(np.zeros(np.shape(xs)))
    for n in range(num_sets):
        for i in range(len(sets[n][0])):
            idx = int((sets[n][0][i] - x_min)/dx)
            ys[idx] += sets[n][1][i]
    return xs, ys

def plot_criteria_sets_2(data, headers, criteria_sets, save_dir):
    """
    :param data:
    :param headers:
    :param criteria_sets: Will use shape to dictate how many figures to create, along with how many data sets to plot
    on each graph
            criteria_sets[i] = [fname, [specs], xlims]

            specs = [[crit_00, crit_01, ..., crit_0n, graph_title],
                     [crit_10, crit_11, ..., crit_1n, graph_title],
                     ...
                     [crit_m0, crit_m1, ..., crit_mn, graph_title]]
            crit_ij = [legend_name, run_idx, orbitals, atom_ids]
    :param save_dir:
    :param titles:
    :return:
    """
    plot_data = []
    num_criteria = len(criteria_sets)
    for i in range(num_criteria):
        fname = criteria_sets[i][0]
        specs_i = criteria_sets[i][1]
        plots_i = len(specs_i)
        xlims_i = criteria_sets[i][2]
        fig, ax = plt.subplots(plots_i, 1, sharex='col')
        plot_data.append([fname])
        plot_data[-1].append(list(np.zeros(plots_i)))
        for m in range(plots_i):
            # plot_data[image index][0] -> image fname
            # plot_data[image index][1][m][n] -> x/y lists for data set n plotted on graph m for specific figure
            data_i_m_count = len(specs_i[m]) - 1
            plot_data[-1][1][m] = list(np.zeros(data_i_m_count))
            for n in range(data_i_m_count):
                crit_m_n = specs_i[m][n]
                label_m_n = crit_m_n[0]
                run_idx_m_n = crit_m_n[1]
                orbitals_m_n = crit_m_n[2]
                atom_ids_m_n = crit_m_n[3]
                xs = np.array(data[run_idx_m_n][0], dtype=float)
                ys = get_ys_easy(data[run_idx_m_n], headers[run_idx_m_n], orbitals_m_n, atom_ids_m_n)
                ys_sum = integrate_dos(xs, ys)
                ax[m].plot(xs, ys/ys_sum, label = label_m_n)
                plot_data[-1][1][m][n] = [xs, ys]
            graph_title = specs_i[m][-1]
            ax[m].set_title(graph_title)
            ax[m].legend()
        #fig.legend()
        if type(xlims_i) == list:
            ax[0].set_xlim(xlims_i[0], xlims_i[1])
        plt.tight_layout()
        fig.savefig(save_dir + '\\' + fname + '.png')
    return plot_data


def empty_2d_list(shape):
    ret_list = list(np.zeros(shape))
    for i in range(len(ret_list)):
        ret_list[i] = list(ret_list[i])
    return ret_list


def get_ys_easy(data, header, orbitals, atom_ids):
    """
    :param data: Should only contain one run
    :param header:
    :param orbitals:
    :param atom_ids:
    :return:
    """
    use_y_idcs = bool_on_list(header,
                              lambda label: check_orbital_and_atom(label,
                                                                   orbitals,
                                                                   atom_ids))
    y_sets = []
    for idx in use_y_idcs:
        y_sets.append(data[idx])
    return np.array(sum_same_ax_sets(y_sets), dtype=float)

def plot_criteria_sets(data, headers, criteria_sets, pwd, titles = None):
    """
    :param data:
    :param criteria_sets: [[Criteria name, orbitals, atom_ids, xlims]]
    :return:
    """
    num_criteria = len(criteria_sets)
    num_runs = len(data)
    ret_list = []
    for i in range(num_criteria):
        ret_list.append([])
        for j in range(num_runs):
            ret_list[-1].append([])
    for i in range(num_criteria):
        crit_name = criteria_sets[i][0]
        fig, ax = plt.subplots(num_runs, 1, sharex='col')
        for j in range(num_runs):
            xs = np.array(data[j][0], dtype=float)

            use_y_idcs = bool_on_list(headers[j],
                                      lambda label: check_orbital_and_atom(label,
                                                                           criteria_sets[i][1],
                                                                           criteria_sets[i][2]))
            y_sets = []
            for idx in use_y_idcs:
                y_sets.append(data[j][idx])
            ys = np.array(sum_same_ax_sets(y_sets), dtype=float)
            ax[j].plot(xs, ys)
            ret_list[i][j].append(xs)
            ret_list[i][j].append(ys)
        xlims = criteria_sets[i][-1]
        if len(xlims) == 2:
            ax[0].set_xlim(xlims[0], xlims[1])
        if not titles is None:
            for i in range(len(titles)):
                ax[i].set_title(titles[i])
        fig.savefig(pwd + '\\' + crit_name + '.png')
    return ret_list

def remove_duplicate_pts(xs, ys):
    dup_idcs = []
    for i in range(len(xs) - 1):
        if math.isclose(xs[i+1], xs[i]):
            dup_idcs.append(i+1)
    new_xs = []
    new_ys = []
    for i in range(len(xs)):
        if not i in dup_idcs:
            new_xs.append(xs[i])
            new_ys.append(ys[i])
    return new_xs, new_ys


def integrate_dos(xs, ys):
    xs, ys = remove_duplicate_pts(xs, ys)
    segments = dos_data_find_segments(xs)
    sum_hold = 0
    for seg in segments:
        sum_hold += integrate_dos_segment(xs, ys, seg)
    return sum_hold


def integrate_dos_segment(xs, ys, seg_bounds):
    xs = xs[seg_bounds[0]:seg_bounds[1]]
    ys = ys[seg_bounds[0]:seg_bounds[1]]
    dx = xs[1] - xs[0]
    sum_hold = 0
    for y in ys:
        sum_hold += y
    sum_hold = sum_hold * dx
    return sum_hold


def dos_data_find_segments(xs):
    points = len(xs)
    dxs = []
    break_idcs = []
    for i in range(points - 1):
        dxs.append(xs[i+1] - xs[i])
        # dxs[i] gives gap between xs[i] and [i+1]
    for i in range(len(dxs) - 1):
        if dxs[i + 1] > dxs[i] * 100:
            break_idcs.append(i)
            # index i in break_idcs means xs[i+1] is the end of a segment
    segments = []
    cur_xi = 0
    for i in range(len(break_idcs)):
        segments.append([cur_xi, break_idcs[i] + 1])
        cur_xi = break_idcs[i] + 1
    return segments


def get_data(paths):
    raw = []
    for p in paths:
        raw.append(get_raw_data(p))
    data = []
    headers = []
    for d in raw:
        tr_data = transpose_raw_data(d)
        data.append(tr_data[1])
        headers.append(tr_data[0])
    return data, headers