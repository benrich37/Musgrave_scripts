import helper_fns as hlpr
import numpy as np
import os
import matplotlib.pyplot as plt
import math

def get_data(paths):
    raw = []
    for p in paths:
        raw.append(hlpr.get_raw_data(p))
    data = []
    headers = []
    for d in raw:
        tr_data = hlpr.transpose_raw_data(d)
        data.append(tr_data[1])
        headers.append(tr_data[0])
    return data, headers


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
                ys = hlpr.get_ys_easy(data[run_idx_m_n], headers[run_idx_m_n], orbitals_m_n, atom_ids_m_n)
                ys_sum = hlpr.integrate_dos(xs, ys)
                ax[m].plot(xs, ys/ys_sum, label = label_m_n)
                plot_data[-1][1][m][n] = [xs, ys]
            graph_title = specs_i[m][-1]
            ax[m].set_title(graph_title)
            ax[m].legend()
        if type(xlims_i) == list:
            ax[0].set_xlim(xlims_i[0], xlims_i[1])
        plt.tight_layout()
        fig.savefig(save_dir + '\\' + fname + '.png')
    return plot_data

