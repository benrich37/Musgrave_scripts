import numpy as np
import matplotlib.pyplot as plt
"""
align_yaxis from https://stackoverflow.com/posts/46901839/revisions
"""
def align_yaxis(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = (ax1, ax2)
    extrema = [ax.get_ylim() for ax in axes]
    tops = [extr[1] / (extr[1] - extr[0]) for extr in extrema]
    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
        axes, extrema, tops = [list(reversed(l)) for l in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    b_new_t = extrema[0][0] + tot_span * (extrema[0][1] - extrema[0][0])
    t_new_b = extrema[1][1] - tot_span * (extrema[1][1] - extrema[1][0])
    axes[0].set_ylim(extrema[0][0], b_new_t)
    axes[1].set_ylim(t_new_b, extrema[1][1])


def get_integrated_array(ygrid, dx):
    integrated_array = []
    cur = 0
    if dx is None:
        return None
    else:
        for y in ygrid:
            cur += y*dx
            integrated_array.append(cur)
    return np.array(integrated_array)

def get_integrated_trapezoidal(ygrid, xgrid):
    integrated_array = []
    cur = 0
    for i in range(len(ygrid) - 2):
        dx = (xgrid[i + 2] - xgrid[i])/2.
        integrated_array.append(cur + (ygrid))
    return np.array(integrated_array)


def ez_pCOHP_sum(uorb_list, vorb_list, Egrid, Emin, Emax, dE, pCOHP_func):
    pCOHP_sum = np.zeros(np.shape(Egrid))
    for u in uorb_list:
        for v in vorb_list:
            pCOHP_sum += pCOHP_func(u, v, Egrid, Emin, Emax, dE)
    return pCOHP_sum

def plot_hmat(atom_labels, orbs_dict, hfunc):
    orbs_want = []
    div_idcs = [0]
    partition_lengths = []
    for label in atom_labels:
        orbs_want += orbs_dict[label]
        div_idcs.append(len(orbs_want))
        partition_lengths.append(len(orbs_dict[label]))
    div_idcs = div_idcs[:-1]
    matrix = np.real(hfunc(orbs_want))
    plt.imshow(matrix)
    norbs = len(orbs_want)
    labels = [''] * norbs
    for i, idx in enumerate(div_idcs):
        plt.axvline(x=idx - 0.5, label=atom_labels[i], color='black', linestyle='--')
        plt.axhline(y=idx - 0.5, label=atom_labels[i], color='black', linestyle='--')
        labels[div_idcs[i] + int(np.floor(partition_lengths[i]/2.))] = atom_labels[i]
    plt.xticks(np.arange(-0.5, norbs - 0.5), labels, rotation=-45)
    plt.yticks(np.arange(-0.5, norbs - 0.5), labels)
    plt.colorbar()