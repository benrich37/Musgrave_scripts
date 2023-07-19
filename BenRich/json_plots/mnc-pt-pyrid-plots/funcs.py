import json
import numpy as np
import matplotlib.pyplot as plt

def ads_energy_nb(data, mol, site):
    F_f = float(data['adsorbed'][mol]['No_bias'][site]['Ecomponents']['F'])
    F_e = float(data['desorbed'][mol]['No_bias']['Ecomponents']['F'])
    return F_f - F_e

def plot_adsorption_prep_data(surf_name, label_strs, data_path):
    data = json.load(open(data_path))
    data = data[surf_name]
    labels = {}
    mols = list(data['adsorbed'].keys())
    for i in range(len(label_strs)):
        if i < 9:
            labels['0' + str(i+1)] = label_strs[i]
            labels[label_strs[i]] = '0' + str(i+1)
        else:
            labels[str(i+1)] = label_strs[i]
            labels[label_strs[i]] = str(i+1)
    data_plot = {}
    for mol in mols:
        data_plot[mol] = []
        for label in label_strs:
            data_plot[mol].append(ads_energy_nb(data, mol, labels[label]))
    return data_plot, mols

def plot_adsorption(surf_name, label_strs, data_path, savedir):
    data_plot, mols = plot_adsorption_prep_data(surf_name, label_strs, data_path)
    bars = len(mols)
    bar_width = 0.1
    indices = np.arange(len(label_strs))
    posns = []
    for i in range(bars):
        posns.append(indices + (bars/2.)*bar_width*(i - (bars/2.)))
    for i in range(bars):
        plt.bar(posns[i], data_plot[mols[i]], width=bar_width, label=mols[i])
    plt.xticks(indices, label_strs)
    plt.xlabel('Binding site')
    plt.ylabel(r'$\Delta F_{ads}$')
    plt.title('Adsorption by mol for surface ' + surf_name)
    plt.axhline(y=0, color='black', linewidth=1)
    plt.legend()
    plt.savefig(surf_name + '.png')
    #plt.savefig(savedir + surf_name + '.png')