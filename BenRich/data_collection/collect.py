import os
import misc_fns
import surf_nrg_comp
import matplotlib.pyplot as plt

def get_ecomp(path):
    ecomp_list = get_ecomp_list(path)
    return get_ecomp_dict(ecomp_list)

def get_ecomp_list(path):
    out = []
    file = open(path)
    for line in file:
        out.append(line.rstrip('\n').replace(' ', '').split('='))
    file.close()
    return out

def get_ecomp_dict(ecomp_list):
    out = {}
    for x in ecomp_list:
        if len(x) == 2:
            out[x[0]] = x[1]
        else:
            pass
    return out

def check(path, hit='Ecomponents'):
    elts = os.listdir(path)
    return hit in elts

def dictify_dir(path):
    dirs = misc_fns.list_dirs(path)
    out = {}
    for d in dirs:
        out[d] = traverse(os.path.join(path, d))
    return out

def traverse(path):
    if check(path):
        return get_ecomp(os.path.join(path, 'Ecomponents'))
    else:
        return dictify_dir(path)

def look_for_ecomp(start_dir):
    data = {}
    os.chdir(start_dir)
    dirs = misc_fns.list_dirs('.')
    for d in dirs:
        data[d] = traverse(d)
    return data

def plot_etype_wrt_V(data_dict, e_type):
    # data dict should only contain dif biases of a specific surface
    biases_labels = data_dict.keys()
    biases = []
    nrgs = []
    for b in biases_labels:
        if b == 'No_bias':
            pass
        else:
            biases.append(float(b.strip('V')))
            data = data_dict[b]
            if '01' in data.keys():
                nrgs.append(float(data_dict[b]['01'][e_type]))
            else:
                nrgs.append(float(data_dict[b][e_type]))
    plt.scatter(biases, nrgs)

# start_dir = '/Users/richb/Desktop/Research/Musgrave/mncs/Aziz Structures'
# data = look_for_ecomp(start_dir)



