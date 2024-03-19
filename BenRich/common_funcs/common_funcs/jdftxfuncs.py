import numpy as np
from numba import jit
from ase import Atoms, Atom
from ase.units import Bohr
from os.path import join as opj
import warnings

def get_start_line(outfile):
    start = None
    for i, line in enumerate(open(outfile)):
        if ('JDFTx' in line) and ('***' in line):
            start = i
    if start is None:
        for i, line in enumerate(open(outfile)):
            if ("Input parsed successfully" in line):
                start = i
    return start

def _get_n_elec_line_reader(line):
    splitted = line.split(":")
    for i, el in enumerate(splitted):
        if "nElectrons" in el:
            nelidx = i
    look = splitted[nelidx+1].split(' ')
    for el in look:
        if len(el) > 0:
            return float(el)

def get_n_elec(outfile):
    start = get_start_line(outfile)
    with open(outfile) as f:
        for i, line in enumerate(f):
            if i > start:
                if "FillingsUpdate:" in line:
                    nelec =  _get_n_elec_line_reader(line)
    return nelec


def _get_n_states_cheap_line_reader(line):
    splitted = line.split(":")
    for i, el in enumerate(splitted):
        if "nStates" in el:
            nstidx = i
    return int(splitted[nstidx+1])


def get_n_states_cheap(outfile):
    start = get_start_line(outfile)
    look = False
    for i, line in enumerate(open(outfile)):
        if i >= start:
            if look:
                if "nStates:" in line:
                    return _get_n_states_cheap_line_reader(line)
            else:
                if "Setting up k-points, bands, fillings" in line:
                    look = True


def parse_real_bandfile(bandfile):
    """ Parser function for the 'bandProjections' file produced by JDFTx
    :param bandfile: the path to the bandProjections file to parse
    :type bandfile: str
    :return: a tuple containing six elements:
        - proj: a rank 3 numpy array containing the complex band projection,
                data (<φ_μ|ψ_j> = T_μj) with dimensions (nStates, nBands, nProj)
        - nStates: the number of electronic states (integer)
        - nBands: the number of energy bands (integer)
        - nProj: the number of band projections (integer)
        - nSpecies: the number of atomic species (integer)
        - nOrbsPerAtom: a list containing the number of orbitals considered
                        for each atom in the crystal structure
    :rtype: tuple
    """
    for iLine, line in enumerate(open(bandfile)):
        tokens = line.split()
        if iLine==0:
            nStates = int(tokens[0])
            nBands = int(tokens[2])
            nProj = int(tokens[4])
            nSpecies = int(tokens[6])
            proj = np.zeros((nStates,nBands,nProj), dtype=complex)
            nOrbsPerAtom = []
        elif iLine>=2:
            if iLine<nSpecies+2:
                nAtoms = int(tokens[1])
                nOrbsPerAtom.extend( [int(tokens[2]),] * nAtoms)
            else:
                iState = (iLine-(nSpecies+2)) // (nBands+1)
                iBand = (iLine-(nSpecies+2)) - iState*(nBands+1) - 1
                if iBand>=0 and iState<nStates:
                    proj[iState,iBand]=np.array(parse_real_bandprojection(tokens))
    return proj, nStates, nBands, nProj, nSpecies, nOrbsPerAtom

def get_nOrbsPerAtoms(bandfile):
    with open(bandfile, 'r') as f:
        for iLine, line in enumerate(f):
            tokens = line.split()
            if iLine==0:
                nStates = int(tokens[0])
                nBands = int(tokens[2])
                nProj = int(tokens[4])
                nSpecies = int(tokens[6])
                proj = np.zeros((nStates,nBands,nProj), dtype=complex)
                nOrbsPerAtom = []
            elif iLine>=2:
                if iLine<nSpecies+2:
                    nAtoms = int(tokens[1])
                    nOrbsPerAtom.extend( [int(tokens[2]),] * nAtoms)
                else:
                    return nOrbsPerAtom
    f.close()

def is_complex_bandfile(bandfile):
    hash_lines = 0
    with open(bandfile, 'r') as f:
        for i, line in enumerate(f):
            if "#" in line:
                hash_lines += 1
                if hash_lines == 2:
                    if "|projection|^2" in line:
                        return False
                    else:
                        return True




def parse_complex_bandfile(bandfile):
    """ Parser function for the 'bandProjections' file produced by JDFTx
    :param bandfile: the path to the bandProjections file to parse
    :type bandfile: str
    :return: a tuple containing six elements:
        - proj: a rank 3 numpy array containing the complex band projection,
                data (<φ_μ|ψ_j> = T_μj) with dimensions (nStates, nBands, nProj)
        - nStates: the number of electronic states (integer)
        - nBands: the number of energy bands (integer)
        - nProj: the number of band projections (integer)
        - nSpecies: the number of atomic species (integer)
        - nOrbsPerAtom: a list containing the number of orbitals considered
                        for each atom in the crystal structure
    :rtype: tuple
    """
    is_complex = is_complex_bandfile(bandfile)
    with open(bandfile, 'r') as f:
        for iLine, line in enumerate(f):
            tokens = line.split()
            if iLine==0:
                nStates = int(tokens[0])
                nBands = int(tokens[2])
                nProj = int(tokens[4])
                nSpecies = int(tokens[6])
                if is_complex:
                    proj = np.zeros((nStates,nBands,nProj), dtype=complex)
                    parser = parse_complex_bandprojection
                else:
                    proj = np.zeros((nStates, nBands, nProj), dtype=float)
                    parser = parse_real_bandprojection
                    warnings.warn("Bandprojections file contains |proj|^2, not proj - invalid data for COHP analysis \n (next time add 'band-projection-params yes no' to inputs file)")
                nOrbsPerAtom = []
            elif iLine>=2:
                if iLine<nSpecies+2:
                    nAtoms = int(tokens[1])
                    nOrbsPerAtom.extend( [int(tokens[2]),] * nAtoms)
                else:
                    iState = (iLine-(nSpecies+2)) // (nBands+1)
                    iBand = (iLine-(nSpecies+2)) - iState*(nBands+1) - 1
                    if iBand>=0 and iState<nStates:
                        proj[iState,iBand]=np.array(parser(tokens))
    f.close()
    if not is_complex:
        proj = np.sqrt(proj) # To avoid rewriting seperate pDOS functions for normed bandProjections
    return proj, nStates, nBands, nProj, nSpecies, nOrbsPerAtom

def parse_dos(filename):
    """
    :param filename:
    :return:
        - header: Header for data indexing (list)
        - data: DOS data (np.ndarray)
    :rtype: tuple
    """
    header = None
    data = []
    for line in open(filename):
        if header is None:
            header = line.rstrip('\n').split('\t')
        else:
            data.append(line.rstrip('\n').split('\t'))
    data = np.array(data, dtype=float).T
    return header, data

def parse_kptsfile(kptsfile):
    wk_list = []
    k_points_list = []
    with open(kptsfile, "r") as f:
        for line in f:
            k_points = line.split("[")[1].split("]")[0].strip().split()
            k_points = [float(v) for v in k_points]
            k_points_list.append(k_points)
            wk = float(line.split("]")[1].strip().split()[0])
            wk_list.append(wk)
    nStates = len(wk_list)
    return wk_list, k_points_list, nStates


def parse_gvecfile(gvecfile):
    """ Parser function for the 'Gvectors' file produced by JDFTx
    :param gvecfile: Path to the G-vector file to be parsed.
    :type gvecfile: str
    :return: A tuple containing the following items:
        - wk (list of floats): A list of weight factors for each k-point.
        - iGarr (list of numpy arrays): A list of numpy arrays representing the
          miller indices for each G-vector used in the expansion of each state
        - k_points (list of lists of floats): A list of k-points (given as 3
          floats) for each k-point.
        - nStates (int): The number of states
    """
    wk = []
    iGarr = []
    k_points = []
    for line in open(gvecfile):
        if line.startswith('#'):
            if 'spin' in line:
                wk.append(float(line.split()[-3]))
            else:
                wk.append(float(line.split()[-1]))
            kpts_str = line[line.index('[') + 1:line.index(']')]
            kpts = []
            for token in kpts_str.split(' '):
                sign = -1 if token.startswith('-') else 1
                if len(token): kpts.append(sign * float(token.strip().strip('+-')))
            k_points.append(kpts)
            iGcur = []
        elif len(line)>1:
            iGcur.append([int(tok) for tok in line.split()])
        else:
            iGarr.append(np.array(iGcur))
    nStates = len(wk)
    return wk, iGarr, k_points, nStates

def parse_gvecfile_noigarr(gvecfile):
    """ Parser function for the 'Gvectors' file produced by JDFTx
    :param gvecfile: Path to the G-vector file to be parsed.
    :type gvecfile: str
    :return: A tuple containing the following items:
        - wk (list of floats): A list of weight factors for each k-point.
        - iGarr (list of numpy arrays): A list of numpy arrays representing the
          miller indices for each G-vector used in the expansion of each state
        - k_points (list of lists of floats): A list of k-points (given as 3
          floats) for each k-point.
        - nStates (int): The number of states
    """
    wk = []
    k_points = []
    for line in open(gvecfile):
        if line.startswith('#'):
            if 'spin' in line:
                wk.append(float(line.split()[-3]))
            else:
                wk.append(float(line.split()[-1]))
            kpts_str = line[line.index('[') + 1:line.index(']')]
            kpts = []
            for token in kpts_str.split(' '):
                sign = -1 if token.startswith('-') else 1
                if len(token): kpts.append(sign * float(token.strip().strip('+-')))
            k_points.append(kpts)
    nStates = len(wk)
    return wk, k_points, nStates

# @jit(nopython=True)
def parse_complex_bandprojection(tokens):
    # This code is unjittable, leave it be for now
    """ Should only be called for data generated by modified JDFTx
    :param tokens: Parsed data from bandProjections file
    :return out: data in the normal numpy complex data format (list(complex))
    """
    out = []
    for i in range(int(len(tokens)/2)):
        repart = tokens[2*i]
        impart = tokens[(2*i) + 1]
        num = complex(float(repart), float(impart))
        out.append(num)
    return out

def parse_real_bandprojection(tokens):
    out = []
    for i in range(len(tokens)):
        out.append(float(tokens[i]))
    return out

def get_grid_shape(outfile):
    """ Get S from outfile
    :param outfile: Path to outfile (str)
    :return S: unit cell grid shape (list(int))
    """
    S = get_vars(outfile)[0]
    return S

def get_vars(outfile):
    """ Get S, R, and mu from outfile
    :param outfile:
    :return:
        - S: Grid shape (list(int))
        - R: Lattice vectors (np.ndarray(float))
        - mu: Chemical potential (float)
    :rtype: tuple
    """
    start = get_start_line(outfile)
    R = np.zeros((3, 3))
    iLine = 0
    refLine = -10
    initDone = False
    Rdone = False
    Sdone = False
    mu = None
    for i, line in enumerate(open(outfile)):
        if i > start:
            if line.startswith('Initialization completed'):
                initDone = True
            if initDone and line.find('FillingsUpdate:') >= 0:
                mu = float(line.split()[2])
            if line.find('Initializing the Grid') >= 0:
                refLine = iLine
            if not Rdone:
                rowNum = iLine - (refLine + 2)
                if rowNum >= 0 and rowNum < 3:
                    R[rowNum, :] = [float(x) for x in line.split()[1:-1]]
                if rowNum == 3:
                    Rdone = True
            if (not Sdone) and line.startswith('Chosen fftbox size'):
                S = np.array([int(x) for x in line.split()[-4:-1]])
                Sdone = True
            iLine += 1
    if not mu is None:
        return S, R, mu
    else:
        return S, R

# def get_coords_vars(outfile):
#     """ get ionPos, ionNames, and R from outfile
#     :param outfile: Path to output file (str)
#     :return:
#         - ionPos: ion positions in lattice coordinates (np.ndarray(float))
#         - ionNames: atom names (list(str))
#         - R: lattice vectors (np.ndarray(float))
#     :rtype: tuple
#     """
#     start = get_start_line(outfile)
#     iLine = 0
#     refLine = -10
#     R = np.zeros((3, 3))
#     Rdone = False
#     ionPosStarted = False
#     ionNames = []
#     ionPos = []
#     for i, line in enumerate(open(outfile)):
#         if i > start:
#             # Lattice vectors:
#             if line.find('Initializing the Grid') >= 0 and (not Rdone):
#                 refLine = iLine
#             rowNum = iLine - (refLine + 2)
#             if rowNum >= 0 and rowNum < 3:
#                 R[rowNum, :] = [float(x) for x in line.split()[1:-1]]
#             if rowNum == 3:
#                 refLine = -10
#                 Rdone = True
#             # Coordinate system and ionic positions:
#             if ionPosStarted:
#                 tokens = line.split()
#                 if len(tokens) and tokens[0] == 'ion':
#                     ionNames.append(tokens[1])
#                     ionPos.append([float(tokens[2]), float(tokens[3]), float(tokens[4])])
#                 else:
#                     break
#             if line.find('# Ionic positions in') >= 0:
#                 coords = line.split()[4]
#                 ionPosStarted = True
#             # Line counter:
#             iLine += 1
#     ionPos = np.array(ionPos)
#     if coords != 'lattice':
#         ionPos = np.dot(ionPos, np.linalg.inv(R.T))  # convert to lattice
#     return ionPos, ionNames, R


def get_coords_vars(outfile):
    """ get ionPos, ionNames, and R from outfile
    :param outfile: Path to output file (str)
    :return:
        - ionPos: ion positions in lattice coordinates (np.ndarray(float))
        - ionNames: atom names (list(str))
        - R: lattice vectors (np.ndarray(float))
    :rtype: tuple
    """
    start = get_start_line(outfile)
    iLine = 0
    refLine = -10
    R = np.zeros((3, 3))
    Rdone = False
    ionPosStarted = False
    ionNames = []
    ionPos = []
    for i, line in enumerate(open(outfile)):
        if i > start:
            # Lattice vectors:
            # if line.find('Initializing the Grid') >= 0 and (not Rdone):
            #     refLine = iLine
            if line[:9] == "lattice  " and (not Rdone):
                refLine = iLine
            rowNum = iLine - (refLine + 1)
            if rowNum >= 0 and rowNum < 3:
                vals = line.split()
                vals = vals[0:3]
                R[rowNum, :] = [float(x) for x in vals]
            if rowNum == 3:
                refLine = -10
                Rdone = True
            # Coordinate system and ionic positions:
            if line[:4] == "ion ":
                tokens = line.split()
                ionNames.append(tokens[1])
                ionPos.append([float(tokens[2]), float(tokens[3]), float(tokens[4])])
            if line.find('coords-type ') >= 0:
                coords = line.strip().split()[-1]
                # ionPosStarted = True
            # Line counter:
            iLine += 1
    ionPos = np.array(ionPos)
    if coords != 'lattice':
        ionPos = np.dot(ionPos, np.linalg.inv(R.T))  # convert to lattice
    return ionPos, ionNames, R

def parse_eigfile(eigfile, nStates):
    """ Get eigenvalues and reshape them into a (nStates x nBands) array
    :param eigfile: Path to eigenvalues file (str)
    :param nStates: the number of electronic states (integer)
    :return E: nStates by nBands array of KS eigenvalues (np.ndarray(float))
    """
    E = np.fromfile(eigfile).reshape(nStates, -1)
    return E

def fmt_num_1(number: np.float64):
    if abs(number) < 1.0:
        num_decimals = 6
    else:
        num_decimals = 6 - int(np.floor(np.log10(abs(number))))
    format_string = f".{num_decimals}f"
    return format(number, format_string)

def count_ions(ionNames):
    ion_names = []
    ion_counts = []
    for name in ionNames:
        if name not in ion_names:
            ion_names.append(name)
            ion_counts.append(ionNames.count(name))
    return ion_names, ion_counts

def fmt_num_2(number):
    exponent = 0
    while abs(number) >= 10:
        number /= 10
        exponent += 1
    while abs(number) < 1:
        number *= 10
        exponent -= 1

    sign = 'E+' if exponent >= 0 else 'E'
    int_part = int(number)
    frac_part = int((number - int_part) * 10 ** 10)
    formatted_number = str(int_part) + '.' + str(frac_part).zfill(10)
    formatted_exponent = (sign if exponent >= 0 else '-') + str(abs(exponent)).zfill(2)
    return formatted_number + formatted_exponent

def write_densities(dtot: np.ndarray):
    out_str = ""
    iterations = int(np.floor(len(dtot) / 5.))
    spill = len(dtot) % 5
    for i in range(spill):
        list(dtot).append(0.0)
    range5 = [0,1,2,3,4]
    for i in range(iterations):
        for j in range5:
            num = dtot[j + 5 * i]
            out_str += ' '
            if num > 0:
                out_str += ' '
            out_str += f"{num:.{10}e}"
            # out_str += funcs.fmt_num_2(dtot[j + 5*i])
        out_str += '\n'
    return out_str

def chgcar_out(n_up, n_dn, out_path, chgcar_name='CHGCAR', savedir=None, pc=True, noncollin=False):
    # Get required variables
    # dtot = np.abs(np.fromfile(n_up))
    # dtot += np.abs(np.fromfile(n_dn))
    dtot = np.fromfile(n_up)
    dtot += np.fromfile(n_dn)
    if noncollin:
        dtot = dtot/2.
    S = get_grid_shape(out_path)
    ionPos, ionNames, R = get_coords_vars(out_path)
    # R *= 0.529177 # Convert to Angstroms
    R_inv = R.T
    dtot *= np.dot(R_inv[0], np.cross(R_inv[1], R_inv[2]))
    ion_names, ion_counts = count_ions(ionNames)
    # Make sure the shapes match up
    try:
        dtot = np.reshape(dtot, S)
        dtot = dtot.flatten(order='F')
    except ValueError:
        print('Error: Ben here! the get_grid_shape function got a shape from your out file that was incompatible with the given d_tot file. Double check and make sure they\'re both from the same calculation, or slack me for help')
    # Write lattice information
    out_str = 'fake_chgcar \n'
    out_str += '   1.0 \n'
    for i in range(3):
        out_str += ' '
        for j in range(3):
            out_str += '    '
            out_str += fmt_num_1(R[i][j])
        out_str += ' \n'
    # Write ion information
    for name in ion_names:
        out_str += '   '
        out_str += name
    out_str += ' \n'
    for count in ion_counts:
        out_str += '     '
        out_str += str(count)
    out_str += ' \n'
    out_str += 'Direct \n'
    for pos in ionPos:
        for p in pos:
            out_str += '  '
            out_str += fmt_num_1(p)
        out_str += ' \n'
    out_str += '\n' # Write empty line (?)
    # Write grid shape
    for length in S:
        out_str += '   '
        out_str += str(length)
    out_str += ' \n'
    out_str += write_densities(dtot)
    out_str += ' \n'
    # Write chgcar
    savepath = ''
    if savedir is not None:
        savepath += savedir
    savepath += chgcar_name
    with open(savepath, 'w') as chgcarfile:
        chgcarfile.write(out_str)



def get_raw_dos_data(file_path):
    file = open(file_path)
    raw_data = []
    for line in file:
        raw_data.append(line.rstrip('\n').split('\t'))
    file.close()
    return raw_data

def cleanup_raw_dos_data(file_path):
    raw_data = get_raw_dos_data(file_path)
    out = []
    out.append([])
    for head in raw_data[0]:
        out[-1].append(head.strip('"'))
    data = []
    for line in raw_data[1:]:
        data.append(np.array(line, dtype=float))
    data = np.array(data, dtype=float)
    out.append(data)
    return out

def get_dos_data_dict(file_path):
    clean_data = cleanup_raw_dos_data(file_path)
    data_dict = {}
    for val in clean_data[0]:
        data_dict[val] = []
    ncol = len(clean_data[0])
    for i in range(ncol):
        data_dict[clean_data[0][i]] = clean_data[1][:,i]
    return data_dict


def SR_wannier_out(fname):
    """
    :param fname: Filename + path for wannier outfile
    :return: S, R
    """
    wout = []
    for line in open(fname):
        wout.append(line.rstrip(('\n')))
    istart = None
    for i, line in enumerate(wout):
        if 'Initializing Wannier Function solver' in line:
            istart = i
    Sline = None
    Rlines = []
    lookR = False
    for i, line in enumerate(wout):
        if i >= istart:
            if 'S =' in line:
                Sline = line
            if lookR:
                if len(Rlines) == 3:
                    lookR = False
                else:
                    Rlines.append(line)
            if 'R =' in line:
                lookR = True
    _Rlines = []
    for line in Rlines:
        _Rlines.append(line.lstrip('[').rstrip(']').split(' '))
    bad_els = [[],[],[]]
    for i, line in enumerate(_Rlines):
        for j, el in enumerate(line):
            if len(el) == 0:
                bad_els[i].append(j)
    R = [[],[],[]]
    for i, line in enumerate(_Rlines):
        for j, el in enumerate(line):
            if not j in bad_els[i]:
                R[i].append(float(el))
    Sline = Sline.split('= ')[1].lstrip('[').rstrip(']').split(' ')
    S = []
    for el in Sline:
        if not len(el) == 0:
            S.append(int(el))
    return np.array(S), np.array(R)


def get_matching_lines(outfile, str_match):
    lines = []
    start = get_start_line(outfile)
    started = False
    for i, line in enumerate(open(outfile)):
        if started:
            if str_match in line:
                lines.append(line)
        elif i == start:
            started = True
    return lines

def get_psuedo_lines(outfile):
    return get_matching_lines(outfile, "ion-species")

def orbs_idx_dict(outfile, nOrbsPerAtom):
    ionNames, ionPos, R = get_input_coord_vars_from_outfile(outfile)
    # ionPos, ionNames, R = get_coords_vars(outfile)
    ion_names, ion_counts = count_ions(ionNames)
    orbs_dict = orbs_idx_dict_helper(ion_names, ion_counts, nOrbsPerAtom)
    return orbs_dict

def orbs_idx_dict_helper(ion_names, ion_counts, nOrbsPerAtom):
    orbs_dict_out = {}
    iOrb = 0
    atom = 0
    for i, count in enumerate(ion_counts):
        for atom_num in range(count):
            atom_label = ion_names[i] + ' #' + str(atom_num + 1)
            norbs = nOrbsPerAtom[atom]
            orbs_dict_out[atom_label] = list(range(iOrb, iOrb + norbs))
            iOrb += norbs
            atom += 1
    return orbs_dict_out


def get_kfolding(outfile):
    key = "kpoint-folding "
    with open(outfile, "r") as f:
        for i, line in enumerate(f):
            if key in line:
                val = np.array(line.split(key)[1].strip().split(), dtype=int)
                return val


def get_mu(outfile):
    mu = 0
    lookkey = "FillingsUpdate:  mu:"
    with open(outfile, "r") as f:
        for line in f:
            if lookkey in line:
                mu = float(line.split(lookkey)[1].strip().split()[0])
    return mu

def get_atom_orb_labels_dict(root):
    fname = opj(root, "bandProjections")
    labels_dict = {}
    ref_lists = [
        ["s"],
        ["px", "py", "pz"],
        ["dxy", "dxz", "dyz", "dx2y2", "dz2"],
        ["fx3-3xy2", "fyx2-yz2", "fxz2", "fz3", "fyz2", "fxyz", "f3yx2-y3"]
    ]
    with open(fname, "r") as f:
        for i, line in enumerate(f):
            if i > 1:
                if "#" in line:
                    return labels_dict
                else:
                    lsplit = line.strip().split()
                    sym = lsplit[0]
                    labels_dict[sym] = []
                    lmax = int(lsplit[3])
                    for j in range(lmax+1):
                        refs = ref_lists[j]
                        nShells = int(lsplit[4+j])
                        for k in range(nShells):
                            if nShells > 1:
                                for r in refs:
                                    labels_dict[sym].append(f"{k}{r}")
                            else:
                                labels_dict[sym] += refs


def get_atoms_from_out(outfile):
    atoms_list = get_atoms_list_from_out(outfile)
    return atoms_list[-1]

def get_start_lines(outfname, add_end=False):
    start_lines = []
    for i, line in enumerate(open(outfname)):
        if "JDFTx 1." in line:
            start_lines.append(i)
    if add_end:
        start_lines.append(i)
    return start_lines

def get_atoms_list_from_out(outfile):
    start_lines = get_start_lines(outfile, add_end=True)
    for i in range(len(start_lines) - 1):
        i_start = start_lines[::-1][i+1]
        i_end = start_lines[::-1][i]
        atoms_list = get_atoms_list_from_out_slice(outfile, i_start, i_end)
        if type(atoms_list) is list:
            if len(atoms_list):
                return atoms_list
    erstr = "Failed getting atoms list from out file"
    raise ValueError(erstr)


def get_atoms_list_from_out_reset_vars(nAtoms=100, _def=100):
    R = np.zeros([3, 3])
    posns = []
    names = []
    chargeDir = {}
    active_lattice = False
    lat_row = 0
    active_posns = False
    log_vars = False
    coords = None
    new_posn = False
    active_lowdin = False
    idxMap = {}
    j = 0
    E = 0
    if nAtoms is None:
        nAtoms = _def
    charges = np.zeros(nAtoms, dtype=float)
    forces = []
    active_forces = False
    coords_forces = None
    return R, posns, names, chargeDir, active_posns, active_lowdin, active_lattice, posns, coords, idxMap, j, lat_row, \
        new_posn, log_vars, E, charges, forces, active_forces, coords_forces


def get_input_coord_vars_from_outfile(outfname):
    start_line = get_start_line(outfname)
    names = []
    posns = []
    R = np.zeros([3,3])
    lat_row = 0
    active_lattice = False
    with open(outfname) as f:
        for i, line in enumerate(f):
            if i > start_line:
                tokens = line.split()
                if len(tokens) > 0:
                    if tokens[0] == "ion":
                        names.append(tokens[1])
                        posns.append(np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])]))
                    elif tokens[0] == "lattice":
                        active_lattice = True
                    elif active_lattice:
                        if lat_row < 3:
                            R[lat_row, :] = [float(x) for x in tokens[:3]]
                            lat_row += 1
                        else:
                            active_lattice = False
                    elif "Initializing the Grid" in line:
                        break
    if not len(names) > 0:
        raise ValueError("No ion names found")
    if len(names) != len(posns):
        raise ValueError("Unequal ion positions/names found")
    if np.sum(R) == 0:
        raise ValueError("No lattice matrix found")
    return names, np.array(posns), R

def get_inputs_atoms_from_outfile(outfname):
    names, posns, R = get_input_coord_vars_from_outfile(outfname)
    atoms = get_atoms_from_outfile_data(names, posns, R)
    return atoms


def get_atoms_from_outfile_data(names, posns, R, charges=None, E=0, momenta=None):
    atoms = Atoms()
    posns *= Bohr
    R = R.T*Bohr
    atoms.cell = R
    if charges is None:
        charges = np.zeros(len(names))
    if momenta is None:
        momenta = np.zeros([len(names), 3])
    for i in range(len(names)):
        atoms.append(Atom(names[i], posns[i], charge=charges[i], momentum=momenta[i]))
    atoms.E = E
    return atoms


def get_atoms_list_from_out_slice(outfile, i_start, i_end):
    charge_key = "oxidation-state"
    opts = []
    nAtoms = None
    R, posns, names, chargeDir, active_posns, active_lowdin, active_lattice, posns, coords, idxMap, j, lat_row, \
        new_posn, log_vars, E, charges, forces, active_forces, coords_forces = get_atoms_list_from_out_reset_vars()
    for i, line in enumerate(open(outfile)):
        if i > i_start and i < i_end:
            if new_posn:
                if "Lowdin population analysis " in line:
                    active_lowdin = True
                elif "R =" in line:
                    active_lattice = True
                elif "# Forces in" in line:
                    active_forces = True
                    coords_forces = line.split()[3]
                elif line.find('# Ionic positions in') >= 0:
                    coords = line.split()[4]
                    active_posns = True
                elif active_lattice:
                    if lat_row < 3:
                        R[lat_row, :] = [float(x) for x in line.split()[1:-1]]
                        lat_row += 1
                    else:
                        active_lattice = False
                        lat_row = 0
                elif active_posns:
                    tokens = line.split()
                    if len(tokens) and tokens[0] == 'ion':
                        names.append(tokens[1])
                        posns.append(np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])]))
                        if tokens[1] not in idxMap:
                            idxMap[tokens[1]] = []
                        idxMap[tokens[1]].append(j)
                        j += 1
                    else:
                        posns = np.array(posns)
                        active_posns = False
                        nAtoms = len(names)
                        if len(charges) < nAtoms:
                            charges = np.zeros(nAtoms)
                ##########
                elif active_forces:
                    tokens = line.split()
                    if len(tokens) and tokens[0] == 'force':
                        forces.append(np.array([float(tokens[2]), float(tokens[3]), float(tokens[4])]))
                    else:
                        forces = np.array(forces)
                        active_forces = False
                ##########
                elif "Minimize: Iter:" in line:
                    if "F: " in line:
                        E = float(line[line.index("F: "):].split(' ')[1])
                    elif "G: " in line:
                        E = float(line[line.index("G: "):].split(' ')[1])
                elif active_lowdin:
                    if charge_key in line:
                        look = line.rstrip('\n')[line.index(charge_key):].split(' ')
                        symbol = str(look[1])
                        line_charges = [float(val) for val in look[2:]]
                        chargeDir[symbol] = line_charges
                        for atom in list(chargeDir.keys()):
                            for k, idx in enumerate(idxMap[atom]):
                                charges[idx] += chargeDir[atom][k]
                    elif "#" not in line:
                        active_lowdin = False
                        log_vars = True
                elif log_vars:
                    if np.sum(R) == 0.0:
                        R = get_input_coord_vars_from_outfile(outfile)[2]
                    if coords != 'cartesian':
                        posns = np.dot(posns, R)
                    if len(forces) == 0:
                        forces = np.zeros([nAtoms, 3])
                    if coords_forces.lower() != 'cartesian':
                        forces = np.dot(forces, R)
                    opts.append(get_atoms_from_outfile_data(names, posns, R, charges=charges, E=E, momenta=forces))
                    R, posns, names, chargeDir, active_posns, active_lowdin, active_lattice, posns, coords, idxMap, j, lat_row, \
                        new_posn, log_vars, E, charges, forces, active_forces, coords_forces = get_atoms_list_from_out_reset_vars(
                        nAtoms=nAtoms)
            elif "Computing DFT-D3 correction:" in line:
                new_posn = True
    return opts