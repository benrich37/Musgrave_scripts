import numpy as np
import numba


# I have to decalre argument data types u dummy
#@numba.jit()
def fmt_num_1(number: np.float64):
    if abs(number) < 1.0:
        num_decimals = 6
    else:
        num_decimals = 6 - int(np.floor(np.log10(abs(number))))
    format_string = f".{num_decimals}f"
    return format(number, format_string)

def get_start_line(outfile):
    start = None
    for i, line in enumerate(open(outfile)):
        if 'JDFTx' in line and '***' in line:
            start = i
    return start
def get_grid_shape(outfile):
    return get_vars(outfile)[0]
def get_vars(outfile):
    start = get_start_line(outfile)
    R = np.zeros((3, 3))
    iLine = 0
    refLine = -10
    initDone = False
    Rdone = False
    Sdone = False
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
    return S, R, mu

def get_coords_vars(outfile):
    start = get_start_line(outfile)
    iLine = 0
    refLine = -10
    R = np.zeros((3, 3))
    Rdone = False
    ionPosStarted = False
    ionNames = []
    ionPos = []
    for i, line in enumerate(open('out')):
        if i > start:
            # Lattice vectors:
            if line.find('Initializing the Grid') >= 0 and (not Rdone):
                refLine = iLine
            rowNum = iLine - (refLine + 2)
            if rowNum >= 0 and rowNum < 3:
                R[rowNum, :] = [float(x) for x in line.split()[1:-1]]
            if rowNum == 3:
                refLine = -10
                Rdone = True
            # Coordinate system and ionic positions:
            if ionPosStarted:
                tokens = line.split()
                if len(tokens) and tokens[0] == 'ion':
                    ionNames.append(tokens[1])
                    ionPos.append([float(tokens[2]), float(tokens[3]), float(tokens[4])])
                else:
                    break
            if line.find('# Ionic positions in') >= 0:
                coords = line.split()[4]
                ionPosStarted = True
            # Line counter:
            iLine += 1
    ionPos = np.array(ionPos)
    if coords != 'lattice':
        ionPos = np.dot(ionPos, np.linalg.inv(R.T))  # convert to lattice
    return ionPos, ionNames, R

def count_ions(ionNames):
    ion_names = []
    ion_counts = []
    for name in ionNames:
        if name not in ion_names:
            ion_names.append(name)
            ion_counts.append(ionNames.count(name))
    return ion_names, ion_counts

#@numba.jit()
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

def write_densities(out_str: str, dtot: np.ndarray):
    iterations = int(np.floor(len(dtot) / 5.))
    spill = len(dtot) % 5
    for i in range(spill):
        list(dtot).append(0.0)
    range5 = [0,1,2,3,4]
    for i in range(iterations):
        for j in range5:
            out_str += '\t'
            out_str += fmt_num_2(dtot[j + 5 * i])
            # out_str += funcs.fmt_num_2(dtot[j + 5*i])
        out_str += '\n'

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
    write_densities(out_str, dtot)
    out_str += ' \n'
    # Write chgcar
    savepath = ''
    if savedir is not None:
        savepath += savedir
    if pc:
        savepath += '//'
    else:
        savepath += '/'
    savepath += chgcar_name
    with open(chgcar_name, 'w') as chgcarfile:
        chgcarfile.write(out_str)