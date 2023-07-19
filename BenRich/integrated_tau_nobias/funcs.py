import numpy as np

def get_start_idx(outfile):
    last_start_line = 0
    for i, line in enumerate(open(outfile)):
        if 'JDFTx' in line and '****' in line:
            last_start_line = i
    return last_start_line


def get_output_shape(outfile):
    start_idx = get_start_idx(outfile)
    R = np.zeros((3, 3))
    iLine = 0
    refLine = -20
    initDone = False
    Rdone = False
    Sdone = False
    for i, line in enumerate(open(outfile)):
        if i > start_idx:
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
    return R, S, mu


def box_xyz_coords(idx_tuple, a_tuple, da_tuple):
    ratios = []
    for i in range(3):
        ratios.append(np.linalg.norm(a_tuple[i]) / da_tuple[i])
    xyz = np.zeros(np.shape(a_tuple[0]))
    for i in range(np.shape(a_tuple[0])[0]):
        xyz += idx_tuple[i] * a_tuple[i] * ratios[i]
    return xyz


def slice_2d_cart_xys(irange, jrange, R, S):
    a_tuple, d_tuple = init_box_xyz_coords(R, S)
    xs = []
    ys = []
    for i in irange:
        for j in jrange:
            xyz = box_xyz_coords((i, j, 0), a_tuple, d_tuple)
            xs.append(xyz[0])
            ys.append(xyz[1])
    return xs, ys


def conv_to_cartesian(slice_2d, R, S, boundsi=None, boundsj=None):
    if boundsi is None:
        boundsi = (0, np.shape(slice_2d)[0])
    if boundsj is None:
        boundsj = (0, np.shape(slice_2d)[1])
    xs, ys = slice_2d_cart_xys(range(boundsi[0], boundsi[1]), range(boundsj[0], boundsj[1]), R, S)
    zs = np.ravel(slice_2d)
    return xs, ys, zs


def init_box_xyz_coords(R, S):
    a1 = R[:, 0]
    a2 = R[:, 1]
    a3 = R[:, 2]
    da1 = np.linalg.norm(a1) / S[0]
    da2 = np.linalg.norm(a2) / S[1]
    da3 = np.linalg.norm(a3) / S[2]
    return (a1, a2, a3), (da1, da2, da3)

def get_ion_posns(outfile):
    start_idx = get_start_idx(outfile)
    #Extract geometry from totalE.out:
    iLine = 0
    refLine = -10
    R = np.zeros((3,3))
    Rdone = False
    ionPosStarted = False
    ionNames = []
    ionPos = []
    for i, line in enumerate(open(outfile)):
        if i > start_idx:
            if line.find('Initializing the Grid') >= 0 and (not Rdone):
                refLine = iLine
            rowNum = iLine - (refLine+2)
            if rowNum>=0 and rowNum<3:
                R[rowNum,:] = [ float(x) for x in line.split()[1:-1] ]
            if rowNum==3:
                refLine = -10
                Rdone = True
    #Coordinate system and ionic positions:
            if ionPosStarted:
                tokens = line.split()
                if len(tokens) and tokens[0] == 'ion':
                    ionNames.append(tokens[1])
                    ionPos.append([float(tokens[2]),float(tokens[3]),float(tokens[4])])
                else:
                    break
            if line.find('# Ionic positions in') >= 0:
                coords = line.split()[4]
                ionPosStarted = True
    #Line counter:
            iLine += 1
    ionPos = np.array(ionPos)
    if coords != 'lattice':
        ionPos = np.dot(ionPos, np.linalg.inv(R.T)) #convert to lattice
    return ionPos, ionNames

def get_plane_idx(outfile, ion_id):
    ionPos, ionId = get_ion_posns(outfile)
    plane_pos = ionPos[ionId.index(ion_id)]
    R, S, mu = get_output_shape(outfile)
    return int(S[2] * plane_pos[2])
