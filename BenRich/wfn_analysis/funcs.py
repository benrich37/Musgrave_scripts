import numpy as np

# If you find me outside the directory wfn_analysis, probably don't use me

def parse_complex_bandprojection(tokens):
    out = []
    for i in range(int(len(tokens)/2)):
        repart = tokens[2*i]
        impart = tokens[(2*i) + 1]
        num = complex(float(repart) + float(impart[1:]))
        out.append(num)
    return out

def parse_bandfile(bandfile):
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
                    proj[iState,iBand]=np.array(parse_complex_bandprojection(tokens))
    return proj, nStates, nBands, nProj, nSpecies, nOrbsPerAtom


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
            if line.split()[-1] == 'spin':
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


def parse_eigfile(eigfile, nStates):
    E = np.fromfile(eigfile)
    return E.reshape(nStates,-1)