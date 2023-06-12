import numpy as np
import jdftxfuncs as jfunc
from numba import jit

def parse_data(bandfile, gvecfile, eigfile, guts=False):
    # TODO: Return array of strings to indicate which orbital is in which orbital index
    # ^ yeah fat chance nerd
    """
    :param bandfile: Path to BandProjections file (str)
    :param gvecfile: Path to Gvectors file (str)
    :param eigfile: Path to eigenvalues file (str)
    :param guts: Whether to data not directly needed by main functions (Boolean)
    :return:
        - proj: a rank 3 numpy array containing the complex band projection,
                data (<φ_μ|ψ_j> = T_μj) with dimensions (nStates, nBands, nProj)
        - nStates: the number of electronic states (integer)
        - nBands: the number of band functions (integer)
        - nProj: the number of band projections (integer)
        - nOrbsPerAtom: a list containing the number of orbitals considered
                        for each atom in the crystal structure (list(int))
        - wk: A list of weight factors for each k-point (list(float))
        - k_points: A list of k-points (given as 3 floats) for each k-point. (list(list(float))
        - E: nStates by nBands array of KS eigenvalues (np.ndarray(float))
        *- iGarr: A list of numpy arrays for the miller indices of each G-vector used in the
                  expansion of each state (list(np.ndarray(int)))
    :rtype: tuple
    """
    proj, nStates, nBands, nProj, nSpecies, nOrbsPerAtom = jfunc.parse_bandfile(bandfile)
    if guts:
        wk, iGarr, k_points, nStates = jfunc.parse_gvecfile(gvecfile)
        E = jfunc.parse_eigfile(eigfile, nStates)
        return proj, nStates, nBands, nProj, nOrbsPerAtom, wk, k_points, E, iGarr
    else:
        wk, k_points, nStates = jfunc.parse_gvecfile_noigarr(gvecfile)
        E = jfunc.parse_eigfile(eigfile, nStates)
        return proj, nStates, nBands, nProj, nOrbsPerAtom, wk, k_points, E

def prepare_small_funcs(proj, E, numba=False):
    """
    :param proj: a rank 3 numpy array containing the complex band projection,
                 data (<φ_μ|ψ_j> = T_μj) with dimensions (nStates, nBands, nProj)
    :param E: nStates by nBands array of KS eigenvalues (np.ndarray(float))
    :return:
        - T_juk:
            - :param uorb: index for orbital u (int)
            - :param jband: index for band j (int)
            - :param kstate: index for k-point k (int)
            - :return: Projection <psi_j(k)|phi_u> (np.complex128)
        - P_uvjk:
            - :param uorb: index for orbital u (int)
            - :param vorb: index for orbital v (int)
            - :param jband: index for band j (int)
            - :param kstate: index for k-point k (int)
            - :return: "Cross"-projection <phi_u|psi_j(k)><psi_j(k)|phi_v> (np.complex128)
        - e_jk:
            - :param jband: index for band j (int)
            - :param kstate: index for k-point k (int)
            - :return: Eigenvalue for KS function |psi_j(k)>
    :rtype: tuple
    """
    if numba:
        @jit(nopython=True)
        def T_juk(uorb, jband, kstate):
            return proj[kstate][jband][uorb]

        @jit(nopython=True)
        def P_uvjk(uorb, vorb, jband, kstate):
            t1 = T_juk(uorb, jband, kstate)
            t2 = T_juk(vorb, jband, kstate)
            t1_conj = t1.real - t1.imag * 1j
            return t1_conj * t2

        @jit(nopython=True)
        def e_jk(jband, kstate):
            return E[kstate][jband]
    else:
        T_juk = lambda uorb, jband, kstate: proj[kstate][jband][uorb]
        P_uvjk = lambda uorb, vorb, jband, kstate: np.conjugate(T_juk(uorb, jband, kstate))*T_juk(vorb, jband, kstate)
        e_jk = lambda jband, kstate: E[kstate][jband]
    return T_juk, P_uvjk, e_jk

def prepare_large_funcs(e_jk,P_uvjk,nBands,nStates,wk,k_points,guts=False,docuprint=False):
    """
    :param e_jk: See return signature for prepare_small_funcs (lambda)
    :param P_uvjk: See return signature for prepare_small_funcs (lambda)
    :param nBands: the number of band functions (integer)
    :param nStates: the number of electronic states (integer)
    :param wk: A list of weight factors for each k-point (list(float))
    :param k_points: A list of k-points (given as 3 floats) for each k-point. (list(list(float))
    :param guts: Whether to data not directly needed by main functions (Boolean)
    :return:
        - pCOHP_uv:
            - :param uorb: index for orbital u (int)
            - :param vorb: index for orbital v (int)
            - :param Emin:
            - :param Emax:
            - :param dE:
            - :return:
                - pCOHP_uv_vals: Evaluated pCOHP_uv values for each E on Egrid (y-values) (np.ndarray(float))
        - pCOHP_uv_u:
            - :param uorb: index for orbital u (int)
            - :param vorb: index for orbital v (int)
            - :param Emin: Lower bound for evaluated energies (float)
            - :param Emax: Upper bound for evaluated energies (float)
            - :param dE: Spacing between evaluated energies (dE)
        - H_atomic_matrix:
            - :param orb_idcs: orbital indices to construct Hamiltonian matrix with (list(int))
            - :param k_points: list of all k-points (only used to enumerate number of kstates) (list(any))
            - :param array_bool: True to change rtype to np.ndarray(np.ndarray(float)),
                                 False for list(np.ndarray(float))
        *- H_uvk:
            - :param uorb: index for orbital u (int)
            - :param vorb: index for orbital v (int)
            - :param kstate: index for k-point k (int)
        *- pCOHP_uvk:
            - :param uorb: index for orbital u (int)
            - :param vorb: index for orbital v (int)
            - :param kstate: index for k-point k (int)
        *- pCOPH_uvk_u:
            - :param uorb: index for orbital u (int)
            - :param vorb: index for orbital v (int)
            - :param kstate: index for k-point k (int)
        *- Hk_atomic_matrix:
            - :param orb_idcs: orbital indices to construct Hamiltonian matrix with (list(int))
            - :param kstate: index for k-point k (int)
            - :param array_bool: True to change rtype to np.ndarray(np.ndarray(float)),
                                 False for list(np.ndarray(float))
        *- H_atomic_matrices:
            - :param orb_idcs: orbital indices to construct Hamiltonian matrix with (list(int))
            - :param k_points: list of all k-points (only used to enumerate number of kstates) (list(any))
            - :param array_bool: True to change rtype to np.ndarray(np.ndarray(float)),
                                 False for list(np.ndarray(float))
    """
    H_uvk = lambda uorb, vorb, kstate: _H_uvk(uorb,vorb,kstate,e_jk,P_uvjk,nBands)
    pCOHP_uvk = lambda uorb, vorb, kstate, Egrid, Emin, Emax, dE: _pCOHP_uvk(uorb,vorb,kstate,Egrid,Emin,Emax,dE,nBands,H_uvk,e_jk,P_uvjk)
    pCOHP_uv = lambda uorb, vorb, Egrid, Emin, Emax, dE: _pCOHP_uv(uorb, vorb, Egrid, Emin, Emax, dE, nStates, wk, pCOHP_uvk)
    if docuprint:
        print(docustrings_printable['pCOHP_uv'])
    pCOHP_uvk_u = lambda uorb, vorb, kstate, Egrid, Emin, Emax, dE: _pCOHP_uvk_u(uorb,vorb,kstate,Egrid,Emin,Emax,dE,nBands,H_uvk,e_jk,P_uvjk)
    pCOHP_uv_u = lambda uorb, vorb, Egrid, Emin, Emax, dE: _pCOHP_uv_u(uorb,vorb,Egrid,Emin,Emax,dE,nStates,wk,pCOHP_uvk_u)
    Hk_atomic_matrix = lambda orb_idcs, kstate, array_bool: _Hk_atomic_matrix(orb_idcs, kstate, H_uvk, array_bool)
    H_atomic_matrices = lambda orb_idcs, k_points, array_bool: _H_atomic_matrices(orb_idcs, k_points, array_bool, Hk_atomic_matrix)
    H_atomic_matrix = lambda orb_idcs: _H_atomic_matrix(orb_idcs, H_atomic_matrices, k_points, wk)
    if guts:
        return pCOHP_uv, pCOHP_uv_u, H_atomic_matrix, H_uvk, pCOHP_uvk, pCOHP_uvk_u, Hk_atomic_matrix, H_atomic_matrices
    else:
        return pCOHP_uv, pCOHP_uv_u, H_atomic_matrix

def adjust_Emax(Emin, Emax, dE):
    delta = Emax - Emin
    counts = np.ceil(delta / dE)
    Emax_new = Emin + ((counts+3)*dE)
    Emin_new = Emin - 3*dE
    return Emax_new

def adjust_Ebounds(Emin, Emax, dE):
    delta = Emax - Emin
    counts = np.ceil(delta / dE)
    Emax_new = Emin + ((counts+3)*dE)
    Emin_new = Emin - 3*dE
    return Emax_new, Emin_new

def orbs_idx_dict(outfile, nOrbsPerAtom):
    ionPos, ionNames, R = jfunc.get_coords_vars(outfile)
    ion_names, ion_counts = jfunc.count_ions(ionNames)
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

def _H_uvk(uorb, vorb, kstate, e_jk, P_uvjk, nBands):
    sum_hold = 0
    for j in range(nBands):
        sum_hold += e_jk(j, kstate) * P_uvjk(uorb, vorb, j, kstate)
    return sum_hold


def _pCOHP_uvk(uorb, vorb, kstate, Egrid, Emin, Emax, dE, nBands, H_uvk, e_jk, P_uvjk):
    Huvk = H_uvk(uorb, vorb, kstate)
    output = np.zeros(np.shape(Egrid))
    for j in range(nBands):
        ejk = e_jk(j, kstate)
        if ((ejk < Emax) and (ejk > Emin)):
            e_idx = int(np.floor((ejk - Emin) / dE))
            e_weight = ejk - Egrid[e_idx]
            e_spill = Egrid[int(e_idx + 1)] - ejk
            x = P_uvjk(uorb, vorb, j, kstate) * Huvk
            output[e_idx] += np.real(x) * e_weight
            output[e_idx + 1] += np.real(x) * e_spill
    return output


def _pCOHP_uv(uorb, vorb, Egrid, Emin, Emax, dE, nStates, wk, pCOHP_uvk):
    pCOHP_uv_vals = np.zeros(np.shape(Egrid))
    for k in range(nStates):
        pCOHP_uv_vals += pCOHP_uvk(uorb, vorb, k, Egrid, Emin, Emax, dE) * wk[k]
    return pCOHP_uv_vals


def _pCOHP_uvk_u(uorb, vorb, kstate, Egrid, Emin, Emax, dE, nBands, H_uvk, e_jk, P_uvjk):
    Huvk = H_uvk(uorb, vorb, kstate)
    output = np.zeros(np.shape(Egrid))
    for j in range(nBands):
        ejk = e_jk(j, kstate)
        if ((ejk < Emax) and (ejk > Emin)):
            e_idx = int(np.floor((ejk - Emin) / dE))
            e_weight = ejk - Egrid[e_idx]
            e_spill = Egrid[int(e_idx + 1)] - ejk
            x = P_uvjk(uorb, uorb, j, kstate) * Huvk
            output[e_idx] += np.real(x) * e_weight
            output[e_idx + 1] += np.real(x) * e_spill
    return output


def _pCOHP_uv_u(uorb, vorb, Egrid, Emin, Emax, dE, nStates, wk, pCOHP_uvk_u):
    output = np.zeros(np.shape(Egrid))
    for k in range(nStates):
        output += pCOHP_uvk_u(uorb, vorb, k, Egrid, Emin, Emax, dE) * wk[k]
    return output


def _Hk_atomic_matrix(orb_idcs, kstate, H_uvk, array_bool):
    dim = len(orb_idcs)
    out = []
    for i in range(dim):
        out.append([])
        for j in range(dim):
            out[-1].append(0)
    for i in range(dim):
        for j in range(dim):
            out[i][j] = H_uvk(orb_idcs[i], orb_idcs[j], kstate)
    if array_bool:
        return np.array(out)
    else:
        return out


def _H_atomic_matrices(orb_idcs, k_points, array_bool, Hk_atomic_matrix):
    select = range(len(k_points))
    out = []
    for kstate in select:
        out.append(Hk_atomic_matrix(orb_idcs, kstate, array_bool))
    return out


def _H_atomic_matrix(orb_idcs, H_atomic_matrices, k_points, wk):
    matrices = H_atomic_matrices(orb_idcs, k_points, True)
    out = np.zeros(np.shape(matrices[0]), dtype=complex)
    for i in range(len(k_points)):
        out += matrices[i] * wk[i]
    return out

docustrings_printable = {
    'pCOHP_uv': 'pCOHP_uv(orb u index, orb v index, Egrid, Emin, Emax, dE) -> pCOHP_uv(E) array',
    'H_atomic_matrix': 'H_atomic_matrix(orb_idcs) -> H Matrix in orbital basis'
}