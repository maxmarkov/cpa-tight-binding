
"""
Hamiltonian construction for diamond-structure sp3s* tight-binding.

Basis (spinless per atom): [s, px, py, pz, s*]
Cell has two atoms A,B -> 10x10 Hamiltonian.

Optional SOC:
If params.delta_so > 0, we build a spinful 20x20 Hamiltonian by Kronecker
doubling and adding an on-site L·S term on the p-manifold.
"""
from __future__ import annotations

from ..utils import backend
from .params import TBParams, onsite_matrix
from .lattice import diamond_nn_vectors, reciprocal_vectors, frac_to_cart_k
from .slater_koster import hopping_sp3s_star

def p_soc_matrix(delta_so: float):
    """
    On-site spin-orbit coupling matrix for p orbitals in basis:
        [px↑, py↑, pz↑, px↓, py↓, pz↓]
    Using H_soc = λ L·S with λ = 2Δ/3 for l=1 so that split-off is Δ.
    """
    if delta_so <= 0:
        return backend.zeros((6, 6), dtype=complex)
    lam = 2.0*delta_so/3.0
    # L matrices in px,py,pz basis
    Lx = backend.array([[0,0,0],[0,0,-1j],[0,1j,0]], dtype=complex)
    Ly = backend.array([[0,0,1j],[0,0,0],[-1j,0,0]], dtype=complex)
    Lz = backend.array([[0,-1j,0],[1j,0,0],[0,0,0]], dtype=complex)
    # Spin-1/2 matrices
    Sx = 0.5*backend.array([[0,1],[1,0]], dtype=complex)
    Sy = 0.5*backend.array([[0,-1j],[1j,0]], dtype=complex)
    Sz = 0.5*backend.array([[1,0],[0,-1]], dtype=complex)
    # Kronecker sum
    H = lam*(backend.kron(Lx,Sx) + backend.kron(Ly,Sy) + backend.kron(Lz,Sz))
    # Our basis ordering is (p orbitals) ⊗ (spin) but we want [p↑, p↓]
    # kron as defined gives [px,py,pz]⊗[↑,↓] -> [px↑,px↓,py↑,py↓,pz↑,pz↓]
    # We need to permute to [px↑,py↑,pz↑,px↓,py↓,pz↓]
    perm = [0,2,4,1,3,5]
    P = backend.eye(6, dtype=complex)[perm]
    return backend.dot(P, backend.dot(H, backend.H(P)))

def bloch_hamiltonian_sp3s_star(k_frac, params: TBParams,
                               include_soc: bool=False):
    """
    Build H(k) for the 2-atom primitive cell.
    k_frac is fractional coordinates in reciprocal primitive basis.
    """
    a = params.a
    bvec = reciprocal_vectors(a)
    kf = backend.asarray(k_frac)
    nd = kf.ndim if hasattr(kf, "ndim") else kf.dim()
    if nd == 1:
        kf = kf.reshape(1, -1)
    k_cart = frac_to_cart_k(kf, bvec)[0]

    # On-site blocks (5x5) per atom
    H0 = backend.asarray(onsite_matrix(params, with_sstar=True), dtype=complex)
    # H_AB(k)
    nn = diamond_nn_vectors(a)
    HAB = backend.zeros((5, 5), dtype=complex)
    for d in nn:
        phase = backend.exp(backend.asarray(1j * float(backend.dot(k_cart, d)), dtype=complex))
        T = hopping_sp3s_star(params, d)
        HAB = HAB + phase * T

    # Assemble 10x10
    H = backend.zeros((10, 10), dtype=complex)
    H[0:5,0:5] = H0
    H[5:10,5:10] = H0
    H[0:5,5:10] = HAB
    H[5:10,0:5] = backend.H(HAB)

    if include_soc and params.delta_so > 0:
        # Double basis with spin
        Hspin = backend.kron(backend.eye(2, dtype=complex), H)
        Hsoc_p = p_soc_matrix(params.delta_so)
        idx_a = [1, 2, 3, 11, 12, 13]
        idx_b = [6, 7, 8, 16, 17, 18]
        soc_full = backend.zeros((20, 20), dtype=complex)
        for idx in (idx_a, idx_b):
            for i in range(6):
                for j in range(6):
                    soc_full[idx[i], idx[j]] = Hsoc_p[i, j]
        Hspin = Hspin + soc_full
        return Hspin

    return H

def hopping_only_matrix(k_frac, params: TBParams, include_soc: bool=False):
    """
    Return hopping-only part H_hop(k) (same size as H(k)), i.e. with on-site terms = 0.
    Useful for CPA where on-site self-energy is treated separately.
    """
    H = bloch_hamiltonian_sp3s_star(k_frac, params, include_soc=include_soc)
    H0 = backend.asarray(onsite_matrix(params, True), dtype=complex)
    if H.shape == (10,10):
        H[0:5,0:5] = H[0:5,0:5] - H0
        H[5:10,5:10] = H[5:10,5:10] - H0
    else:
        Hcell0 = backend.zeros((10,10), dtype=complex)
        Hcell0[0:5,0:5] = H0
        Hcell0[5:10,5:10] = H0
        H = H - backend.kron(backend.eye(2, dtype=complex), Hcell0)
    return H

def velocity_matrix_sp3s_star(k_frac, params: TBParams, alpha: int,
                              include_soc: bool = False):
    """
    Velocity operator v_alpha(k) = (1/hbar) dH/dk_alpha for Cartesian direction alpha.

    Since H_AB(k) = sum_d T_d exp(i k.d), the derivative is:
        dH_AB/dk_alpha = sum_d (i * d_alpha) * T_d * exp(i k.d)

    On-site blocks are k-independent, so their derivative is zero.

    Parameters
    ----------
    k_frac : array (3,)
        Fractional k-point.
    params : TBParams
    alpha : int
        Cartesian direction (0=x, 1=y, 2=z).
    include_soc : bool
        If True and params.delta_so > 0, return 20x20 spinful matrix.

    Returns
    -------
    V : array (norb_cell, norb_cell)
        Velocity matrix in units of eV*Angstrom (hbar=1).
    """
    a = params.a
    bvec = reciprocal_vectors(a)
    kf = backend.asarray(k_frac)
    nd = kf.ndim if hasattr(kf, "ndim") else kf.dim()
    if nd == 1:
        kf = kf.reshape(1, -1)
    k_cart = frac_to_cart_k(kf, bvec)[0]

    nn = diamond_nn_vectors(a)
    dHAB = backend.zeros((5, 5), dtype=complex)
    for d in nn:
        phase = backend.exp(backend.asarray(1j * float(backend.dot(k_cart, d)), dtype=complex))
        T = hopping_sp3s_star(params, d)
        dHAB = dHAB + (1j * float(d[alpha])) * phase * T

    V = backend.zeros((10, 10), dtype=complex)
    V[0:5, 5:10] = dHAB
    V[5:10, 0:5] = backend.H(dHAB)

    if include_soc and params.delta_so > 0:
        V = backend.kron(backend.eye(2, dtype=complex), V)

    return V
