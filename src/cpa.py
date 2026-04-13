
"""
Single-site CPA solver for diagonal (on-site) binary disorder in a multiband tight-binding model.

We treat the alloy as:
  H(k) = H_hop(k) + V_site
where V_site is diagonal (onsite energies). Disorder means V_site is either V_A or V_B
with probabilities (1-x) and x.

CPA replaces disorder by a coherent, energy-dependent self-energy Σ(E) added to all sites.
We solve for Σ(E) such that the average impurity Green's function equals the medium local Green's function.

This module implements a practical fixed-point iteration at each energy.
"""
from __future__ import annotations

from . import backend

def embed_onsite_in_cell(V_atom, n_atoms: int = 2):
    """Embed a per-atom onsite matrix into the 2-atom cell (spinless 10x10)."""
    n = V_atom.shape[0]
    V = backend.zeros((n_atoms * n, n_atoms * n), dtype=complex)
    for a in range(n_atoms):
        o = a * n
        V[o:o+n, o:o+n] = V_atom
    return V

def cpa_solve_energy(
    z: complex,
    Hhop_k,
    V_A_atom,
    V_B_atom,
    x: float,
    sigma_init_atom=None,
    mix: float = 0.6,
    tol: float = 1e-7,
    max_iter: int = 200,
):
    """
    Solve CPA at single complex energy z.
    Inputs:
      Hhop_k: array (Nk, Norb, Norb) of hopping-only Hamiltonians for each k.
      V_A_atom, V_B_atom: (norb_atom,norb_atom) onsite matrices for species A,B (spinless per atom).
      x: concentration of B.
    Returns:
      sigma_atom(z): coherent self-energy per atom (same size as V_A_atom)
      Gloc_atom(z): local Green's function per atom block (same size)
    """
    x = float(x)
    nk = Hhop_k.shape[0]
    norb_atom = V_A_atom.shape[0]
    norb_cell = 2 * norb_atom

    VA = backend.asarray(V_A_atom, dtype=complex)
    VB = backend.asarray(V_B_atom, dtype=complex)

    if sigma_init_atom is None:
        sigma = (1-x)*VA + x*VB
    else:
        sigma = backend.copy(backend.asarray(sigma_init_atom, dtype=complex))

    z_arr = backend.asarray(z, dtype=complex)
    Icell = backend.eye(norb_cell, dtype=complex)

    for it in range(max_iter):
        Sigma_cell = embed_onsite_in_cell(sigma, 2)
        Gsum = backend.zeros((norb_cell, norb_cell), dtype=complex)
        for Hk in Hhop_k:
            Gk = backend.inv(z_arr*Icell - Hk - Sigma_cell)
            Gsum = Gsum + Gk
        Gloc_cell = Gsum / nk
        Gloc = Gloc_cell[0:norb_atom, 0:norb_atom]

        G0_inv = backend.inv(Gloc) + sigma

        GA = backend.inv(G0_inv - VA)
        GB = backend.inv(G0_inv - VB)
        Gavg = (1-x)*GA + x*GB

        sigma_new = G0_inv - backend.inv(Gavg)

        err = float(backend.amax(backend.abs(sigma_new - sigma)))
        sigma = (1-mix)*sigma + mix*sigma_new
        if err < tol:
            return sigma, Gloc

    return sigma, Gloc

def cpa_solve_grid(
    energies,
    eta: float,
    Hhop_k,
    V_A_atom,
    V_B_atom,
    x: float,
    mix: float = 0.6,
    tol: float = 1e-7,
    max_iter: int = 200,
    continuation: bool = True,
) -> dict:
    """
    Solve CPA for an array of real energies, using z = E + i*eta.
    Returns dict with sigma(E), Gloc(E), and bookkeeping.
    """
    E = backend.asarray(energies, dtype=float)
    sigma_list = []
    gloc_list = []
    sigma_prev = None
    for i in range(E.shape[0]):
        Ei = float(E[i])
        z = complex(Ei, eta)
        sigma_prev, gloc = cpa_solve_energy(
            z, Hhop_k, V_A_atom, V_B_atom, x,
            sigma_init_atom=sigma_prev if continuation else None,
            mix=mix, tol=tol, max_iter=max_iter
        )
        sigma_list.append(backend.copy(sigma_prev))
        gloc_list.append(backend.copy(gloc))
    return {
        "energies": E,
        "eta": float(eta),
        "sigma_atom": backend.stack(sigma_list),
        "gloc_atom": backend.stack(gloc_list),
    }
