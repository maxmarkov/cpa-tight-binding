
"""
Green's function utilities: DOS and spectral function for VCA and CPA.
"""
from __future__ import annotations

from ..utils import backend

def dos_from_eigs(eigs, energies, eta: float):
    """
    Broadened DOS from eigenvalues using Lorentzian broadening.
    eigs: (Nk, Nb)
    energies: (Ne,)
    """
    E = energies[None, :, None]
    ev = eigs[:, None, :]
    lor = (eta/3.141592653589793)/((E-ev)**2 + eta**2)
    return backend.mean(backend.sum(lor, axis=2), axis=0)

def dos_from_gloc(gloc_atom):
    """
    DOS from local Green's function per atom block:
      D(E) = -(2/pi) Im Tr Gloc_atom(E)
    Factor 2 accounts for two atoms per cell (spinless).
    """
    tr = backend.trace(gloc_atom, axis1=1, axis2=2)
    return -(2.0/3.141592653589793)*backend.imag(tr)

def spectral_function_k(E: float, eta: float, Hk, Sigma_cell) -> float:
    """A(k,E) = -(1/pi) Im Tr [ (E+iη - Hk - Σ)^-1 ]"""
    z = backend.asarray(complex(E, eta), dtype=complex)
    Gk = backend.inv(z*backend.eye(Hk.shape[0], dtype=complex) - Hk - Sigma_cell)
    return -(1.0/3.141592653589793)*float(backend.imag(backend.trace(Gk)))

def spectral_map_kpath(energies, eta: float, Hhop_path,
                       onsite_cell, sigma_atom_E=None):
    """
    Compute spectral intensity A(k,E) along a k-path.
    If sigma_atom_E is None => VCA (use onsite_cell).
    If provided => CPA, sigma_atom_E has shape (Ne, norb_atom, norb_atom)
    """
    Ne = energies.shape[0] if hasattr(energies, 'shape') else len(energies)
    Nk = Hhop_path.shape[0]
    norb_atom = Hhop_path.shape[1] // 2
    A = backend.zeros((Ne, Nk), dtype=float)
    for ie in range(Ne):
        E_val = float(energies[ie])
        if sigma_atom_E is None:
            Sigma_cell = onsite_cell
        else:
            Sigma_cell = backend.zeros((2*norb_atom, 2*norb_atom), dtype=complex)
            for a in range(2):
                o = a * norb_atom
                Sigma_cell[o:o+norb_atom, o:o+norb_atom] = sigma_atom_E[ie]
        for ik in range(Nk):
            Hk = Hhop_path[ik]
            A[ie, ik] = spectral_function_k(E_val, eta, Hk, Sigma_cell)
    return A
