
"""
Green's function utilities: DOS and spectral function for VCA and CPA.
"""
from __future__ import annotations

from ..utils import backend

_PI = 3.141592653589793


def dos_from_eigs(eigs, energies, eta: float):
    """
    Broadened DOS from eigenvalues using Lorentzian broadening.
    eigs: (Nk, Nb)
    energies: (Ne,)
    """
    E = energies[None, :, None]
    ev = eigs[:, None, :]
    lor = (eta / _PI) / ((E - ev) ** 2 + eta ** 2)
    return backend.mean(backend.sum(lor, axis=2), axis=0)

def dos_from_gloc(gloc_atom):
    """
    DOS from local Green's function per atom block:
      D(E) = -(2/pi) Im Tr Gloc_atom(E)
    Factor 2 accounts for two atoms per cell (spinless).
    """
    tr = backend.trace(gloc_atom, axis1=1, axis2=2)
    return -(2.0 / _PI) * backend.imag(tr)

def spectral_function_k(E: float, eta: float, Hk, Sigma_cell) -> float:
    """A(k,E) = -(1/pi) Im Tr [ (E+iη - Hk - Σ)^-1 ]"""
    z = backend.asarray(complex(E, eta), dtype=complex)
    Gk = backend.inv(z*backend.eye(Hk.shape[0], dtype=complex) - Hk - Sigma_cell)
    return -(1.0 / _PI) * float(backend.imag(backend.trace(Gk)))

def spectral_function_k_matrix(E: float, eta: float, Hk, Sigma_cell):
    """
    Full spectral matrix A(k,E) = -(1/pi) Im G(k,E).

    Uses the anti-Hermitian part  Im G = (G - G^dag) / (2i)  so that
    A is guaranteed Hermitian (and positive semi-definite for causal Sigma).
    """
    z = backend.asarray(complex(E, eta), dtype=complex)
    Gk = backend.inv(z * backend.eye(Hk.shape[0], dtype=complex) - Hk - Sigma_cell)
    # anti-Hermitian part: (G - G^dag) / (2i)
    ImG = (Gk - backend.H(Gk)) / backend.asarray(2j, dtype=complex)
    return -(1.0 / _PI) * backend.real(ImG)


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


# ---------------------------------------------------------------------------
#  Orbital-resolved (partial) DOS
# ---------------------------------------------------------------------------

def pdos_from_gloc(gloc_atom):
    """
    Orbital-resolved partial DOS from CPA local Green's function.

    Parameters
    ----------
    gloc_atom : array (Ne, norb_atom, norb_atom)

    Returns
    -------
    dict with keys ``'s'``, ``'p'``, ``'sstar'``, ``'total'``,
    each an array of shape ``(Ne,)``.  Factor 2 for two atoms per cell.
    """
    diag = backend.diagonal(gloc_atom, axis1=1, axis2=2)   # (Ne, norb)
    diag_im = backend.imag(diag)
    prefactor = -2.0 / _PI
    s_dos     = prefactor * diag_im[:, 0]
    p_dos     = prefactor * (diag_im[:, 1] + diag_im[:, 2] + diag_im[:, 3])
    sstar_dos = prefactor * diag_im[:, 4]
    return {
        's':     s_dos,
        'p':     p_dos,
        'sstar': sstar_dos,
        'total': s_dos + p_dos + sstar_dos,
    }


# ---------------------------------------------------------------------------
#  Quasiparticle scattering rates
# ---------------------------------------------------------------------------

def scattering_rates(sigma_atom):
    """
    Per-orbital scattering rates from CPA self-energy.

    tau^{-1}_alpha(E) = -2 Im Sigma_{alpha,alpha}(E)

    Parameters
    ----------
    sigma_atom : array (Ne, norb_atom, norb_atom)

    Returns
    -------
    dict with keys ``'s'``, ``'p'``, ``'sstar'``, each ``(Ne,)`` in eV.
    """
    diag_im = backend.imag(backend.diagonal(sigma_atom, axis1=1, axis2=2))
    return {
        's':     -2.0 * diag_im[:, 0],
        'p':     -2.0 * diag_im[:, 1],   # px; py=pz by cubic symmetry
        'sstar': -2.0 * diag_im[:, 4],
    }


# ---------------------------------------------------------------------------
#  Effective (renormalised) band structure from spectral-function peaks
# ---------------------------------------------------------------------------

def effective_bands(A_kE, energies, min_prominence_frac=0.08):
    """
    Extract disorder-renormalised band positions and widths from A(k,E).

    For each k-point column, find local maxima of A(k,E) vs E whose
    prominence exceeds ``min_prominence_frac * max(A)`` at that k.

    Parameters
    ----------
    A_kE : array (Ne, Nk)
    energies : array (Ne,)
    min_prominence_frac : float
        Minimum peak prominence as a fraction of the column maximum.

    Returns
    -------
    list of length Nk.  Each element is a list of ``(E_peak, fwhm)`` tuples.
    ``fwhm`` is estimated from half-maximum crossings (NaN if unresolved).
    """
    import numpy as np
    A_np = backend.to_numpy(A_kE)
    E_np = backend.to_numpy(energies)
    Ne, Nk = A_np.shape
    dE = float(E_np[1] - E_np[0]) if Ne > 1 else 1.0
    result = []
    for ik in range(Nk):
        col = A_np[:, ik]
        col_max = col.max()
        if col_max <= 0:
            result.append([])
            continue
        prom_thresh = min_prominence_frac * col_max
        peaks = []
        for i in range(1, Ne - 1):
            if col[i] > col[i - 1] and col[i] > col[i + 1]:
                # estimate prominence (simplified: min of left/right dips)
                left_min = col[:i].min() if i > 0 else 0.0
                right_min = col[i + 1:].min() if i < Ne - 1 else 0.0
                prom = col[i] - max(left_min, right_min)
                if prom < prom_thresh:
                    continue
                half = col[i] / 2.0
                # FWHM: search left and right for half-max crossing
                left_e = float('nan')
                for j in range(i - 1, -1, -1):
                    if col[j] <= half:
                        left_e = E_np[j] + (half - col[j]) / (col[j + 1] - col[j]) * dE
                        break
                right_e = float('nan')
                for j in range(i + 1, Ne):
                    if col[j] <= half:
                        right_e = E_np[j - 1] + (col[j - 1] - half) / (col[j - 1] - col[j]) * dE
                        break
                fwhm = right_e - left_e if not (np.isnan(left_e) or np.isnan(right_e)) else float('nan')
                peaks.append((float(E_np[i]), fwhm))
        result.append(peaks)
    return result
