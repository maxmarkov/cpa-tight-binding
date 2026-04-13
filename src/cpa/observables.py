
"""
High-level CPA observables that combine solver output with analysis.

- Spectral gap / band gap extraction from DOS
- Composition sweep and bowing-parameter fitting
"""
from __future__ import annotations

from ..utils import backend


def find_spectral_gap(dos, energies, threshold_frac=0.01, search_range=(-3.0, 6.0)):
    """
    Find the band gap from a DOS curve.

    Scans the DOS within *search_range* for the widest contiguous region
    where DOS < threshold_frac * max(DOS).

    Parameters
    ----------
    dos : (Ne,) array
    energies : (Ne,) array
    threshold_frac : float
        Fraction of peak DOS used as the "gap" threshold.
    search_range : (float, float)
        Energy window (eV) to search for the gap.

    Returns
    -------
    dict with ``'E_vbm'``, ``'E_cbm'``, ``'gap'``  or ``None`` if no gap found.
    """
    import numpy as _np
    E_np = backend.to_numpy(energies)
    D_np = backend.to_numpy(dos)

    mask = (E_np >= search_range[0]) & (E_np <= search_range[1])
    idx = _np.where(mask)[0]
    if len(idx) == 0:
        return None

    thresh = threshold_frac * D_np[idx].max()
    below = D_np[idx] < thresh

    # find longest contiguous run of True in `below`
    best_start = best_len = 0
    cur_start = cur_len = 0
    for i, b in enumerate(below):
        if b:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0

    if best_len < 2:
        return None

    E_vbm = float(E_np[idx[best_start]])
    E_cbm = float(E_np[idx[best_start + best_len - 1]])
    return {'E_vbm': E_vbm, 'E_cbm': E_cbm, 'gap': E_cbm - E_vbm}


def fit_bowing(x_arr, gap_arr):
    """
    Fit the alloy band gap to:

        E_gap(x) = (1-x)*E_gap(0) + x*E_gap(1) - b*x*(1-x)

    Returns the bowing parameter *b* (positive means downward bowing).

    Uses simple least-squares on interior points.
    """
    import numpy as _np
    x = _np.asarray(x_arr, dtype=float)
    g = _np.asarray(gap_arr, dtype=float)
    E0 = g[0] if len(g) > 0 else 0.0
    E1 = g[-1] if len(g) > 0 else 0.0

    linear = (1 - x) * E0 + x * E1
    deviation = linear - g                     # b * x*(1-x)
    quad = x * (1 - x)
    valid = quad > 1e-12
    if not _np.any(valid):
        return 0.0
    # least-squares: b = sum(dev * quad) / sum(quad^2) over valid points
    b = float(_np.sum(deviation[valid] * quad[valid]) / _np.sum(quad[valid] ** 2))
    return b
