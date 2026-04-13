
"""
High-symmetry k-path generation using seekpath.

Generates standardised band-structure paths for the diamond/FCC lattice
following the HPKOT convention (Hinuma, Pizzi, Kumagai, Oba, Tanaka 2017).
"""
from __future__ import annotations

import numpy as np
import seekpath


# Default diamond Si structure for path generation
_DEFAULT_A = 5.431  # Si lattice constant (Angstrom)


def _diamond_structure(a: float, Z: int = 14):
    """Build seekpath input tuple for a diamond structure."""
    cell = (a / 2.0) * np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    positions = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
    numbers = [Z, Z]
    return (cell, positions, numbers)


_LABEL_MAP = {
    "GAMMA": "\u0393",
    "SIGMA_0": "\u03a3\u2080",
    "DELTA_0": "\u0394\u2080",
    "LAMBDA_0": "\u039b\u2080",
}


def _pretty_label(label: str) -> str:
    """Convert seekpath label names to Unicode symbols for plotting."""
    return _LABEL_MAP.get(label, label)


def make_kpath(a: float = _DEFAULT_A, n_per_segment: int = 60):
    """
    Generate a high-symmetry k-path for the diamond/FCC Brillouin zone.

    Uses seekpath with the HPKOT convention.

    Parameters
    ----------
    a : float
        Lattice constant in Angstrom (determines reciprocal space scale).
    n_per_segment : int
        Approximate number of k-points per segment.

    Returns
    -------
    k_frac : ndarray, shape (Nk, 3)
        Fractional coordinates in the reciprocal primitive basis.
    labels : list of (int, str)
        (index, label) pairs for tick marks on band plots.
    distances : ndarray, shape (Nk,)
        Cumulative linear distance along the path (1/Angstrom).
    """
    structure = _diamond_structure(a)

    # Estimate reference_distance from n_per_segment.
    # seekpath uses this as approximate spacing between k-points in 1/Ang.
    # A typical FCC BZ segment is ~1-2 1/Ang, so this gives roughly
    # the right density.
    ref_dist = 1.5 / max(n_per_segment, 1)

    result = seekpath.get_explicit_k_path(
        structure,
        reference_distance=ref_dist,
        with_time_reversal=True,
        recipe="hpkot",
    )

    k_frac = np.array(result["explicit_kpoints_rel"])
    distances = np.array(result["explicit_kpoints_linearcoord"])
    raw_labels = result["explicit_kpoints_labels"]

    labels = []
    for i, lbl in enumerate(raw_labels):
        if lbl:
            labels.append((i, _pretty_label(lbl)))

    return k_frac, labels, distances
