
"""
High-symmetry k-path for FCC Brillouin zone (Setyawan & Curtarolo convention).

Points are given in fractional coordinates of reciprocal primitive basis (b1,b2,b3)
for the FCC primitive cell.
"""
from __future__ import annotations
from dataclasses import dataclass

from . import backend

@dataclass(frozen=True)
class KPoint:
    name: str
    k_frac: object

def fcc_high_symmetry_points() -> dict:
    # Setyawan & Curtarolo for FCC (conventional cubic):
    # Here expressed in reciprocal primitive basis of FCC primitive vectors.
    # We use a commonly used equivalent set for the FCC primitive basis:
    # Γ=(0,0,0)
    # X=(0,1/2,1/2)
    # W=(1/4,3/4,1/2)
    # K=(3/8,3/4,3/8)
    # L=(1/2,1/2,1/2)
    return {
        "G": backend.array([0.0, 0.0, 0.0]),
        "X": backend.array([0.0, 0.5, 0.5]),
        "W": backend.array([0.25, 0.75, 0.5]),
        "K": backend.array([0.375, 0.75, 0.375]),
        "L": backend.array([0.5, 0.5, 0.5]),
    }

def make_kpath(n_per_segment: int = 60):
    """
    Returns:
      k_frac: (Nk,3) fractional k-points along Γ-X-W-K-Γ-L
      labels: list of (index, label) for tick marks
    """
    pts = fcc_high_symmetry_points()
    order = [("G","Γ"), ("X","X"), ("W","W"), ("K","K"), ("G","Γ"), ("L","L")]
    k_list = []
    labels = []
    idx = 0
    for (a_key,a_lab), (b_key,b_lab) in zip(order[:-1], order[1:]):
        ka = pts[a_key]; kb = pts[b_key]
        if not k_list:
            k_list.append(backend.copy(ka))
            labels.append((idx, a_lab))
        for t in range(1, n_per_segment+1):
            f = t/(n_per_segment)
            k_list.append((1-f)*ka + f*kb)
        idx = len(k_list)-1
        labels.append((idx, b_lab))
    k_frac = backend.array(k_list, dtype=float)
    return k_frac, labels

def kpath_distances(k_cart) -> object:
    """Cumulative distance along a k-path in Cartesian coordinates."""
    d_list = [0.0]
    for i in range(1, len(k_cart)):
        d_list.append(d_list[-1] + float(backend.norm(k_cart[i]-k_cart[i-1])))
    return backend.array(d_list, dtype=float)
