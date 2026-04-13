
"""
Lattice helpers for the diamond structure (FCC Bravais + 2-atom basis)
and k-point generation.
"""
from __future__ import annotations

from .. import backend

def fcc_primitive_vectors(a: float):
    """
    Primitive lattice vectors for FCC (in Cartesian Å):
        a1 = (0, a/2, a/2)
        a2 = (a/2, 0, a/2)
        a3 = (a/2, a/2, 0)
    """
    return backend.array([
        [0.0, a/2, a/2],
        [a/2, 0.0, a/2],
        [a/2, a/2, 0.0],
    ], dtype=float)

def reciprocal_vectors(a: float):
    """Reciprocal primitive vectors (Cartesian 1/Å) for FCC primitive vectors."""
    A = fcc_primitive_vectors(a)
    V = backend.dot(A[0], backend.cross(A[1], A[2]))
    two_pi = 2 * 3.141592653589793
    b1 = two_pi * backend.cross(A[1], A[2]) / V
    b2 = two_pi * backend.cross(A[2], A[0]) / V
    b3 = two_pi * backend.cross(A[0], A[1]) / V
    return backend.array([b1, b2, b3], dtype=float)

def diamond_basis(a: float):
    """Basis positions (Å) for the 2 atoms in the primitive cell."""
    tauA = backend.array([0.0, 0.0, 0.0])
    tauB = backend.array([a/4, a/4, a/4])
    return tauA, tauB

def diamond_nn_vectors(a: float):
    """
    Nearest-neighbor vectors from A to B atoms (Å):
      (a/4) * (±1, ±1, ±1) with an even number of minus signs.
    Equivalent set:
      (+,+,+), (+,-,-), (-,+,-), (-,-,+)
    """
    c = a/4.0
    return backend.array([
        [ c, c, c],
        [ c,-c,-c],
        [-c, c,-c],
        [-c,-c, c],
    ], dtype=float)

def monkhorst_pack(n1: int, n2: int, n3: int, shift: float = 0.5):
    """
    Monkhorst-Pack grid in fractional coordinates of reciprocal primitive basis.
    Returns (N,3) fractional points in [0,1).
    Default shift=0.5 gives the standard centered grid.
    """
    n1,n2,n3 = int(n1), int(n2), int(n3)
    g = []
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                u = (i + shift)/n1 - 0.5
                v = (j + shift)/n2 - 0.5
                w = (k + shift)/n3 - 0.5
                g.append([u,v,w])
    return backend.array(g, dtype=float)

def frac_to_cart_k(k_frac, bvec):
    """Convert k in fractional reciprocal primitive coords to Cartesian (1/Å)."""
    return backend.dot(k_frac, bvec)

