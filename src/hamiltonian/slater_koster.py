
"""
Slater-Koster hopping matrices for an orthogonal nearest-neighbor sp3s* model.

Basis per atom: [s, px, py, pz, s*]
Returns hopping matrix T (5x5) from atom A orbitals to atom B orbitals
for a bond direction with direction cosines (l,m,n).
"""
from __future__ import annotations

from .. import backend
from .params import TBParams

def hopping_sp3s_star(params: TBParams, direction) -> object:
    v = backend.asarray(direction)
    l, m, n = float(v[0]), float(v[1]), float(v[2])
    # Normalize just in case
    r = backend.sqrt(l*l + m*m + n*n)
    if r == 0:
        raise ValueError("Zero direction vector.")
    l, m, n = l/r, m/r, n/r

    Vss = params.Vss_sigma
    Vsp = params.Vsp_sigma
    VppS = params.Vpp_sigma
    VppP = params.Vpp_pi
    Vsps = params.Vsstar_p_sigma

    T = backend.zeros((5, 5), dtype=float)

    # s - s
    T[0,0] = Vss

    # s - p  (A:s -> B:p)
    T[0,1] =  l*Vsp
    T[0,2] =  m*Vsp
    T[0,3] =  n*Vsp

    # p - s  (A:p -> B:s) = - (B:s -> A:p) for orthogonal two-center? 
    # In SK convention, <p|H|s> = -<s|H|p> for same bond direction.
    T[1,0] = -l*Vsp
    T[2,0] = -m*Vsp
    T[3,0] = -n*Vsp

    # p - p
    # <px|px> = l^2 Vppσ + (1-l^2) Vppπ
    # <px|py> = l m (Vppσ - Vppπ)
    def pp(a,b,la,lb):
        if a==b:
            return (la*la)*VppS + (1-la*la)*VppP
        else:
            return la*lb*(VppS - VppP)

    dirs = [l,m,n]
    for i in range(3):
        for j in range(3):
            T[1+i, 1+j] = pp(i,j, dirs[i], dirs[j])

    # s* - p  (A:s* -> B:p)
    T[4,1] =  l*Vsps
    T[4,2] =  m*Vsps
    T[4,3] =  n*Vsps

    # p - s* (A:p -> B:s*) = - (A:s* -> B:p) ? same sign convention as s-p
    T[1,4] = -l*Vsps
    T[2,4] = -m*Vsps
    T[3,4] = -n*Vsps

    # s* - s and s* - s* are set to 0 in this minimal parameterization
    return T
