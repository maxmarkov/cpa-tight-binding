
"""
Parameter utilities for Si/Ge tight-binding (diamond structure) + VCA composition mixing.

Default parameter set: nearest-neighbor orthogonal sp3s* (Vogl, Hjalmarson, Dow 1983).
We store:
- on-site energies: Es, Ep, Es*
- two-center SK integrals: Vssσ, Vspσ, Vppσ, Vppπ, Vs*pσ
Optionally we include an atomic p-manifold SOC using Δ_SO (split-off energy).

Notes
-----
This code is meant as a compact, didactic reference implementation.
For production-quality ETB, consider larger bases (sp3d5s*) and/or second neighbors.
"""
from __future__ import annotations
from dataclasses import dataclass

from .. import backend

@dataclass(frozen=True)
class TBParams:
    # on-site energies (eV)
    Es: float
    Ep: float
    Es_star: float
    # SK integrals (eV)
    Vss_sigma: float
    Vsp_sigma: float
    Vpp_sigma: float
    Vpp_pi: float
    Vsstar_p_sigma: float
    # lattice constant (Angstrom)
    a: float
    # optional SOC split-off energy (eV); set 0.0 for no SOC
    delta_so: float = 0.0

# Room-temperature lattice constants (Å), commonly used
A_SI = 5.431
A_GE = 5.658

# Vogl (1983) orthogonal sp3s* parameters (homopolar diamond limit a=c)
# Values as widely circulated from Table 1 (units eV).
SI_VOGL = TBParams(
    Es=-4.2000, Ep= 1.7150, Es_star= 6.6850,
    Vss_sigma=-2.0750, Vsp_sigma= 2.4803,
    Vpp_sigma= 2.7163, Vpp_pi=-0.7150,
    Vsstar_p_sigma= 2.3271,
    a=A_SI,
    delta_so=0.044,  # optional
)

GE_VOGL = TBParams(
    Es=-5.8800, Ep= 1.6100, Es_star= 6.3900,
    Vss_sigma=-1.6950, Vsp_sigma= 2.3660,
    Vpp_sigma= 2.8525, Vpp_pi=-0.8225,
    Vsstar_p_sigma= 2.2590,
    a=A_GE,
    delta_so=0.290,  # optional
)

def vegard_a(x: float, a_si: float = A_SI, a_ge: float = A_GE) -> float:
    """Vegard's law lattice constant (Å)."""
    return (1.0 - x) * a_si + x * a_ge

def mix_params_vca(x: float, si: TBParams = SI_VOGL, ge: TBParams = GE_VOGL,
                   use_vegard_a: bool = True,
                   scale_hoppings_with_bond: bool = True,
                   harrison_n: float = 2.0,
                   include_soc: bool = False) -> TBParams:
    """
    VCA mixing of parameters at composition x (Ge fraction).
    - Onsite energies mixed linearly.
    - SK integrals mixed linearly then optionally scaled by bond length.
    """
    x = float(x)
    # base linear interpolation
    Es = (1-x)*si.Es + x*ge.Es
    Ep = (1-x)*si.Ep + x*ge.Ep
    Es_star = (1-x)*si.Es_star + x*ge.Es_star

    Vss = (1-x)*si.Vss_sigma + x*ge.Vss_sigma
    Vsp = (1-x)*si.Vsp_sigma + x*ge.Vsp_sigma
    VppS = (1-x)*si.Vpp_sigma + x*ge.Vpp_sigma
    VppP = (1-x)*si.Vpp_pi + x*ge.Vpp_pi
    Vsps = (1-x)*si.Vsstar_p_sigma + x*ge.Vsstar_p_sigma

    a = vegard_a(x, si.a, ge.a) if use_vegard_a else ((1-x)*si.a + x*ge.a)

    delta_so = (1-x)*si.delta_so + x*ge.delta_so if include_soc else 0.0

    if scale_hoppings_with_bond:
        # bond length in diamond: d = sqrt(3)*a/4
        d_si = backend.sqrt(3)*si.a/4.0
        d_ge = backend.sqrt(3)*ge.a/4.0
        d_x = backend.sqrt(3)*a/4.0
        # reference bond: VCA linear interpolation of integrals at reference bond d_ref
        d_ref = (1-x)*d_si + x*d_ge
        # scale with (d_ref/d_x)^n
        s = (d_ref / d_x) ** harrison_n
        Vss *= s; Vsp *= s; VppS *= s; VppP *= s; Vsps *= s

    return TBParams(
        Es=Es, Ep=Ep, Es_star=Es_star,
        Vss_sigma=Vss, Vsp_sigma=Vsp,
        Vpp_sigma=VppS, Vpp_pi=VppP,
        Vsstar_p_sigma=Vsps,
        a=a,
        delta_so=delta_so
    )

def onsite_matrix(params: TBParams, with_sstar: bool=True):
    """
    On-site matrix for one atom in basis [s, px, py, pz, s*] (5x5).
    If with_sstar=False -> [s, px, py, pz] (4x4).
    """
    if with_sstar:
        E = backend.diag([params.Es, params.Ep, params.Ep, params.Ep, params.Es_star])
    else:
        E = backend.diag([params.Es, params.Ep, params.Ep, params.Ep])
    return E

def disorder_onsites(si: TBParams = SI_VOGL, ge: TBParams = GE_VOGL, include_soc: bool=False):
    """
    Return (V_Si, V_Ge) onsite matrices for the diagonal disorder model
    in basis [s,px,py,pz,s*]. SOC not included here (handled in Hamiltonian).
    """
    return onsite_matrix(si, True), onsite_matrix(ge, True)
