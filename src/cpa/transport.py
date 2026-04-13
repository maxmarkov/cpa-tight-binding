
"""
Transport properties from CPA Green's functions via the Kubo-Greenwood formalism.

Implements:
- Transport distribution function (TDF)
- Optical conductivity sigma(omega) and dielectric function
- DC conductivity (omega -> 0 limit)
- Thermoelectric coefficients (Seebeck S, electronic thermal conductivity kappa_e)
  from generalised transport integrals L_n

All quantities are computed in the single-site CPA "bubble" approximation
(vertex corrections neglected).
"""
from __future__ import annotations

import math

from ..utils import backend

_PI = math.pi

# ---- physical constants in eV-Angstrom-second-Kelvin units ----
HBAR_EV_S = 6.582119569e-16       # hbar in eV*s
E_CHARGE  = 1.602176634e-19       # electron charge in C
KB_EV     = 8.617333262e-5        # Boltzmann constant in eV/K
BOHR_TO_ANG = 0.529177210903      # 1 bohr in Angstrom


# ---------------------------------------------------------------------------
#  Fermi-Dirac helpers
# ---------------------------------------------------------------------------

def fermi(E, mu, temperature):
    """Fermi-Dirac distribution f(E) = 1/(exp((E-mu)/kT) + 1)."""
    if temperature <= 0:
        return 1.0 if E < mu else (0.5 if E == mu else 0.0)
    x = (E - mu) / (KB_EV * temperature)
    if x > 40:
        return 0.0
    if x < -40:
        return 1.0
    return 1.0 / (math.exp(x) + 1.0)


def _neg_df_dE_array(energies, mu, temperature):
    """
    -df/dE evaluated on an energy array.  Returns backend array (Ne,).

    This is the thermal window centred on mu with width ~ kT.
    """
    if temperature <= 0:
        # delta-function approximation: pick the grid point closest to mu
        E_np = backend.to_numpy(energies)
        import numpy as _np
        out = _np.zeros_like(E_np)
        idx = int(_np.argmin(_np.abs(E_np - mu)))
        dE = float(E_np[1] - E_np[0]) if len(E_np) > 1 else 1.0
        out[idx] = 1.0 / dE
        return backend.asarray(out)
    kT = KB_EV * temperature
    vals = []
    for i in range(energies.shape[0]):
        Ei = float(energies[i])
        x = (Ei - mu) / kT
        if abs(x) > 40:
            vals.append(0.0)
        else:
            ex = math.exp(x)
            vals.append(ex / (kT * (ex + 1.0) ** 2))
    return backend.array(vals, dtype=float)


# ---------------------------------------------------------------------------
#  Core: transport distribution function  Xi(E)
# ---------------------------------------------------------------------------

def _build_sigma_cell(sigma_atom_ie, norb_atom):
    """Embed per-atom self-energy into cell-diagonal block."""
    norb_cell = 2 * norb_atom
    Sigma_cell = backend.zeros((norb_cell, norb_cell), dtype=complex)
    for a in range(2):
        o = a * norb_atom
        Sigma_cell[o:o + norb_atom, o:o + norb_atom] = sigma_atom_ie
    return Sigma_cell


def transport_distribution(energies, eta, Hhop_k, k_frac_grid, params,
                           sigma_atom_E, alpha=0):
    """
    Transport distribution function (TDF):

        Xi(E) = (1/Nk) sum_k Tr[ v_a(k) A(k,E) v_a(k) A(k,E) ]

    where A(k,E) = -(1/pi) Im G(k,E)  (full matrix, not trace).

    Parameters
    ----------
    energies : (Ne,) real energy grid
    eta : float, broadening
    Hhop_k : (Nk, Norb, Norb) hopping-only Hamiltonians
    k_frac_grid : (Nk, 3) fractional k-points
    params : TBParams (for velocity matrix computation)
    sigma_atom_E : (Ne, norb_atom, norb_atom) CPA self-energy
    alpha : int, Cartesian direction (0/1/2).  For cubic systems all are equal.

    Returns
    -------
    tdf : (Ne,) transport distribution in units of (eV*Ang)^2 / eV^2
    """
    from ..hamiltonian.bloch import velocity_matrix_sp3s_star

    Ne = energies.shape[0] if hasattr(energies, 'shape') else len(energies)
    Nk = Hhop_k.shape[0]
    norb_atom = Hhop_k.shape[1] // 2
    norb_cell = Hhop_k.shape[1]

    tdf = backend.zeros((Ne,), dtype=float)

    for ik in range(Nk):
        kf = k_frac_grid[ik]
        Hk = Hhop_k[ik]
        v_a = velocity_matrix_sp3s_star(kf, params, alpha)

        for ie in range(Ne):
            Ei = float(energies[ie])
            z = backend.asarray(complex(Ei, eta), dtype=complex)
            Sigma_cell = _build_sigma_cell(sigma_atom_E[ie], norb_atom)
            Gk = backend.inv(z * backend.eye(norb_cell, dtype=complex) - Hk - Sigma_cell)
            # Anti-Hermitian part: Im G = (G - G^dag)/(2i)
            ImG = (Gk - backend.H(Gk)) / backend.asarray(2j, dtype=complex)
            Ak = -(1.0 / _PI) * ImG   # Hermitian, real eigenvalues
            # Tr[ v A v A ]  (take real part; imaginary part is zero by symmetry)
            vA = backend.dot(v_a, Ak)
            tdf[ie] = tdf[ie] + float(backend.real(backend.trace(backend.dot(vA, vA))))

    tdf = tdf / Nk
    return tdf


# ---------------------------------------------------------------------------
#  Transport integrals  L_n
# ---------------------------------------------------------------------------

def transport_integrals(energies, tdf, mu, temperature, n_max=2):
    """
    Generalised transport integrals:

        L_n = integral dE (-df/dE) (E - mu)^n Xi(E)

    Parameters
    ----------
    energies : (Ne,)
    tdf : (Ne,) transport distribution function
    mu : float, chemical potential (eV)
    temperature : float (K)
    n_max : int, compute L_0 .. L_{n_max}

    Returns
    -------
    dict  ``{'L0': float, 'L1': float, 'L2': float, ...}``
    """
    ndf = _neg_df_dE_array(energies, mu, temperature)
    E_np = backend.to_numpy(energies)
    tdf_np = backend.to_numpy(tdf)
    ndf_np = backend.to_numpy(ndf)
    dE = float(E_np[1] - E_np[0]) if len(E_np) > 1 else 1.0

    result = {}
    for n in range(n_max + 1):
        integrand = ndf_np * ((E_np - mu) ** n) * tdf_np
        result[f'L{n}'] = float(integrand.sum() * dE)
    return result


# ---------------------------------------------------------------------------
#  Optical conductivity  sigma(omega)
# ---------------------------------------------------------------------------

def _sigma_prefactor(volume_cell):
    """
    Unit-conversion prefactor for conductivity.

    In our natural units (hbar=1, eV, Angstrom) the dimensionless
    conductivity integral has units Ang^2 / Ang^3 = 1/Ang.
    To convert to S/m:

        sigma(S/m) = sigma_nat(1/Ang) * e^2 / (hbar_SI * 1 Ang_in_m)

    Returns (prefactor_Sm, prefactor_Ocm) for a given cell volume.
    """
    hbar_SI = HBAR_EV_S * E_CHARGE            # J*s
    e2_over_hbar_ang = E_CHARGE**2 / (hbar_SI * 1e-10)  # S/m per (1/Ang)
    return e2_over_hbar_ang


def optical_conductivity(energies, tdf, omega_grid, mu, temperature,
                         volume_cell):
    """
    Real part of the optical conductivity (Kubo-Greenwood, constant-matrix-element
    approximation):

        sigma_1(omega) = (pi / V) int dE [f(E)-f(E+omega)]/omega * Xi(E)

    converted to (Ohm*cm)^-1.

    Parameters
    ----------
    energies : (Ne,)
    tdf : (Ne,) transport distribution function
    omega_grid : (Nomega,) photon energies in eV (> 0)
    mu, temperature, volume_cell : as elsewhere

    Returns
    -------
    sigma_1 : (Nomega,) in (Ohm*cm)^{-1}
    epsilon_2 : (Nomega,) imaginary part of dielectric function
    """
    import numpy as _np
    E_np = backend.to_numpy(energies)
    tdf_np = backend.to_numpy(tdf)
    dE = float(E_np[1] - E_np[0]) if len(E_np) > 1 else 1.0
    Ne = len(E_np)

    conv = _sigma_prefactor(volume_cell)
    hbar_SI = HBAR_EV_S * E_CHARGE
    eps0 = 8.854187817e-12

    omega_np = backend.to_numpy(backend.asarray(omega_grid))
    Nomega = len(omega_np)

    sigma_1_Ocm = _np.zeros(Nomega)
    epsilon_2 = _np.zeros(Nomega)

    for iw in range(Nomega):
        w = float(omega_np[iw])
        if w <= 0:
            continue
        integrand = _np.zeros(Ne)
        for ie in range(Ne):
            Ei = float(E_np[ie])
            fE = fermi(Ei, mu, temperature)
            fEw = fermi(Ei + w, mu, temperature)
            integrand[ie] = (fE - fEw) / w * tdf_np[ie]
        # natural units: pi/V * integral  (units 1/Ang)
        sigma_nat = _PI * integrand.sum() * dE / volume_cell
        sigma_Sm = sigma_nat * conv          # S/m
        sigma_1_Ocm[iw] = sigma_Sm * 1e-2   # (Ohm*cm)^-1

        omega_SI = w * E_CHARGE / hbar_SI    # rad/s
        epsilon_2[iw] = sigma_Sm / (eps0 * omega_SI)

    return backend.asarray(sigma_1_Ocm), backend.asarray(epsilon_2)


# ---------------------------------------------------------------------------
#  DC conductivity & resistivity
# ---------------------------------------------------------------------------

def dc_conductivity(energies, tdf, mu, temperature, volume_cell):
    """
    DC electrical conductivity (omega -> 0 limit):

        sigma_DC = (pi e^2) / (hbar V) integral dE (-df/dE) Xi(E)

    Parameters
    ----------
    energies, tdf, mu, temperature : as in transport_integrals
    volume_cell : float, primitive cell volume in Angstrom^3

    Returns
    -------
    sigma_dc : float, in (Ohm*cm)^{-1}
    rho_dc   : float, resistivity in Ohm*cm  (inf if sigma_dc ~ 0)
    """
    L = transport_integrals(energies, tdf, mu, temperature, n_max=0)
    L0 = L['L0']

    conv = _sigma_prefactor(volume_cell)
    sigma_nat = _PI * L0 / volume_cell   # 1/Ang
    sigma_Sm = sigma_nat * conv            # S/m
    sigma_Ocm = sigma_Sm * 1e-2            # (Ohm*cm)^-1

    rho = 1.0 / sigma_Ocm if abs(sigma_Ocm) > 1e-30 else float('inf')
    return sigma_Ocm, rho


# ---------------------------------------------------------------------------
#  Thermoelectric coefficients
# ---------------------------------------------------------------------------

def thermoelectric_coefficients(energies, tdf, mu, temperature, volume_cell):
    """
    Compute thermoelectric coefficients from generalised transport integrals.

    Parameters
    ----------
    energies : (Ne,)
    tdf : (Ne,) transport distribution function
    mu : float (eV)
    temperature : float (K)
    volume_cell : float (Ang^3)

    Returns
    -------
    dict with:
      'sigma'   : electrical conductivity in (Ohm*cm)^-1
      'seebeck' : Seebeck coefficient in V/K  (microV/K = 1e6 * seebeck)
      'kappa_e' : electronic thermal conductivity in W/(m*K)
      'L0', 'L1', 'L2' : raw transport integrals
    """
    L = transport_integrals(energies, tdf, mu, temperature, n_max=2)
    L0, L1, L2 = L['L0'], L['L1'], L['L2']

    conv = _sigma_prefactor(volume_cell)

    # --- sigma_DC ---
    sigma_nat = _PI * L0 / volume_cell   # 1/Ang
    sigma_Sm = sigma_nat * conv            # S/m
    sigma_Ocm = sigma_Sm * 1e-2            # (Ohm*cm)^-1

    # --- Seebeck ---
    # S = -(1/T) * L1/L0   [L1/L0 is in eV, so S is eV/K = V/K]
    seebeck = -(L1 / L0) / temperature if abs(L0) > 1e-30 else 0.0  # V/K

    # --- Electronic thermal conductivity ---
    # kappa_e / sigma = (L2/L0 - (L1/L0)^2) / T
    # ratio is in eV^2/K -> convert eV to J: * E_CHARGE
    # kappa(W/(m*K)) = sigma(S/m) * ratio(eV/K) * E_CHARGE
    # because  S/m * eV/K * C = (A/V/m) * (C*V)/K * C = A*C^2/(m*K) ...
    # Actually: S * eV = (C/s) * (C*V) / V = C^2/s = A*C = W, so S*eV/K = W/K. Per meter: W/(m*K). ✓
    if abs(L0) > 1e-30 and temperature > 0:
        lorenz_term = L2 - L1**2 / L0   # eV^2 * Ang^2
        ratio = lorenz_term / (L0 * temperature)  # eV^2/K (Ang cancel)
        kappa_e = sigma_Sm * ratio * E_CHARGE  # W/(m*K)
    else:
        kappa_e = 0.0

    return {
        'sigma': sigma_Ocm,
        'seebeck': seebeck,
        'kappa_e': kappa_e,
        'L0': L0,
        'L1': L1,
        'L2': L2,
    }


# ---------------------------------------------------------------------------
#  Cell volume helper
# ---------------------------------------------------------------------------

def diamond_cell_volume(a: float) -> float:
    """Primitive cell volume of diamond/FCC in Angstrom^3.  V = a^3 / 4."""
    return a**3 / 4.0
