
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
            Ak = -(1.0 / _PI) * backend.imag(Gk)
            # Tr[ v A v A ]
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

def optical_conductivity(energies, tdf, omega_grid, mu, temperature,
                         volume_cell):
    """
    Real part of the optical conductivity in the bubble approximation:

        sigma_1(omega) = (pi e^2 / V_cell) *
            integral dE [f(E) - f(E+omega)] / omega  *  Xi_conv(E, omega)

    For a more tractable formula we use the joint-spectral convolution
    approach:  at each omega we convolve the TDF with the Fermi window.

    For simplicity we use the Kubo-Greenwood form at finite omega,
    approximated via the TDF:

        sigma_1(omega) ~ (pi e^2 / V_cell) *
            integral dE  [f(E) - f(E+omega)] / omega  * Xi(E)

    This is the "constant-matrix-element" approximation where the velocity
    matrix element is taken at energy E (valid when omega << bandwidth).

    Parameters
    ----------
    energies : (Ne,)
    tdf : (Ne,) transport distribution function
    omega_grid : (Nomega,) photon energies in eV (must be > 0)
    mu : float (eV)
    temperature : float (K)
    volume_cell : float, primitive cell volume in Angstrom^3

    Returns
    -------
    sigma_1 : (Nomega,) in (Ohm*cm)^{-1}
    epsilon_2 : (Nomega,) imaginary part of dielectric function (dimensionless)
    """
    import numpy as _np
    E_np = backend.to_numpy(energies)
    tdf_np = backend.to_numpy(tdf)
    dE = float(E_np[1] - E_np[0]) if len(E_np) > 1 else 1.0
    Ne = len(E_np)

    # prefactor: pi * e^2 / V_cell  in SI then convert to (Ohm*cm)^-1
    # e^2 in eV*Ang: e^2/(4 pi eps0) = 14.3996 eV*Ang, but here we need e^2 alone.
    # Work in SI: sigma has units 1/(Ohm*m).
    # sigma = (pi * e^2 / V_cell_m3) * integral (1/eV) ... eV * (eV*Ang)^2 / eV^2
    # Convert Ang^2 to m^2 and eV integral units.
    V_m3 = volume_cell * 1e-30          # Ang^3 -> m^3
    # The TDF has units (eV*Ang)^2 / eV^2 = Ang^2.
    # integral dE * Xi(E) * [f(E)-f(E+w)]/w has units eV * Ang^2 * 1/eV = Ang^2
    # sigma = pi * e^2 / V_m3 * Ang^2 -> pi * e^2 * 1e-20 / V_m3  in S/m
    prefactor_Sm = _PI * E_CHARGE**2 * 1e-20 / (V_m3 * HBAR_EV_S)
    # Above includes 1/hbar because velocity was in eV*Ang/hbar units
    # Actually v = (1/hbar) dH/dk in eV*Ang, so v has units eV*Ang/hbar.
    # TDF = sum Tr[v A v A] has units (eV*Ang)^2/hbar^2 * (1/eV)^2 = Ang^2/hbar^2
    # integral dE (-df/dE)(...) gives * eV, so L_0 ~ eV*Ang^2/hbar^2
    # sigma = pi e^2 L_0 / V  where L_0 in eV*Ang^2/hbar^2
    # sigma (S/m) = pi * e^2 * L_0(eV * Ang^2 / hbar^2) / V(m^3)
    #   convert eV -> J:  * e,  Ang^2 -> m^2: * 1e-20,  hbar^2(eV*s)^2 -> (J*s)^2: * e^2
    # sigma = pi * e^2 * (L_0_val * e * 1e-20 / (hbar_SI)^2) / V_m3
    hbar_SI = HBAR_EV_S * E_CHARGE   # in J*s
    conv = _PI * E_CHARGE**2 * E_CHARGE * 1e-20 / (hbar_SI**2)
    # conv has units C^2 * J * m^2 / (J*s)^2 = C^2/(s^2 * J) ... = S * J / m ... let's be systematic
    # sigma(S/m) = conv * integral_value / V_m3
    # where integral_value is in eV * Ang^2 * (1/hbar_eV)^2 ...
    # Let's just use a known result:
    # sigma_DC = (pi e^2)/(hbar * V) * integral dE (-df/dE) Xi(E)
    # with Xi in Ang^2/s (if v in Ang/s) ...
    # Simpler: compute everything in eV-Ang-hbar=1 system then convert at the end.
    # In natural units (hbar=1, energies in eV, lengths in Ang):
    #   sigma_nat = pi * integral dE (-df/dE) Xi(E) / V_cell_Ang3
    #   has units 1/Ang (since Xi~Ang^2, E~eV, (-df/dE)~1/eV, V~Ang^3)
    # Convert to SI:  sigma(S/m) = sigma_nat * e^2 / (hbar * Ang)
    #   = sigma_nat * e^2 / (hbar_SI * 1e-10)
    e2_over_hbar_ang = E_CHARGE**2 / (hbar_SI * 1e-10)   # S/m per (1/Ang)

    Nomega = len(omega_grid) if hasattr(omega_grid, '__len__') else omega_grid.shape[0]
    omega_np = backend.to_numpy(backend.asarray(omega_grid))

    sigma_1 = _np.zeros(Nomega)
    for iw in range(Nomega):
        w = float(omega_np[iw])
        if w <= 0:
            continue
        # [f(E) - f(E+w)] / w  integrated against Xi(E)
        integrand = _np.zeros(Ne)
        for ie in range(Ne):
            Ei = float(E_np[ie])
            fE = fermi(Ei, mu, temperature)
            fEw = fermi(Ei + w, mu, temperature)
            integrand[ie] = (fE - fEw) / w * tdf_np[ie]
        sigma_1[iw] = _PI * integrand.sum() * dE

    # sigma_1 is now in 1/Ang^3 (natural units), convert to S/m
    sigma_1_Sm = sigma_1 * e2_over_hbar_ang
    # convert S/m to (Ohm*cm)^-1:  1 S/m = 0.01 (Ohm*cm)^-1
    sigma_1_Ocm = sigma_1_Sm * 0.01 / volume_cell  # WAIT -- V already divided out above? No.
    # Let me redo cleanly:
    # sigma_nat(w) = pi/V * integral, where V = volume_cell in Ang^3
    # so sigma_nat is in 1/Ang, and sigma(S/m) = sigma_nat * e^2/(hbar_SI * 1e-10)
    sigma_1_nat = sigma_1 / volume_cell   # now in 1/Ang
    sigma_1_Sm_final = sigma_1_nat * e2_over_hbar_ang
    sigma_1_Ocm_final = sigma_1_Sm_final * 1e-2  # S/m -> (Ohm*cm)^-1

    # epsilon_2(omega) = sigma_1(omega) / (epsilon_0 * omega)  (SI)
    # epsilon_0 = 8.854e-12 F/m
    eps0 = 8.854187817e-12
    epsilon_2 = _np.zeros(Nomega)
    for iw in range(Nomega):
        w = float(omega_np[iw])
        if w > 0:
            omega_SI = w * E_CHARGE / hbar_SI   # rad/s
            epsilon_2[iw] = sigma_1_Sm_final[iw] / (eps0 * omega_SI)

    return backend.asarray(sigma_1_Ocm_final), backend.asarray(epsilon_2)


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
    rho_dc   : float, resistivity in Ohm*cm  (inf if sigma_dc == 0)
    """
    L = transport_integrals(energies, tdf, mu, temperature, n_max=0)
    L0 = L['L0']

    hbar_SI = HBAR_EV_S * E_CHARGE
    e2_over_hbar_ang = E_CHARGE**2 / (hbar_SI * 1e-10)

    sigma_nat = _PI * L0 / volume_cell          # 1/Ang
    sigma_Sm = sigma_nat * e2_over_hbar_ang      # S/m
    sigma_Ocm = sigma_Sm * 1e-2                   # (Ohm*cm)^-1

    rho = 1.0 / sigma_Ocm if sigma_Ocm > 0 else float('inf')
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

    hbar_SI = HBAR_EV_S * E_CHARGE
    e2_over_hbar_ang = E_CHARGE**2 / (hbar_SI * 1e-10)

    # sigma_DC
    sigma_nat = _PI * L0 / volume_cell
    sigma_Sm = sigma_nat * e2_over_hbar_ang
    sigma_Ocm = sigma_Sm * 1e-2

    # Seebeck  S = -(1/(eT)) * L1/L0   (e in eV units = 1)
    if abs(L0) > 1e-30:
        seebeck_eV_per_K = -(1.0 / temperature) * L1 / L0   # eV/(eV*K) = 1/K ... no
        # L1 has units eV * [same as L0], so L1/L0 is in eV.
        # S = -(1/(eT)) L1/L0;  with e=|e|, T in K, L1/L0 in eV:
        # S in V/K = -(L1/L0) / T   since eV / (e * K) = V/K
        seebeck = -(L1 / L0) / temperature    # V/K
    else:
        seebeck = 0.0

    # electronic thermal conductivity
    # kappa_e = (pi e^2)/(hbar V T) * (L2 - L1^2/L0)
    # Same unit conversion as sigma, times eV^2 / T
    if abs(L0) > 1e-30:
        lorenz_term = L2 - L1**2 / L0   # eV^2 * [Ang^2 / hbar^2 units]
    else:
        lorenz_term = L2

    kappa_nat = _PI * lorenz_term / (volume_cell * temperature)  # eV/(Ang*K)
    # Convert: eV/(Ang*K) * e^2/(hbar*Ang^-1) * ...
    # kappa(W/(m*K)) = kappa_nat * e^2 * eV_to_J / (hbar_SI * 1e-10)
    # but kappa has extra eV^2 from L2, so:
    # kappa(W/(m*K)) = pi/(V*T) * (L2-L1^2/L0) * e^2 * eV_to_J / (hbar_SI * Ang_to_m)
    # where L2 in eV^2 * Ang^2, V in Ang^3, T in K
    # = pi * e^2 * eV_J * L2_eff / (hbar_SI * 1e-10 * V_Ang3 * 1e-30 * T) ...
    # Let me factor properly:
    # kappa = pi * e^2 / (hbar * V * T) * (L2 - L1^2/L0)
    # [L2] = eV^2 * Ang^2  (from integral dE * (E-mu)^2 * Xi, Xi~Ang^2, E~eV)
    # [kappa_nat] = eV^2 * Ang^2 / (Ang^3 * K) = eV^2 / (Ang * K)
    # to SI:  * e^2 (C^2) * eV_to_J (J/eV) / hbar_SI (J*s) / Ang_to_m (m/Ang)
    # kappa_SI = kappa_nat * E_CHARGE^2 * E_CHARGE / (hbar_SI * 1e-10)
    # = kappa_nat * E_CHARGE^3 / (hbar_SI * 1e-10)   in W/(m*K) ...
    # Actually e^2/hbar gives S (siemens), times eV^2/K gives S*eV^2/K
    # S * eV^2 = A/V * (eV)^2 = A * eV = A * e * V => W * e => need /e somewhere
    # Let me just do dimensional analysis more carefully.
    # sigma(S/m) = pi/V * L0 * e^2/(hbar * Ang)  where L0 = integral dE (-df/dE) Xi
    #   L0 units: eV * 1/eV * Ang^2 = Ang^2  (since -df/dE ~ 1/eV, Xi ~ Ang^2)
    # kappa = pi/V * 1/T * (L2-L1^2/L0) * e^2/(hbar * Ang)
    #   (L2-L1^2/L0) units: eV^2 * Ang^2
    # so kappa in S/m * eV^2 / K.  Convert eV^2 to J^2: * E_CHARGE^2
    # Then kappa_SI = sigma_prefactor * (L2-L1^2/L0) * E_CHARGE^2 / (K)  ... hmm
    # Actually kappa = sigma * (L2/L0 - (L1/L0)^2) / (e^2 T)  (Wiedemann-Franz-like)
    # So kappa(W/(m*K)) = sigma(S/m) * [(L2/L0 - (L1/L0)^2)] / T
    # where (L2/L0 - (L1/L0)^2) is in eV^2. Convert to J^2: * E_CHARGE^2
    # Then / (e^2 * T) ... no wait.
    # Standard: kappa_e = (1/(e^2 T)) * (L2 - L1^2/L0) * [same prefactor as sigma]
    # kappa_e(W/(m*K)) = sigma_prefactor_Sm / T * (L2-L1^2/L0)
    # but (L2-L1^2/L0) is in eV^2*Ang^2 vs L0 in Ang^2
    # sigma_prefactor = pi * e^2 / (hbar_SI * 1e-10 * V_m3)  ... no, V is in Ang^3
    # Let's just compute numerically.
    # From sigma: sigma_Sm = pi * L0 / V_Ang3 * e^2/(hbar_SI * 1e-10)
    # kappa = pi * lorenz_term / (V_Ang3 * T) * e^2/(hbar_SI * 1e-10)
    # but lorenz_term is in eV^2 * Ang^2, vs L0 in Ang^2
    # so we need an extra factor converting eV^2 -> J^2 / e^2 ...
    # eV^2 * e^2/(hbar*Ang) gives eV^2 * S/m ... not W/(m*K)
    # kappa: [W/(m*K)] = [J/(s*m*K)] = [eV * e / (s*m*K)]
    # We have: pi * (eV^2 * Ang^2) / (Ang^3 * K) * e^2/(hbar_SI * 1e-10)
    # = pi * eV^2 / (Ang * K) * e^2 / (hbar_SI * 1e-10)
    # = pi * eV^2 * e^2 / (Ang * K * hbar_SI * 1e-10)
    # 1 eV = e (in C) * 1V = E_CHARGE J
    # eV^2 * e^2 = (E_CHARGE)^2 J^2 * (E_CHARGE)^2 = E_CHARGE^4 J^2 ... no
    # eV^2 = E_CHARGE^2 V^2   (V = volts)
    # eV^2 * e^2 = E_CHARGE^2 * V^2 * E_CHARGE^2 ... wrong
    # kappa_e = pi/(V*T) * lorenz * e^2/hbar   where lorenz in eV^2 * Ang^2
    # In SI fully:
    # = pi * lorenz_eV2_Ang2 * E_CHARGE^2 * (1e-10)^2 /
    #   (V_Ang3*(1e-10)^3 * T * hbar_SI * (1e-10))  ???
    # I'm overcomplicating this. Use the relation:
    # kappa_e / sigma = (1/T) * (L2/L0 - (L1/L0)^2)
    # where L2/L0 - (L1/L0)^2 is in eV^2.
    # kappa_e(W/(m*K)) = sigma(S/m) / T * (L2/L0 - (L1/L0)^2)(eV^2) / e^2(eV^2/V^2)
    # Since 1 eV = e * 1V, eV^2/e^2 = V^2 and S*V^2/K = W*V/K ... still messy.
    # Direct: kappa = sigma * (L_ratio_eV2) * (conversion)
    # L_ratio = L2/L0 - (L1/L0)^2  in eV^2
    # In Lorentz units:  kappa/(sigma*T) = L_ratio / e^2  (if e in eV -> e=1 so just L_ratio/T)
    # kappa(W/mK) = sigma(S/m) * L_ratio(eV^2) / (T(K))
    # But [S/m * eV^2 / K] = [A/(V*m) * eV^2 / K]
    # 1 eV = E_CHARGE Joules, eV^2 = E_CHARGE^2 J^2  ? No. eV is energy, eV^2 = energy^2
    # [S * eV^2 / K] = [A/V * eV^2 / K] = [A * eV^2/(V * K)]
    # eV / V = eV / (J/C) = eV * C/J = C (since eV/J = E_CHARGE)
    # So A * eV * C / K = A * E_CHARGE * eV / K
    # eV = E_CHARGE J
    # = A * E_CHARGE^2 * J / K = C/s * E_CHARGE^2 * J / K = E_CHARGE^3 * J / (s*K) = E_CHARGE^3 * W/K
    # So kappa(W/(m*K)) = sigma(S/m)/T * L_ratio(eV^2) => units S*eV^2/(m*K)
    # Need to convert eV^2 to SI energy^2:  eV^2 = (E_CHARGE)^2 J^2
    # kappa = sigma/T * L_ratio * E_CHARGE^2   ... but  S * J^2 / (m*K) = A/V * J^2 /(m*K)
    # = A*J/(m*K) = W/(m*K) ✓  (since J/V = C and A*C = A*A*s ... no let's check:
    # S = A/V = A^2*s/J, so S*J^2/(m*K) = A^2*s*J/(m*K) = W*A^2*s^2/(m*K)...
    # OK let me just use kB^2 relation.
    # Wiedemann-Franz: kappa/(sigma T) = L (Lorenz number)
    # For free electrons L = pi^2/3 * (kB/e)^2
    # Our L_ratio = L2/L0 - (L1/L0)^2 is analogous to L * T^2 * e^2 / kB^2 ...
    #
    # SIMPLEST APPROACH: compute everything in eV and Ang, then convert at the end.
    # kappa_e / sigma_DC = (L2/L0 - (L1/L0)^2) / T   (all in eV, K)
    # This ratio is in eV/K.
    # kappa_e(W/mK) = sigma(S/m) * ratio(eV/K)
    # ratio(eV/K) -> SI: ratio * E_CHARGE (J/K)
    # kappa_e = sigma_Sm * ratio_eV_per_K * E_CHARGE   in S*J/(m*K) = W/(m*K) ✓

    if abs(L0) > 1e-30 and temperature > 0:
        ratio = lorenz_term / (L0 * temperature)  # eV/K
        kappa_e = sigma_Sm * ratio * E_CHARGE      # W/(m*K)
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
