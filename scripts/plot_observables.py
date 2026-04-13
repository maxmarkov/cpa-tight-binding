
#!/usr/bin/env python3
"""
Compute and plot all CPA observables for Si1-xGex:

  1. Orbital-resolved (partial) DOS
  2. Quasiparticle scattering rates
  3. Effective band structure (peak positions + widths)
  4. Band gap vs composition (with bowing fit)
  5. Optical conductivity & dielectric function
  6. DC resistivity vs composition
  7. Thermoelectric coefficients vs composition

Usage:
  python scripts/plot_observables.py --x 0.5 --out outputs_obs
  python scripts/plot_observables.py --observable all --x 0.5 --out outputs_obs
  python scripts/plot_observables.py --observable gap --out outputs_obs
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.utils.backend import set_backend, to_numpy
from src.utils import backend as be
from src.hamiltonian import (
    SI_VOGL, GE_VOGL, mix_params_vca, disorder_onsites, onsite_matrix,
    monkhorst_pack, bloch_hamiltonian_sp3s_star, hopping_only_matrix,
    velocity_matrix_sp3s_star,
)
from src.utils.kpath import make_kpath
from src.cpa import (
    cpa_solve_grid, embed_onsite_in_cell,
    dos_from_eigs, dos_from_gloc, spectral_map_kpath,
    pdos_from_gloc, scattering_rates, effective_bands,
    transport_distribution, transport_integrals,
    optical_conductivity, dc_conductivity, thermoelectric_coefficients,
    diamond_cell_volume, find_spectral_gap, fit_bowing,
)


# ---------------------------------------------------------------------------
#  Shared CPA pipeline
# ---------------------------------------------------------------------------

def run_cpa_pipeline(x, args):
    """Run the CPA solver for composition x.  Returns a results dict."""
    params = mix_params_vca(x, SI_VOGL, GE_VOGL, include_soc=False)
    kgrid = monkhorst_pack(args.nk, args.nk, args.nk)
    Hhop_k = be.stack([hopping_only_matrix(kf, params) for kf in kgrid])
    VA, VB = disorder_onsites(SI_VOGL, GE_VOGL)

    # energy grid
    eigs_sample = be.stack([be.real(be.eigvalsh(
        bloch_hamiltonian_sp3s_star(kf, params))) for kf in kgrid])
    emin = float(be.amin(eigs_sample) - 1.5)
    emax = float(be.amax(eigs_sample) + 1.5)
    energies = be.linspace(emin, emax, args.ne)

    cpa = cpa_solve_grid(
        energies=energies, eta=args.eta,
        Hhop_k=Hhop_k, V_A_atom=VA, V_B_atom=VB, x=x,
        mix=args.cpa_mix, tol=args.cpa_tol, max_iter=args.cpa_max_iter,
        continuation=True,
    )
    dos_vca = dos_from_eigs(eigs_sample, energies, args.eta)
    dos_cpa = dos_from_gloc(cpa['gloc_atom'])

    return {
        'params': params, 'kgrid': kgrid, 'Hhop_k': Hhop_k,
        'energies': energies, 'sigma_atom': cpa['sigma_atom'],
        'gloc_atom': cpa['gloc_atom'], 'dos_vca': dos_vca, 'dos_cpa': dos_cpa,
        'VA': VA, 'VB': VB,
    }


# ---------------------------------------------------------------------------
#  1.  Partial DOS
# ---------------------------------------------------------------------------

def plot_pdos(x, res, outdir):
    pdos = pdos_from_gloc(res['gloc_atom'])
    E = to_numpy(res['energies'])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(E, to_numpy(pdos['s']),     label='s')
    ax.plot(E, to_numpy(pdos['p']),     label='p')
    ax.plot(E, to_numpy(pdos['sstar']), label='s*')
    ax.plot(E, to_numpy(pdos['total']), 'k--', lw=1.2, label='total')
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Partial DOS (arb.)')
    ax.set_title(f'Orbital-resolved CPA DOS, x={x:.2f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / f'pdos_x{x:.2f}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  pdos_x{x:.2f}.png')


# ---------------------------------------------------------------------------
#  2.  Scattering rates
# ---------------------------------------------------------------------------

def plot_scattering(x, res, outdir):
    rates = scattering_rates(res['sigma_atom'])
    E = to_numpy(res['energies'])

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(E, to_numpy(rates['s']),     label='s')
    ax.plot(E, to_numpy(rates['p']),     label='p')
    ax.plot(E, to_numpy(rates['sstar']), label='s*')
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel(r'$\tau^{-1}$ (eV)')
    ax.set_title(f'CPA scattering rates, x={x:.2f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / f'scattering_x{x:.2f}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  scattering_x{x:.2f}.png')


# ---------------------------------------------------------------------------
#  3.  Effective band structure
# ---------------------------------------------------------------------------

def plot_effective_bands(x, res, args, outdir):
    params = res['params']
    k_frac_path, labels, s_path = make_kpath(a=params.a, n_per_segment=80)
    k_frac_path = be.asarray(k_frac_path)
    Hhop_path = be.stack([hopping_only_matrix(kf, params) for kf in k_frac_path])
    onsite_cell = embed_onsite_in_cell(
        be.asarray(onsite_matrix(params, True), dtype=complex), 2)
    A_cpa = spectral_map_kpath(res['energies'], args.eta, Hhop_path,
                               onsite_cell, sigma_atom_E=res['sigma_atom'])

    bands = effective_bands(A_cpa, res['energies'])
    E = to_numpy(res['energies'])
    A_np = to_numpy(A_cpa)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4.5),
                                   gridspec_kw={'width_ratios': [1.4, 1]})

    # spectral map
    im = ax0.imshow(A_np, origin='lower', aspect='auto',
                    extent=[s_path[0], s_path[-1], E[0], E[-1]])
    # overlay effective band positions
    for ik, pk_list in enumerate(bands):
        for (Ep, fwhm) in pk_list:
            ax0.plot(s_path[ik], Ep, 'r.', ms=1.0)
    xt = [s_path[i] for i, _ in labels]
    xl = [lab for _, lab in labels]
    ax0.set_xticks(xt, xl)
    ax0.set_ylabel('Energy (eV)')
    ax0.set_title('CPA spectral function + effective bands')
    fig.colorbar(im, ax=ax0, fraction=0.04, pad=0.02)

    # FWHM at selected k-points
    mid = len(bands) // 2
    fwhms = [fw for (_, fw) in bands[mid] if not np.isnan(fw)]
    epeaks = [ep for (ep, fw) in bands[mid] if not np.isnan(fw)]
    if fwhms:
        ax1.barh(epeaks, fwhms, height=(E[-1] - E[0]) / len(E) * 3, color='C1')
        ax1.set_xlabel('FWHM (eV)')
        ax1.set_ylabel('Energy (eV)')
        ax1.set_title(f'Peak widths at mid-path k')
    ax1.grid(True, alpha=0.25)
    fig.suptitle(f'Effective bands, x={x:.2f}', y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / f'eff_bands_x{x:.2f}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  eff_bands_x{x:.2f}.png')


# ---------------------------------------------------------------------------
#  4.  Band gap vs composition
# ---------------------------------------------------------------------------

def plot_gap_vs_x(args, outdir):
    x_vals = np.linspace(0.0, 1.0, 11)
    gaps = []
    for x in x_vals:
        print(f'  gap sweep x={x:.2f} ...')
        res = run_cpa_pipeline(float(x), args)
        info = find_spectral_gap(res['dos_cpa'], res['energies'])
        gaps.append(info['gap'] if info else float('nan'))

    gaps = np.array(gaps)
    valid = ~np.isnan(gaps)
    b = fit_bowing(x_vals[valid], gaps[valid]) if valid.sum() >= 3 else 0.0

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x_vals[valid], gaps[valid], 'ko-', label='CPA gap')
    xf = np.linspace(0, 1, 100)
    g0, g1 = gaps[0] if valid[0] else 0, gaps[-1] if valid[-1] else 0
    ax.plot(xf, (1 - xf) * g0 + xf * g1, 'b--', alpha=0.5, label='linear (VCA)')
    ax.plot(xf, (1 - xf) * g0 + xf * g1 - b * xf * (1 - xf),
            'r-', alpha=0.7, label=f'bowing b={b:.3f} eV')
    ax.set_xlabel('Ge fraction x')
    ax.set_ylabel('Band gap (eV)')
    ax.set_title('CPA band gap vs composition')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / 'gap_vs_x.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('  gap_vs_x.png')
    return b


# ---------------------------------------------------------------------------
#  5.  Optical conductivity & dielectric function
# ---------------------------------------------------------------------------

def plot_optical(x, res, args, outdir):
    params = res['params']
    Vcell = diamond_cell_volume(params.a)
    tdf = transport_distribution(
        res['energies'], args.eta, res['Hhop_k'], res['kgrid'],
        params, res['sigma_atom'], alpha=0)

    omega = be.linspace(0.05, 8.0, 200)
    sigma_1, eps_2 = optical_conductivity(
        res['energies'], tdf, omega, mu=0.0, temperature=300.0,
        volume_cell=Vcell)

    omega_np = to_numpy(omega)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 4))

    ax0.plot(omega_np, to_numpy(sigma_1))
    ax0.set_xlabel(r'$\hbar\omega$ (eV)')
    ax0.set_ylabel(r'$\sigma_1$ ($\Omega^{-1}$cm$^{-1}$)')
    ax0.set_title(f'Optical conductivity, x={x:.2f}')
    ax0.grid(True, alpha=0.25)

    ax1.plot(omega_np, to_numpy(eps_2))
    ax1.set_xlabel(r'$\hbar\omega$ (eV)')
    ax1.set_ylabel(r'$\varepsilon_2$')
    ax1.set_title(f'Dielectric function (imag), x={x:.2f}')
    ax1.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(outdir / f'optical_x{x:.2f}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  optical_x{x:.2f}.png')


# ---------------------------------------------------------------------------
#  6.  DC resistivity vs composition
# ---------------------------------------------------------------------------

def plot_dc_resistivity(args, outdir):
    x_vals = np.linspace(0.0, 1.0, 11)
    rho_vals = []
    for x in x_vals:
        print(f'  DC sweep x={x:.2f} ...')
        res = run_cpa_pipeline(float(x), args)
        params = res['params']
        Vcell = diamond_cell_volume(params.a)
        tdf = transport_distribution(
            res['energies'], args.eta, res['Hhop_k'], res['kgrid'],
            params, res['sigma_atom'], alpha=0)
        _, rho = dc_conductivity(res['energies'], tdf, mu=0.0,
                                 temperature=300.0, volume_cell=Vcell)
        rho_vals.append(rho)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.semilogy(x_vals, rho_vals, 'ko-')
    ax.set_xlabel('Ge fraction x')
    ax.set_ylabel(r'$\rho$ ($\Omega\cdot$cm)')
    ax.set_title('DC resistivity vs composition (T=300 K)')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / 'dc_rho_vs_x.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('  dc_rho_vs_x.png')


# ---------------------------------------------------------------------------
#  7.  Thermoelectric coefficients
# ---------------------------------------------------------------------------

def plot_thermoelectric(x, res, args, outdir):
    params = res['params']
    Vcell = diamond_cell_volume(params.a)
    tdf = transport_distribution(
        res['energies'], args.eta, res['Hhop_k'], res['kgrid'],
        params, res['sigma_atom'], alpha=0)

    temps = np.linspace(100, 800, 15)
    seebeck_vals = []
    kappa_vals = []
    sigma_vals = []
    for T in temps:
        tc = thermoelectric_coefficients(
            res['energies'], tdf, mu=0.0, temperature=float(T),
            volume_cell=Vcell)
        sigma_vals.append(tc['sigma'])
        seebeck_vals.append(tc['seebeck'] * 1e6)   # to microV/K
        kappa_vals.append(tc['kappa_e'])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].plot(temps, sigma_vals, 'o-')
    axes[0].set_xlabel('T (K)')
    axes[0].set_ylabel(r'$\sigma$ ($\Omega^{-1}$cm$^{-1}$)')
    axes[0].set_title('Conductivity')
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(temps, seebeck_vals, 'o-')
    axes[1].set_xlabel('T (K)')
    axes[1].set_ylabel(r'S ($\mu$V/K)')
    axes[1].set_title('Seebeck coefficient')
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(temps, kappa_vals, 'o-')
    axes[2].set_xlabel('T (K)')
    axes[2].set_ylabel(r'$\kappa_e$ (W/mK)')
    axes[2].set_title('Electronic thermal cond.')
    axes[2].grid(True, alpha=0.25)

    fig.suptitle(f'Thermoelectric coefficients, x={x:.2f}', y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / f'thermo_x{x:.2f}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'  thermo_x{x:.2f}.png')


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

OBSERVABLES = {
    'pdos': 'Orbital-resolved partial DOS',
    'scattering': 'Quasiparticle scattering rates',
    'bands': 'Effective band structure',
    'gap': 'Band gap vs composition',
    'optical': 'Optical conductivity & dielectric function',
    'dc': 'DC resistivity vs composition',
    'thermo': 'Thermoelectric coefficients',
}


def main():
    ap = argparse.ArgumentParser(
        description='Compute and plot CPA observables for Si1-xGex.')
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--observable', nargs='+',
                    default=['all'],
                    choices=list(OBSERVABLES) + ['all'],
                    help='Which observables to compute')
    ap.add_argument('--x', type=float, default=0.5, help='Ge fraction')
    ap.add_argument('--out', type=str, default='outputs_obs')
    ap.add_argument('--nk', type=int, default=4, help='MP grid size')
    ap.add_argument('--ne', type=int, default=400, help='Energy points')
    ap.add_argument('--eta', type=float, default=0.06)
    ap.add_argument('--cpa_tol', type=float, default=1e-6)
    ap.add_argument('--cpa_mix', type=float, default=0.6)
    ap.add_argument('--cpa_max_iter', type=int, default=200)
    args = ap.parse_args()
    set_backend(args.gpu)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    obs = set(args.observable)
    if 'all' in obs:
        obs = set(OBSERVABLES)

    x = args.x

    # single-composition observables need one CPA run
    need_single = obs & {'pdos', 'scattering', 'bands', 'optical', 'thermo'}
    res = None
    if need_single:
        print(f'Running CPA for x={x:.2f} ...')
        res = run_cpa_pipeline(x, args)

    if 'pdos' in obs:
        print('Computing partial DOS ...')
        plot_pdos(x, res, outdir)

    if 'scattering' in obs:
        print('Computing scattering rates ...')
        plot_scattering(x, res, outdir)

    if 'bands' in obs:
        print('Computing effective bands ...')
        plot_effective_bands(x, res, args, outdir)

    if 'gap' in obs:
        print('Computing band gap vs composition ...')
        plot_gap_vs_x(args, outdir)

    if 'optical' in obs:
        print('Computing optical conductivity ...')
        plot_optical(x, res, args, outdir)

    if 'dc' in obs:
        print('Computing DC resistivity vs composition ...')
        plot_dc_resistivity(args, outdir)

    if 'thermo' in obs:
        print('Computing thermoelectric coefficients ...')
        plot_thermoelectric(x, res, args, outdir)

    print('Done.')


if __name__ == '__main__':
    main()
