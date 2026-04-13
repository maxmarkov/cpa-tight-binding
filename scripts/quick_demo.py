
#!/usr/bin/env python3
"""
Quick demo: compute VCA band structure (lines), CPA spectral map and DOS
for Si1-xGex in a compact sp3s* model.

Examples:
  python scripts/quick_demo.py --x 0.50 --out outputs_demo
  python scripts/quick_demo.py --x 0.0 0.25 0.5 0.75 1.0 --out outputs_demo
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Allow importing src when run as script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt

from src.backend import set_backend, to_numpy
from src import backend as be
from src.hamiltonian import (
    SI_VOGL, GE_VOGL, mix_params_vca, disorder_onsites, onsite_matrix,
    reciprocal_vectors, frac_to_cart_k, monkhorst_pack,
    bloch_hamiltonian_sp3s_star, hopping_only_matrix,
)
from src.kpath import make_kpath
from src.cpa import cpa_solve_grid, embed_onsite_in_cell
from src.greens import dos_from_eigs, dos_from_gloc, spectral_map_kpath

def compute_vca_bands(k_frac_path, params, include_soc: bool=False):
    bands = be.stack([be.real(be.eigvalsh(bloch_hamiltonian_sp3s_star(kf, params, include_soc=include_soc))) for kf in k_frac_path])
    return bands

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="Use PyTorch GPU backend")
    ap.add_argument("--x", nargs="+", type=float, default=[0.5], help="Ge fraction(s)")
    ap.add_argument("--out", type=str, default="outputs_demo", help="Output directory")
    ap.add_argument("--nk_dos", type=int, default=4, help="MP grid size for DOS (demo default)")
    ap.add_argument("--nk_cpa", type=int, default=3, help="MP grid size for CPA self-consistency (demo default)")
    ap.add_argument("--ne", type=int, default=320, help="Number of energy points")
    ap.add_argument("--eta", type=float, default=0.08, help="Small imaginary broadening (eV)")
    ap.add_argument("--cpa_tol", type=float, default=3e-6)
    ap.add_argument("--cpa_mix", type=float, default=0.7)
    ap.add_argument("--cpa_max_iter", type=int, default=180)
    args = ap.parse_args()
    set_backend(args.gpu)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    for x in args.x:
        params = mix_params_vca(x, SI_VOGL, GE_VOGL, include_soc=False)

        # k path (lattice-constant-aware via seekpath)
        k_frac_path, labels, s = make_kpath(a=params.a, n_per_segment=60)
        k_frac_path = be.asarray(k_frac_path)
        s = be.asarray(s)

        # VCA bands
        bands = compute_vca_bands(k_frac_path, params, include_soc=False)

        # energy window
        emin = float(be.amin(bands) - 1.0)
        emax = float(be.amax(bands) + 1.0)
        energies = be.linspace(emin, emax, args.ne)

        # DOS VCA (from eigenvalues on MP grid)
        kgrid = monkhorst_pack(args.nk_dos, args.nk_dos, args.nk_dos)
        eigs = be.stack([be.real(be.eigvalsh(bloch_hamiltonian_sp3s_star(kf, params, include_soc=False))) for kf in kgrid])
        dos_vca = dos_from_eigs(eigs, energies, args.eta)

        # CPA precompute hopping-only for CPA kgrid and for kpath
        kgrid_cpa = monkhorst_pack(args.nk_cpa, args.nk_cpa, args.nk_cpa)
        Hhop_k = be.stack([hopping_only_matrix(kf, params, include_soc=False) for kf in kgrid_cpa])
        Hhop_path = be.stack([hopping_only_matrix(kf, params, include_soc=False) for kf in k_frac_path])

        VA, VB = disorder_onsites(SI_VOGL, GE_VOGL, include_soc=False)

        cpa = cpa_solve_grid(
            energies=energies, eta=args.eta,
            Hhop_k=Hhop_k, V_A_atom=VA, V_B_atom=VB, x=x,
            mix=args.cpa_mix, tol=args.cpa_tol, max_iter=args.cpa_max_iter,
            continuation=True
        )
        sigma_atom = cpa["sigma_atom"]
        gloc_atom = cpa["gloc_atom"]
        dos_cpa = dos_from_gloc(gloc_atom)

        # CPA spectral map along kpath
        onsite_vca_atom = be.asarray(onsite_matrix(params, True), dtype=complex)
        onsite_vca_cell = embed_onsite_in_cell(onsite_vca_atom, 2)
        A_cpa = spectral_map_kpath(energies, args.eta, Hhop_path, onsite_vca_cell, sigma_atom_E=sigma_atom)

        # convert to numpy for plotting
        bands = to_numpy(bands)
        s = to_numpy(s)
        energies = to_numpy(energies)
        dos_vca = to_numpy(dos_vca)
        dos_cpa = to_numpy(dos_cpa)
        A_cpa = to_numpy(A_cpa)

        # plot: combined figure
        fig = plt.figure(figsize=(11,4.2))
        gs = fig.add_gridspec(1,3, width_ratios=[1.6,1.0,1.4], wspace=0.25)

        ax0 = fig.add_subplot(gs[0,0])
        # VCA bands
        for b in range(bands.shape[1]):
            ax0.plot(s, bands[:,b], lw=0.9)
        ax0.set_title(f"VCA bands (x={x:.2f})")
        ax0.set_ylabel("Energy (eV)")
        ax0.set_xlabel("k-path")
        # ticks
        xt = [i for i,_ in labels]
        xl = [lab for _,lab in labels]
        ax0.set_xticks([s[i] for i in xt], xl)
        ax0.grid(True, alpha=0.25)

        ax1 = fig.add_subplot(gs[0,1])
        ax1.plot(dos_vca, energies, label="VCA")
        ax1.plot(dos_cpa, energies, label="CPA")
        ax1.set_xlabel("DOS (arb.)")
        ax1.set_title("DOS")
        ax1.grid(True, alpha=0.25)
        ax1.legend(fontsize=8, loc="best")

        ax2 = fig.add_subplot(gs[0,2])
        im = ax2.imshow(
            A_cpa, origin="lower", aspect="auto",
            extent=[s[0], s[-1], energies[0], energies[-1]]
        )
        ax2.set_title("CPA spectral intensity A(k,E)")
        ax2.set_xlabel("k-path")
        ax2.set_ylabel("Energy (eV)")
        ax2.set_xticks([s[i] for i in xt], xl)
        fig.colorbar(im, ax=ax2, fraction=0.045, pad=0.02)

        fig.suptitle("Si$_{1-x}$Ge$_x$ sp$^3$s$^*$ tight-binding: VCA vs CPA", y=1.02, fontsize=11)
        fig.tight_layout()

        outpng = outdir / f"demo_SiGe_x{x:.2f}.png"
        fig.savefig(outpng, dpi=180, bbox_inches="tight")
        plt.close(fig)

        # also save CPA self-energy imaginary part (trace)
        tr_sigma = be.trace(sigma_atom, axis1=1, axis2=2)
        tr_sigma = to_numpy(tr_sigma)
        fig2 = plt.figure(figsize=(6.2,3.2))
        ax = fig2.add_subplot(111)
        ax.plot(energies, -tr_sigma.imag)
        ax.set_title(f"CPA scattering (-Im Tr Σ), x={x:.2f}")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("-Im Tr Σ (eV)")
        ax.grid(True, alpha=0.25)
        fig2.tight_layout()
        fig2.savefig(outdir / f"sigma_im_x{x:.2f}.png", dpi=180, bbox_inches="tight")
        plt.close(fig2)

        print(f"Wrote: {outpng}")

if __name__ == "__main__":
    main()
