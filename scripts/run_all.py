
#!/usr/bin/env python3
"""
Heavier runner with tunable convergence for generating publication-quality plots.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="Use PyTorch GPU backend")
    ap.add_argument("--x", nargs="+", type=float, default=[0.0,0.25,0.5,0.75,1.0])
    ap.add_argument("--out", type=str, default="outputs")
    ap.add_argument("--nk", type=int, default=10, help="MP grid size for DOS and CPA")
    ap.add_argument("--ne", type=int, default=800)
    ap.add_argument("--eta", type=float, default=0.04)
    ap.add_argument("--cpa_tol", type=float, default=1e-7)
    ap.add_argument("--cpa_mix", type=float, default=0.5)
    ap.add_argument("--cpa_max_iter", type=int, default=300)
    args = ap.parse_args()
    set_backend(args.gpu)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    VA, VB = disorder_onsites(SI_VOGL, GE_VOGL, include_soc=False)

    for x in args.x:
        params = mix_params_vca(x, SI_VOGL, GE_VOGL, include_soc=False)

        # k path (lattice-constant-aware via seekpath)
        k_frac_path, labels, s = make_kpath(a=params.a, n_per_segment=120)
        k_frac_path = be.asarray(k_frac_path)
        s = be.asarray(s)

        # VCA bands
        bands = be.stack([be.real(be.eigvalsh(bloch_hamiltonian_sp3s_star(kf, params, include_soc=False))) for kf in k_frac_path])

        emin = float(be.amin(bands) - 1.5)
        emax = float(be.amax(bands) + 1.5)
        energies = be.linspace(emin, emax, args.ne)

        # VCA DOS
        kgrid = monkhorst_pack(args.nk, args.nk, args.nk)
        eigs = be.stack([be.real(be.eigvalsh(bloch_hamiltonian_sp3s_star(kf, params))) for kf in kgrid])
        dos_vca = dos_from_eigs(eigs, energies, args.eta)

        # CPA prep
        Hhop_k = be.stack([hopping_only_matrix(kf, params) for kf in kgrid])
        Hhop_path = be.stack([hopping_only_matrix(kf, params) for kf in k_frac_path])
        cpa = cpa_solve_grid(energies, args.eta, Hhop_k, VA, VB, x, mix=args.cpa_mix, tol=args.cpa_tol, max_iter=args.cpa_max_iter)
        dos_cpa = dos_from_gloc(cpa["gloc_atom"])

        onsite_vca_atom = be.asarray(onsite_matrix(params, True), dtype=complex)
        onsite_vca_cell = embed_onsite_in_cell(onsite_vca_atom, 2)
        A_cpa = spectral_map_kpath(energies, args.eta, Hhop_path, onsite_vca_cell, sigma_atom_E=cpa["sigma_atom"])

        # convert to numpy for plotting
        bands = to_numpy(bands)
        s = to_numpy(s)
        energies = to_numpy(energies)
        dos_vca = to_numpy(dos_vca)
        dos_cpa = to_numpy(dos_cpa)
        A_cpa = to_numpy(A_cpa)

        # plot
        fig = plt.figure(figsize=(12,4.6))
        gs = fig.add_gridspec(1,3, width_ratios=[1.7,1.0,1.5], wspace=0.25)
        ax0 = fig.add_subplot(gs[0,0])
        for b in range(bands.shape[1]):
            ax0.plot(s, bands[:,b], lw=0.8)
        xt = [i for i,_ in labels]
        xl = [lab for _,lab in labels]
        ax0.set_xticks([s[i] for i in xt], xl)
        ax0.set_title("VCA bands")
        ax0.set_ylabel("Energy (eV)")
        ax0.grid(True, alpha=0.25)

        ax1 = fig.add_subplot(gs[0,1])
        ax1.plot(dos_vca, energies, label="VCA")
        ax1.plot(dos_cpa, energies, label="CPA")
        ax1.set_title("DOS")
        ax1.set_xlabel("DOS (arb.)")
        ax1.grid(True, alpha=0.25)
        ax1.legend(fontsize=8)

        ax2 = fig.add_subplot(gs[0,2])
        im = ax2.imshow(A_cpa, origin="lower", aspect="auto",
                        extent=[s[0], s[-1], energies[0], energies[-1]])
        ax2.set_xticks([s[i] for i in xt], xl)
        ax2.set_title("CPA A(k,E)")
        ax2.set_xlabel("k-path")
        ax2.set_ylabel("Energy (eV)")
        fig.colorbar(im, ax=ax2, fraction=0.045, pad=0.02)
        fig.suptitle(f"Si$_{{1-x}}$Ge$_x$ (x={x:.2f}) VCA vs CPA", y=1.02)
        fig.tight_layout()
        fig.savefig(outdir / f"SiGe_x{x:.2f}_vca_cpa.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        print(f"Done x={x:.2f}")

if __name__ == "__main__":
    main()
