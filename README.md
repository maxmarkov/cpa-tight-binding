
# Virtual Crystal Approximation (VCA) and Coherent Potential Approximation for alloys

This package implements:

- A nearest-neighbor **orthogonal sp³s\*** tight-binding Hamiltonian for the **diamond** structure (Si, Ge).
- **VCA (Virtual Crystal Approximation)**: linear composition mixing (with optional bond-length scaling).
- **Single-site CPA (Coherent Potential Approximation)** for **diagonal (on-site) substitutional disorder**.
- Plots: **band structure** (VCA), **CPA spectral intensity** A(k,E), **DOS** (VCA vs CPA), and a simple self-energy diagnostic.

It is meant as a *simple, readable reference implementation*.

## Quick start

```bash
# from this folder
python -m tests.test_basic
python scripts/quick_demo.py --x 0.50 --out outputs_demo
```

Generate several compositions:

```bash
python scripts/quick_demo.py --x 0.00 0.25 0.50 0.75 1.00 --out outputs_demo
```

Higher-resolution run:

```bash
python scripts/run_all.py --out outputs --nk 10 --ne 800 --eta 0.04 --cpa_tol 1e-7 --cpa_max_iter 300
```

Outputs are PNG figures written to the chosen output directory.

## What’s inside

- `tb_cpa/params.py`: Si/Ge sp³s\* parameters (Vogl-style) and VCA mixing.
- `tb_cpa/lattice.py`: diamond/FCC geometry + Monkhorst–Pack grids.
- `tb_cpa/slater_koster.py`: SK hopping matrices for sp³s\*.
- `tb_cpa/hamiltonian.py`: Bloch Hamiltonian H(k) and hopping-only H_hop(k).
- `tb_cpa/cpa.py`: single-site CPA solver (energy-by-energy).
- `tb_cpa/greens.py`: DOS and spectral map routines.
- `tb_cpa/kpath.py`: Γ–X–W–K–Γ–L k-path.

See `docs/approach.md` for the algorithm summary.

## Notes / limitations

- Disorder model is **on-site only** (no off-diagonal/bond disorder).
- Parameter set is **nearest neighbor sp³s\*** (fast enough for CPA, but not “state of the art”).
- SOC hooks exist but are not enabled by default in the demo scripts.

If you need sp³d⁵s\* or second neighbors, the CPA machinery stays the same, but matrices get bigger.VV
