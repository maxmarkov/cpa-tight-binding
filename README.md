
# Coherent Potential Approximation for Si/Ge alloys

A tight-binding toolkit for computing electronic properties of Si₁₋ₓGeₓ alloys using the **Virtual Crystal Approximation (VCA)** and **Coherent Potential Approximation (CPA)**.

VCA constructs an effective periodic crystal by linearly interpolating on-site energies and hopping parameters between the pure endpoints, with optional bond-length scaling. CPA goes further by treating substitutional disorder explicitly: it embeds a single impurity site in a self-consistently determined effective medium so that the average scattering vanishes, capturing alloy-induced band broadening and lifetime effects that VCA misses.

The underlying Hamiltonian is a nearest-neighbor **orthogonal sp³s*** model for the **diamond** lattice (Si, Ge), parameterised in the Vogl/Hjalmarson/Dow style.

The code produces:

- **Band structure** along high-symmetry paths (VCA)
- **Spectral function** $A(\mathbf{k}, E)$ showing disorder-broadened bands (CPA)
- **Density of states** comparison (VCA vs CPA)
- **Self-energy diagnostics** $\Sigma(E)$ for inspecting convergence and scattering rates

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

- `tb_cpa/params.py`: Si/Ge sp³s* parameters (Vogl-style) and VCA mixing.
- `tb_cpa/lattice.py`: diamond/FCC geometry + Monkhorst–Pack grids.
- `tb_cpa/slater_koster.py`: SK hopping matrices for sp³s*.
- `tb_cpa/hamiltonian.py`: Bloch Hamiltonian H(k) and hopping-only H_hop(k).
- `tb_cpa/cpa.py`: single-site CPA solver (energy-by-energy).
- `tb_cpa/greens.py`: DOS and spectral map routines.
- `tb_cpa/kpath.py`: Γ–X–W–K–Γ–L k-path.

See `docs/approach.md` for the algorithm summary.

## Notes / limitations

- Disorder model is **on-site only** (no off-diagonal/bond disorder).
- Parameter set is **nearest neighbor sp³s*** (fast enough for CPA, but not “state of the art”).
- SOC hooks exist but are not enabled by default in the demo scripts.

If you need sp³d⁵s* or second neighbors, the CPA machinery stays the same, but matrices get bigger.
