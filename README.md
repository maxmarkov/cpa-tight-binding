
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

- `src/hamiltonian/params.py`: Si/Ge sp³s* parameters (Vogl-style) and VCA mixing.
- `src/hamiltonian/lattice.py`: diamond/FCC geometry + Monkhorst–Pack grids.
- `src/hamiltonian/slater_koster.py`: SK hopping matrices for sp³s*.
- `src/hamiltonian/bloch.py`: Bloch Hamiltonian H(k) and hopping-only H_hop(k).
- `src/cpa/solver.py`: single-site CPA solver (energy-by-energy).
- `src/cpa/greens.py`: DOS and spectral map routines.
- `src/utils/backend.py`: NumPy / PyTorch array backend abstraction.
- `src/utils/kpath.py`: high-symmetry k-path via [seekpath](https://github.com/giovannipizzi/seekpath) (HPKOT convention).

See `docs/approach.md` for the algorithm summary.

## Notes / limitations

- Disorder model is **on-site only** (no off-diagonal/bond disorder).
- Parameter set is **nearest neighbor sp³s*** (fast enough for CPA, but not “state of the art”).
- SOC hooks exist but are not enabled by default in the demo scripts.

If you need sp³d⁵s* or second neighbors, the CPA machinery stays the same, but matrices get bigger.

## References

### Tight-binding Hamiltonian

1. P. Vogl, H. P. Hjalmarson, and J. D. Dow, "A semi-empirical tight-binding theory of the electronic structure of semiconductors," *J. Phys. Chem. Solids* **44**, 365–378 (1983). — Original sp³s* parameterisation for diamond/zincblende semiconductors used in this code.

2. J. C. Slater and G. F. Koster, "Simplified LCAO method for the periodic potential problem," *Phys. Rev.* **94**, 1498–1524 (1954). — Two-center integral formalism underlying the hopping-matrix construction.

3. W. A. Harrison, *Electronic Structure and the Properties of Solids* (Freeman, San Francisco, 1980). — Bond-length scaling rules ($d^{-n}$) used for the VCA interpolation of hopping parameters.

### Coherent Potential Approximation

4. P. Soven, "Coherent-potential model of substitutional disordered alloys," *Phys. Rev.* **156**, 809–813 (1967). — Foundational single-site CPA formulation for substitutional alloys.

5. B. Velický, S. Kirkpatrick, and H. Ehrenreich, "Single-site approximations in the electronic theory of simple binary alloys," *Phys. Rev.* **175**, 747–766 (1968). — Systematic development of single-site CPA and its relation to the average T-matrix approximation.

6. R. J. Elliott, J. A. Krumhansl, and P. L. Leath, "The theory and properties of randomly disordered crystals and related physical systems," *Rev. Mod. Phys.* **46**, 465–543 (1974). — Comprehensive review of disorder methods including CPA; useful for broader context.

7. D. W. Taylor, "Vibrational properties of imperfect crystals with large defect concentrations," *Phys. Rev.* **156**, 1017–1029 (1967). — Independent early derivation of the CPA self-consistency condition.
