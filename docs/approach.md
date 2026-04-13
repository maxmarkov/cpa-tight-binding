
# Approach and documentation (VCA + CPA for Si/Ge tight-binding)

## Model

We use a diamond lattice (FCC Bravais + two-atom basis). Each atom has an orthogonal sp³s\* basis:

\[
|s\rangle,\ |p_x\rangle,\ |p_y\rangle,\ |p_z\rangle,\ |s^\*\rangle
\]

The primitive cell has two atoms, so the Hamiltonian is 10×10 (spinless).

Nearest-neighbor (tetrahedral) hopping uses Slater–Koster two-center integrals:

- \(V_{ss\sigma}\)
- \(V_{sp\sigma}\)
- \(V_{pp\sigma}\)
- \(V_{pp\pi}\)
- \(V_{s^\*p\sigma}\)

On-site energies are \(E_s, E_p, E_{s^\*}\).

Defaults are a classic sp³s\* parameterization often attributed to the Vogl/Hjalmarson/Dow style (1980s).

## VCA (Virtual Crystal Approximation)

For Ge fraction \(x\), VCA makes a periodic crystal with composition-averaged parameters:

- On-site energies: linear interpolation
- Hoppings: linear interpolation, optionally scaled by bond-length (Harrison-like \(d^{-n}\), default \(n=2\))
- Lattice constant: Vegard’s law (linear in \(x\))

VCA yields sharp Bloch bands \(E_n(\mathbf{k})\).

## CPA (single-site, diagonal disorder)

We model substitutional disorder as random on-site matrices:

- species A = Si with onsite \(V_A\)
- species B = Ge with onsite \(V_B\)
- probability: \(1-x\) and \(x\)

The Hamiltonian is written as:

\[
H(\mathbf{k}) = H_{\mathrm{hop}}(\mathbf{k}) + V_{\mathrm{site}}
\]

CPA replaces the random alloy by an **effective medium** with a complex coherent potential (self-energy)
\(\Sigma(E)\) such that the *average single-site scattering vanishes*.

### Practical self-consistency used here

For each complex energy \(z = E + i\eta\):

1. Medium Green’s function:

\[
G(\mathbf{k},z) = \left[zI - H_{\mathrm{hop}}(\mathbf{k}) - \Sigma_{\mathrm{cell}}(z)\right]^{-1}
\]

and the local Green’s function is the BZ average:

\[
G_{\mathrm{loc}}(z) = \langle G(\mathbf{k},z)\rangle_{\mathbf{k}}
\]

2. Define the cavity (“Weiss”) Green’s function:

\[
\mathcal{G}^{-1}(z) = G_{\mathrm{loc}}^{-1}(z) + \Sigma(z)
\]

3. Solve two “impurity” Green’s functions:

\[
G_A = [\mathcal{G}^{-1} - V_A]^{-1},\quad
G_B = [\mathcal{G}^{-1} - V_B]^{-1}
\]

4. Enforce \(G_{\mathrm{loc}} = (1-x)G_A + xG_B\) by updating:

\[
\Sigma_{\mathrm{new}} = \mathcal{G}^{-1} - \left((1-x)G_A + xG_B\right)^{-1}
\]

and use linear mixing \(\Sigma \leftarrow (1-\alpha)\Sigma + \alpha\Sigma_{\mathrm{new}}\).

This is implemented in `tb_cpa/cpa.py`.

## Observables

### DOS

- VCA DOS is computed from eigenvalues on a Monkhorst–Pack grid with Lorentzian broadening.
- CPA DOS is computed from the local Green’s function:

\[
D(E) = -\frac{2}{\pi}\,\mathrm{Im}\,\mathrm{Tr}\,G_{\mathrm{loc}}(E+i\eta)
\]

The factor 2 accounts for two atoms per cell (spinless).

### Spectral function \(A(\mathbf{k},E)\)

CPA produces an energy-dependent complex self-energy, so “bands” broaden into finite-width features.

We compute:

\[
A(\mathbf{k},E) = -\frac{1}{\pi}\,\mathrm{Im}\,\mathrm{Tr}\,G(\mathbf{k},E+i\eta)
\]

along the k-path Γ–X–W–K–Γ–L.

## Running the demo

- `scripts/quick_demo.py`: fast settings intended to finish quickly and produce the basic plots.
- `scripts/run_all.py`: higher resolution.

Typical knobs:

- `--nk_dos`, `--nk_cpa` (grid sizes): larger = smoother DOS and more stable CPA
- `--eta`: smaller = sharper spectra, but CPA may converge more slowly
- `--cpa_tol`, `--cpa_mix`, `--cpa_max_iter`: CPA iteration controls

## Limitations and extensions

- The disorder model is **on-site only** (no bond disorder).
- Nearest-neighbor sp³s\* is minimal; more quantitative work often uses sp³d⁵s\* and/or second neighbors.
- Single-site CPA ignores short-range order and localization; cluster CPA can address some effects.

Still, the code is a solid template: replace the TB basis/parameters and keep the CPA machinery.
