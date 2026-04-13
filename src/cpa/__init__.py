"""
CPA solver and Green's function observables.

Submodules:
  solver  – single-site CPA self-consistency
  greens  – DOS and spectral function routines
"""
from .solver import embed_onsite_in_cell, cpa_solve_energy, cpa_solve_grid
from .greens import dos_from_eigs, dos_from_gloc, spectral_function_k, spectral_map_kpath
