"""
CPA solver, Green's function observables, and transport properties.

Submodules:
  solver       – single-site CPA self-consistency
  greens       – DOS, spectral function, partial DOS, scattering rates, effective bands
  transport    – optical/DC conductivity, thermoelectric coefficients
  observables  – band-gap extraction, composition sweep, bowing parameter
"""
from .solver import embed_onsite_in_cell, cpa_solve_energy, cpa_solve_grid
from .greens import (
    dos_from_eigs, dos_from_gloc, spectral_function_k, spectral_function_k_matrix,
    spectral_map_kpath, pdos_from_gloc, scattering_rates, effective_bands,
)
from .transport import (
    transport_distribution, transport_integrals,
    optical_conductivity, dc_conductivity, thermoelectric_coefficients,
    diamond_cell_volume,
)
from .observables import find_spectral_gap, fit_bowing
