"""
Tight-binding Hamiltonian for diamond-structure sp3s* model.

Submodules:
  params         – TB parameters & VCA mixing
  lattice        – Diamond/FCC geometry & k-grids
  slater_koster  – Slater–Koster hopping matrices
  bloch          – Bloch Hamiltonian H(k) assembly + SOC
"""
from .params import (
    TBParams, SI_VOGL, GE_VOGL, A_SI, A_GE,
    vegard_a, mix_params_vca, onsite_matrix, disorder_onsites,
)
from .lattice import (
    fcc_primitive_vectors, reciprocal_vectors, diamond_basis,
    diamond_nn_vectors, monkhorst_pack, frac_to_cart_k,
)
from .slater_koster import hopping_sp3s_star
from .bloch import (
    p_soc_matrix, bloch_hamiltonian_sp3s_star, hopping_only_matrix,
    velocity_matrix_sp3s_star,
)
