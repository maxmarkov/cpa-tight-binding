
import numpy as np
from tb_cpa.params import SI_VOGL, GE_VOGL, mix_params_vca, disorder_onsites
from tb_cpa.lattice import monkhorst_pack
from tb_cpa.hamiltonian import bloch_hamiltonian_sp3s_star, hopping_only_matrix
from tb_cpa.cpa import cpa_solve_energy

def test_hermiticity():
    params = mix_params_vca(0.3, SI_VOGL, GE_VOGL)
    for kf in monkhorst_pack(2,2,2):
        H = bloch_hamiltonian_sp3s_star(kf, params)
        assert np.max(np.abs(H - H.conj().T)) < 1e-10

def test_cpa_endmembers():
    VA, VB = disorder_onsites(SI_VOGL, GE_VOGL)
    params = mix_params_vca(0.4, SI_VOGL, GE_VOGL)
    kgrid = monkhorst_pack(2,2,2)
    Hhop_k = np.array([hopping_only_matrix(kf, params) for kf in kgrid])
    z = 0.5 + 0.2j
    sig0, _ = cpa_solve_energy(z, Hhop_k, VA, VB, x=0.0, tol=1e-8, max_iter=50)
    sig1, _ = cpa_solve_energy(z, Hhop_k, VA, VB, x=1.0, tol=1e-8, max_iter=50)
    assert np.max(np.abs(sig0 - VA)) < 1e-6
    assert np.max(np.abs(sig1 - VB)) < 1e-6

if __name__ == "__main__":
    test_hermiticity()
    test_cpa_endmembers()
    print("All tests passed.")
