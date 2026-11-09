from types import SimpleNamespace

import numpy as np

from mos.lab import evaluate_stress, evaluate_traction, model_summary, run_feature


def _results():
    sigma = np.zeros((3, 3, 2))
    sigma[:, :, 0] = np.diag([3.0, 2.0, 1.0]) * 1e6
    sigma[:, :, 1] = np.array([[2, 1, 0], [1, 4, 0], [0, 0, 1]]) * 1e6
    return SimpleNamespace(
        sig_nodal=sigma,
        sig_cell=sigma[:, :, :1],
        sig_q=sigma[:, :, :1, None],
        eps_nodal=np.zeros_like(sigma),
        u_nodal=np.zeros((3, 2)),
        traction_on_plane=lambda xyz, normal, points, where="nodal":
            __import__("mos.core.post", fromlist=["traction_on_plane"]).traction_on_plane(
                sigma, points, xyz, normal),
    )


def test_point_stress_returns_nearest_node_and_invariants():
    result = evaluate_stress(_results(), np.array([[0, 0, 0], [1, 0, 0]]), [0.9, 0, 0])
    assert result["node_index"] == 1
    assert np.allclose(result["principal"], [(3 + np.sqrt(2)) * 1e6, (3 - np.sqrt(2)) * 1e6, 1e6])
    from mos.core.post import von_mises
    assert np.isclose(result["von_mises"], von_mises(result["tensor"]))
    assert np.isclose(result["max_shear"], ((3 + np.sqrt(2)) - 1) * 0.5e6)


def test_traction_adapter_normalises_arbitrary_normal():
    points = np.array([[0, 0, 0], [1, 0, 0]])
    out = evaluate_traction(_results(), points, [0, 0, 0], [0, 0, 5])
    assert np.allclose(out["normal"], [0, 0, 1])
    assert np.allclose(out["traction"], [0, 0, 1e6])
    assert np.isclose(out["normal_stress"], 1e6)
    assert np.isclose(out["shear_stress"], 0.0)


def test_traction_rejects_zero_normal():
    import pytest
    with pytest.raises(ValueError, match="non-zero"):
        evaluate_traction(_results(), np.zeros((2, 3)), [0, 0, 0], [0, 0, 0])


def test_model_summary_is_graceful_without_a_solve():
    assert "No solved model" in model_summary(None, None, None)


def test_feature_registry_dispatches_and_rejects_unknown_keys():
    assert "No solved model" in run_feature("summary", None, None, None)
    import pytest
    with pytest.raises(KeyError, match="Unknown Learning Lab feature"):
        run_feature("buckling")
