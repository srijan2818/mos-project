"""Library-independent tests (no scikit-fem needed)."""
import numpy as np

from mos.core import Field, Material, Region, facets_where, nearest_node, nodes_where
from mos.core.loads import cross0, moment_coefficients
from mos.core import post


def test_isotropic_hooke():
    E, nu = 200e9, 0.3
    m = Material.isotropic(E, nu)
    # uniaxial strain state -> uniaxial stress
    eps = np.zeros((3, 3)); eps[0, 0] = 1e-3; eps[1, 1] = eps[2, 2] = -nu * 1e-3
    sig = np.einsum("ijkl,kl->ij", m.C4, eps)
    assert np.isclose(sig[0, 0], E * 1e-3)
    assert np.allclose(sig - np.diag(np.diag(sig)), 0) and np.isclose(sig[1, 1], 0, atol=1e-3)
    # pure shear
    eps = np.zeros((3, 3)); eps[0, 1] = eps[1, 0] = 5e-4          # gamma = 1e-3
    sig = np.einsum("ijkl,kl->ij", m.C4, eps)
    assert np.isclose(sig[0, 1], E / (2 * (1 + nu)) * 1e-3)
    # C4 symmetries
    C4 = m.C4
    assert np.allclose(C4, C4.transpose(1, 0, 2, 3)) and np.allclose(C4, C4.transpose(2, 3, 0, 1))


def test_material_validation():
    for bad in (np.eye(5), -np.eye(6)):
        try:
            Material(bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_fields():
    X = np.random.default_rng(0).random((3, 4, 5))
    g = Field.from_expr(["0", "-rho*g*x^2", "sin(pi*z)"], rho=2.0, g=9.81)(X)
    assert g.shape == (3, 4, 5)
    assert np.allclose(g[1], -2 * 9.81 * X[0] ** 2) and np.allclose(g[2], np.sin(np.pi * X[2]))
    assert Field.constant([1, 2, 3])(X).shape == (3, 4, 5)
    assert Field.from_function(lambda x, y, z: x + y)(X).shape == (4, 5)


def test_field_rejects_unsafe_input():
    for bad in ("__import__('os').system('ls')", "x.__class__", "foo(x)", "q + 1",
                "[1,2]", "x if y else z", "lambda: 1"):
        try:
            Field.from_expr(bad)
        except (ValueError, SyntaxError):
            continue
        raise AssertionError(f"accepted {bad!r}")


class _OneTet:
    p = np.array([[0., 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    t = np.array([[0], [1], [2], [3]])
    facets = np.array([[0, 0, 0, 1], [1, 1, 2, 2], [2, 3, 3, 3]])

    def boundary_facets(self):
        return np.arange(4)


def test_regions():
    m = _OneTet()
    r = facets_where(m, lambda x, y, z: np.isclose(x, 0), "yz")
    # facets are columns of m.facets: f0=(0,1,2) f1=(0,1,3) f2=(0,2,3) f3=(1,2,3);
    # only f2 has all three vertices on x = 0
    assert r.kind == "facets" and r.ids.tolist() == [2]
    assert nodes_where(m, lambda x, y, z: np.isclose(x, 0), "n").ids.tolist() == [0, 2, 3]
    assert nearest_node(m, (0.9, 0.1, 0), "tip").ids.tolist() == [1]
    assert len(facets_where(m, lambda x, y, z: x > 5, "none")) == 0


def _plate(n=60):
    """Quadrature-like points on the rectangle x=0, y in [-1,1], z in [-.5,.5]."""
    ys, zs = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-.5, .5, n))
    X = np.stack([np.zeros(ys.size), ys.ravel(), zs.ravel()])
    w = np.full(ys.size, 2.0 * 1.0 / ys.size)
    return X, w


def test_moment_traction_is_statically_equivalent():
    X, w = _plate()
    for M in ([0, 0, 7.0], [3.0, 0, 0], [0, -2.0, 0], [1.0, 2.0, 3.0]):
        omega, o = moment_coefficients(X, w, M)
        t = cross0(omega, X - o[:, None])
        force = (t * w).sum(axis=1)
        moment = (cross0_each(X - o[:, None], t) * w).sum(axis=1)
        assert np.allclose(force, 0, atol=1e-10), force
        assert np.allclose(moment, M, atol=1e-10), moment


def test_moment_with_explicit_origin_remains_a_zero_force_couple():
    X, w = _plate()
    M = np.array([2.0, -3.0, 5.0])
    reported_origin = np.array([12.0, -7.0, 9.0])
    omega, centroid = moment_coefficients(X, w, M, reported_origin)
    traction = cross0(omega, X - centroid[:, None])
    resultant_force = (traction * w).sum(axis=1)
    # The resultant moment of a pure couple is independent of the reference point.
    r = X - reported_origin[:, None]
    resultant_moment = (cross0_each(r, traction) * w).sum(axis=1)
    assert np.allclose(resultant_force, 0.0, atol=1e-12)
    assert np.allclose(resultant_moment, M, atol=1e-10)


def cross0_each(a, b):
    return np.stack([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def test_bending_moment_gives_linear_normal_traction():
    X, w = _plate()
    omega, o = moment_coefficients(X, w, [0, 0, 5.0])      # M_z on a face with normal +x
    t = cross0(omega, X - o[:, None])
    assert np.allclose(t[1:], 0)                            # normal traction only
    I = (w * X[1] ** 2).sum()
    assert np.allclose(t[0], -5.0 * X[1] / I)               # sigma = -M y / I


def test_torsion_gives_tangential_traction_proportional_to_radius():
    X, w = _plate()
    omega, o = moment_coefficients(X, w, [4.0, 0, 0])
    t = cross0(omega, X - o[:, None])
    J = (w * (X[1] ** 2 + X[2] ** 2)).sum()
    assert np.allclose(t[0], 0)
    assert np.allclose(np.hypot(t[1], t[2]), 4.0 * np.hypot(X[1], X[2]) / J)


def test_degenerate_face_rejected():
    X = np.stack([np.zeros(10), np.linspace(-1, 1, 10), np.zeros(10)])   # a line
    try:
        moment_coefficients(X, np.full(10, .1), [0, 0, 1.0])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_von_mises_and_principal():
    S = np.zeros((3, 3, 2)); S[0, 0] = 100.0                    # uniaxial
    assert np.allclose(post.von_mises(S), 100)
    s, _ = post.principal(S)
    assert np.allclose(s[:, 0], [100, 0, 0])
    S = np.zeros((3, 3)); S[0, 1] = S[1, 0] = 30.0              # pure shear
    assert np.isclose(post.von_mises(S), 30 * np.sqrt(3))
    s, v = post.principal(S)
    assert np.allclose(s, [30, 0, -30]) and np.isclose(post.tresca(S), 60)
    # eigen-decomposition reproduces the tensor
    assert np.allclose(v @ np.diag(s) @ v.T, S)


def test_nodal_average():
    # cell0 = nodes (0,1,2,3), cell1 = nodes (1,2,3,4); volumes 1 and 3
    t = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    Tc = np.array([[10.0, 30.0]])
    out = post.nodal_average(Tc, t, np.array([1.0, 3.0]), 5)
    assert np.allclose(out[0], [10, 25, 25, 25, 30])           # (10*1 + 30*3)/4 = 25


# ---------------------------------------------------------------- step / where / conditions
from mos.core.fields import compile_predicate, eval_constant, split_top_level

_X = np.array([[-1.0, 0.0, 2.0], [0, 0, 0], [0, 0, 0]])


def test_step():
    assert np.array_equal(Field.from_expr("step(x)")(_X), [0, 1, 1])      # step(0) = 1
    assert np.array_equal(Field.from_expr("step(x - 1)*5")(_X), [0, 0, 5])


def test_where_and_conditions():
    f = lambda e: Field.from_expr(e)(_X)
    assert np.array_equal(f("where(x > 0, 10, -10)"), [-10, -10, 10])
    assert np.array_equal(f("where(-1 < x < 1, 1, 0)"), [0, 1, 0])        # chained
    assert np.array_equal(f("where(x > -2 and x < 1, 1, 0)"), [1, 1, 0])
    assert np.array_equal(f("where(x < -0.5 or x > 1, 1, 0)"), [1, 0, 1])
    assert np.array_equal(f("where(not x > 0, 1, 0)"), [1, 1, 0])
    assert np.array_equal(f("where(abs(x) < 1e-12, 7, 0)"), [0, 7, 0])
    g = Field.from_expr(["0", "where(x < L/2, -p, 0)", "0"], p=3.0, L=2.0)(_X)
    assert np.array_equal(g[1], [-3, -3, 0])
    assert np.array_equal(f("(x > 0) * 4"), [0, 0, 4])                  # bool in arithmetic


def test_where_step_misuse_rejected():
    for bad in ("where(x)", "where(x, 1)", "step(x, 1)", "step()"):
        try:
            Field.from_expr(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_predicates():
    p = compile_predicate("abs(x - L) < 1e-6 and y >= 0", L=2.0)
    x = np.array([2.0, 2.0, 1.0]); y = np.array([0.0, -1.0, 0.0])
    assert p(x, y, y).tolist() == [True, False, False]
    for bad in ("x + 1", "step(x)", "x"):
        try:
            compile_predicate(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_eval_constant_and_split():
    assert abs(eval_constant("2*L^2", L=3.0) - 18.0) < 1e-12
    for bad in ("x", "1 < 2", "foo", "1/0", "log(-1)", "sqrt(-4)"):
        try:
            eval_constant(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")
    assert split_top_level("where(x>1, 2, 3), 0, -g") == ["where(x>1, 2, 3)", "0", "-g"]
    for bad in ("f(1, 2", "1), 2"):                      # unclosed / over-closed bracket
        try:
            split_top_level(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_reserved_parameter_names_rejected():
    for name in ("x", "pi", "step", "where"):
        try:
            Field.from_expr("1", **{name: 1.0})
        except ValueError:
            continue
        raise AssertionError(name)
