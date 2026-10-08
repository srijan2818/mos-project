"""Load / material / region / field construction and validation."""
import numpy as np

from mos.core import BodyForce, Field, Material, Moment, PointForce, Pressure, Region, Traction
from mos.core.fields import as_field

F = Region("f", "facets", [0, 1])
N = Region("n", "nodes", [3])
X = np.zeros((3, 2))


def _raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except ValueError:
        return
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise ValueError")


def test_loads_accept_raw_constants():
    t = Traction(F, [0, -1, 0])
    assert t.field.vector and np.array_equal(t.field(X), [[0, 0], [-1, -1], [0, 0]])
    p = Pressure(F, 5.0)
    assert not p.p.vector and np.array_equal(p.p(X), [5, 5])
    b = BodyForce([0, 0, -9.81])
    assert np.allclose(b.field(X)[2], -9.81)
    assert np.array_equal(PointForce(N, [1, 2, 3]).force, [1, 2, 3])
    assert np.array_equal(Moment(F, [0, 0, 1]).M, [0, 0, 1]) and Moment(F, [0, 0, 1]).origin is None
    assert Field.constant(2.0)(X).shape == (2,)


def test_load_validation():
    _raises(PointForce, F, [1, 2, 3])                    # needs node region
    _raises(PointForce, N, [1, 2])                       # needs 3-vector
    _raises(PointForce, Region("e", "nodes", []), [1, 2, 3])
    _raises(Traction, N, [0, 0, 0])                      # needs facets
    _raises(Traction, Region("e", "facets", []), [0, 0, 0])
    _raises(Pressure, N, 1.0)
    _raises(Moment, F, [1, 2])
    _raises(Pressure, F, Field.constant([1, 2, 3]))      # vector where scalar expected
    _raises(BodyForce, Field.constant(1.0))              # scalar where vector expected
    _raises(Region, "bad", "cells", [0])


def test_field_construction_errors():
    _raises(Field.constant, [1, 2])
    _raises(Field.from_expr, ["0", "0"])                 # 2 components
    _raises(lambda: Field.from_function(lambda x, y, z: [x, y], vector=True)(X))
    assert as_field(Field.constant(1.0), False).vector is False
    assert as_field([1, 2, 3], True).vector is True


def test_material_validation_and_anisotropic_c():
    C = np.eye(6); C[0, 1] = 0.5                         # not symmetric
    _raises(Material, C)
    _raises(Material.isotropic, 1.0, 0.5)                # nu out of range
    _raises(Material.isotropic, 1.0, -1.0)
    C = np.diag([1.0, 2, 3, 4, 5, 6])                    # general orthotropic-style C is accepted
    m = Material(C, rho=7.0)
    assert m.rho == 7.0 and m.C4[0, 1, 0, 1] == 6.0 and m.C4[1, 2, 2, 1] == 4.0
