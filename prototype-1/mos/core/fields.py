"""Scalar / vector fields of position (loads, body forces, prescribed displacements)
and region conditions, written as text expressions.

Convention: positions have shape (3, ...); a field returns shape (...) for a scalar
field or (3, ...) for a vector field. This is scikit-fem's layout, so a field can be
evaluated directly at quadrature points.

Expressions are parsed with `ast` and run by a small whitelisted interpreter; nothing
is ever passed to eval()/exec(). Supported:
  numbers, x y z, named parameters, pi, e, + - * / ** (or ^), unary +/-
  functions   sin cos tan asin acos atan atan2 sinh cosh tanh exp log sqrt abs
              step(t)          0 for t < 0, 1 for t >= 0
              where(c, a, b)   a where condition c holds, else b
  conditions  < <= > >= == !=   (chains like  0 < x < 1  work),  and  or  not
"""
from __future__ import annotations

import ast
import functools
from typing import Callable, Sequence

import numpy as np

_BINOPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
           ast.Div: np.divide, ast.Pow: np.power}
_UNOPS = {ast.UAdd: np.positive, ast.USub: np.negative, ast.Not: np.logical_not}
_CMPOPS = {ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater,
           ast.GtE: np.greater_equal, ast.Eq: np.equal, ast.NotEq: np.not_equal}
_FUNCS = {"sin": np.sin, "cos": np.cos, "tan": np.tan, "asin": np.arcsin,
          "acos": np.arccos, "atan": np.arctan, "atan2": np.arctan2,
          "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh, "exp": np.exp,
          "log": np.log, "sqrt": np.sqrt, "abs": np.abs,
          "step": lambda t: np.heaviside(t, 1.0),
          "where": lambda c, a, b: np.where(c, a, b)}
_CONSTS = {"pi": np.pi, "e": np.e}
RESERVED = frozenset(_FUNCS) | frozenset(_CONSTS) | {"x", "y", "z"}


def _eval(node, env):
    if isinstance(node, ast.Expression):
        return _eval(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(f"unknown name '{node.id}'")
        return env[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval(node.left, env), _eval(node.right, env))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
        return _UNOPS[type(node.op)](_eval(node.operand, env))
    if isinstance(node, ast.Compare) and all(type(o) in _CMPOPS for o in node.ops):
        left, out = _eval(node.left, env), None
        for op, comp in zip(node.ops, node.comparators):
            right = _eval(comp, env)
            r = _CMPOPS[type(op)](left, right)
            out = r if out is None else np.logical_and(out, r)
            left = right
        return out
    if isinstance(node, ast.BoolOp):
        f = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
        return functools.reduce(f, [_eval(v, env) for v in node.values])
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS and not node.keywords):
        try:
            return _FUNCS[node.func.id](*[_eval(a, env) for a in node.args])
        except TypeError:
            raise ValueError(f"wrong number of arguments to {node.func.id}()") from None
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


def _parse(expr: str):
    return ast.parse(expr.replace("^", "**").strip(), mode="eval")


def _check_params(params: dict):
    bad = RESERVED & set(params)
    if bad:
        raise ValueError(f"parameter name(s) {sorted(bad)} are reserved")


def _compile_expr(expr: str, params: dict) -> Callable:
    _check_params(params)
    tree = _parse(expr)
    consts = {**_CONSTS, **{k: float(v) for k, v in params.items()}}

    def fn(x, y, z):
        """Evaluate the parsed expression at coordinate arrays x, y, z."""
        return _eval(tree, {**consts, "x": x, "y": y, "z": z})

    one = np.ones(1)
    with np.errstate(all="ignore"):
        fn(one, one, one)  # validate names / syntax now, not at solve time
    return fn


def eval_constant(expr: str, **params: float) -> float:
    """Evaluate an expression that must not depend on x, y, z (e.g. '2*L')."""
    _check_params(params)
    with np.errstate(all="ignore"):
        out = _eval(_parse(expr), {**_CONSTS, **{k: float(v) for k, v in params.items()}})
    if np.ndim(out) != 0 or isinstance(out, (bool, np.bool_)):
        raise ValueError(f"'{expr}' is not a number")
    if not np.isfinite(out):
        raise ValueError(f"'{expr}' is not finite (division by zero, log of a negative, ...)")
    return float(out)


def compile_predicate(expr: str, **params: float) -> Callable:
    """Condition on position, e.g. 'abs(x - L) < 1e-9 and y > 0'.  Returns pred(x, y, z) -> bool array."""
    fn = _compile_expr(expr, params)

    def pred(x, y, z):
        """Boolean array: where the condition holds at the coordinate arrays x, y, z."""
        out = np.asarray(fn(x, y, z))
        if out.dtype != bool:
            raise ValueError(f"'{expr}' is not a condition (use <, >, ==, and, or ...)")
        return np.array(np.broadcast_to(out, np.shape(x)))

    one = np.ones(1)
    pred(one, one, one)
    return pred


def split_top_level(s: str, sep: str = ",") -> list:
    """Split on `sep` outside brackets: 'where(x>1, 2, 3), 0, -g' -> 3 parts."""
    parts, depth, cur = [], 0, []
    for ch in s:
        depth += ch in "([{"
        depth -= ch in ")]}"
        if depth < 0:
            raise ValueError("unbalanced brackets")
        if ch == sep and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if depth:
        raise ValueError("unbalanced brackets")
    parts.append("".join(cur).strip())
    return parts


class Field:
    """A scalar or vector function of position, evaluated on arrays of points.

    Build with `constant`, `from_expr` (text) or `from_function` (Python callable).
    Call it with positions of shape (3, ...); it returns (...) if scalar, (3, ...) if vector.
    """

    def __init__(self, fn: Callable, vector: bool):
        self._fn = fn
        self.vector = vector

    @classmethod
    def constant(cls, value) -> "Field":
        """Field with the same value everywhere: a number (scalar) or a 3-vector."""
        v = np.asarray(value, dtype=float)
        if v.ndim == 0:
            return cls(lambda x, y, z: v, False)
        if v.shape == (3,):
            return cls(lambda x, y, z: v, True)
        raise ValueError("constant must be a scalar or a 3-vector")

    @classmethod
    def from_function(cls, fn: Callable, vector: bool = False) -> "Field":
        """fn(x, y, z) -> array (scalar) or a sequence of 3 arrays (vector)."""
        return cls(fn, vector)

    @classmethod
    def from_expr(cls, expr: str | Sequence[str], **params: float) -> "Field":
        """Field.from_expr("-rho*g", rho=7850, g=9.81)                      scalar
           Field.from_expr(["0", "where(x < L/2, -p, 0)", "0"], p=1e3, L=10)  vector"""
        if isinstance(expr, str):
            return cls(_compile_expr(expr, params), False)
        exprs = [_compile_expr(e, params) for e in expr]
        if len(exprs) != 3:
            raise ValueError("a vector expression needs exactly 3 components")
        return cls(lambda x, y, z: [f(x, y, z) for f in exprs], True)

    def __call__(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        grid = X.shape[1:]
        out = self._fn(X[0], X[1], X[2])
        if self.vector:
            comps = list(out)
            if len(comps) != 3:
                raise ValueError("vector field must return 3 components")
            return np.stack([np.broadcast_to(np.asarray(c, float), grid) for c in comps])
        return np.array(np.broadcast_to(np.asarray(out, float), grid))


def as_field(value, vector: bool) -> Field:
    """Return `value` if it is already a Field of the right rank, else wrap it as a constant."""
    if isinstance(value, Field):
        if value.vector != vector:
            raise ValueError("expected a %s field" % ("vector" if vector else "scalar"))
        return value
    return Field.constant(value)
