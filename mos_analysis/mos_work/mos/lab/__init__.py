"""Composable Learning Lab feature registry.

Feature modules consume solved Results/Model data and never invoke the solver.
Add future analysis modules by registering a small adapter here.
"""
from .stress import evaluate_stress
from .traction import evaluate_traction
from .summary import model_summary, tensor_matrix_html

FEATURES = {
    "stress": evaluate_stress,
    "traction": evaluate_traction,
    "summary": model_summary,
}


def run_feature(name, *args, **kwargs):
    """Dispatch a registered post-processing feature by its stable key."""
    try:
        feature = FEATURES[name]
    except KeyError:
        raise KeyError(f"Unknown Learning Lab feature {name!r}; available: {', '.join(FEATURES)}") from None
    return feature(*args, **kwargs)


__all__ = ["FEATURES", "run_feature", "evaluate_stress", "evaluate_traction",
           "model_summary", "tensor_matrix_html"]
