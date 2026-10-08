"""Linear elastic material, defined by its 6x6 stiffness matrix C (SI units).

Voigt order: xx, yy, zz, yz, xz, xy with *engineering* shear strains (gamma = 2*eps),
so that  sigma_voigt = C @ eps_voigt.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VOIGT = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def voigt_to_c4(C: np.ndarray) -> np.ndarray:
    """Expand a 6x6 Voigt matrix into the 3x3x3x3 tensor C_ijkl (sigma_ij = C_ijkl eps_kl)."""
    C4 = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(VOIGT):
        for b, (k, l) in enumerate(VOIGT):
            for p, q in {(i, j), (j, i)}:
                for r, s in {(k, l), (l, k)}:
                    C4[p, q, r, s] = C[a, b]
    return C4


@dataclass(frozen=True, eq=False)
class Material:
    """Linear elastic material: 6x6 stiffness `C` (Pa, Voigt order) and density `rho`."""

    C: np.ndarray          # (6, 6) stiffness, Pa
    rho: float = 0.0       # density, kg/m^3 (used to build gravity-type body forces)

    def __post_init__(self):
        C = np.asarray(self.C, dtype=float)
        if C.shape != (6, 6):
            raise ValueError("C must be 6x6")
        if not np.allclose(C, C.T, rtol=1e-9, atol=1e-9 * np.abs(C).max()):
            raise ValueError("C must be symmetric")
        if np.linalg.eigvalsh(C).min() <= 0:
            raise ValueError("C must be positive definite")
        object.__setattr__(self, "C", C)

    @property
    def C4(self) -> np.ndarray:
        """The 3x3x3x3 tensor form of C (sigma_ij = C4_ijkl eps_kl)."""
        return voigt_to_c4(self.C)

    @classmethod
    def isotropic(cls, E: float, nu: float, rho: float = 0.0) -> "Material":
        """Isotropic material from Young's modulus E, Poisson ratio nu and density rho."""
        if not -1.0 < nu < 0.5:
            raise ValueError("nu must be in (-1, 0.5)")
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        C = np.zeros((6, 6))
        C[:3, :3] = lam
        C[range(3), range(3)] += 2 * mu
        C[range(3, 6), range(3, 6)] = mu
        return cls(C, rho)
