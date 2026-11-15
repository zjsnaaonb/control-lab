"""Box-constrained quadratic programming via ADMM.

    minimise    0.5 * x'Px + q'x
    subject to  lb <= x <= ub

No external solver is required.  The quadratic term is factorised once, so
repeated solves that share the same ``P`` -- the usual situation in model
predictive control, where only the linear term depends on the current state --
cost one triangular solve per ADMM iteration.

ADMM splits the problem into a smooth quadratic part and the box, which
projects onto itself in closed form:

    x-update :  (P + rho*I) x = rho*(z - u) - q
    z-update :  z = clip(x + u, lb, ub)
    u-update :  u = u + x - z

Because the projection is exact and cheap, this converges reliably on the
small dense problems that arise with a horizon of a few tens of steps.
"""

from __future__ import annotations

import numpy as np


class BoxQP:
    """Reusable solver for ``0.5*x'Px + q'x`` subject to a box.

    Parameters
    ----------
    P:
        Positive definite (or at least positive semi-definite plus the
        ``rho*I`` shift) quadratic term, shape ``(n, n)``.
    rho:
        ADMM penalty.  Larger values converge faster but rotate the
        iterates; 1.0 is a reasonable default for normalised problems.
    max_iter:
        Iteration cap.  The solver returns its best iterate regardless.
    tol:
        Relative tolerance applied to both primal and dual residuals.
    """

    def __init__(self, P, rho: float = 0.5, max_iter: int = 200, tol: float = 1e-8):
        self.P = np.asarray(P, dtype=float)
        if self.P.ndim != 2 or self.P.shape[0] != self.P.shape[1]:
            raise ValueError("P must be square")
        self.n = self.P.shape[0]
        self.rho = float(rho)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        shifted = self.P + self.rho * np.eye(self.n)
        # Cholesky fails on a merely semi-definite matrix; fall back to a
        # symmetric solve in that case so the class stays usable.
        try:
            self._L = np.linalg.cholesky(shifted)
            self._reduced = False
        except np.linalg.LinAlgError:
            self._L = None
            self._reduced = True
            self._M = shifted

    def _factor_solve(self, rhs):
        if self._L is None:
            return np.linalg.solve(self._M, rhs)
        y = np.linalg.solve(self._L, rhs)
        return np.linalg.solve(self._L.T, y)

    def solve(self, q, lb, ub, x0=None):
        """Solve one QP.

        Returns ``(x, info)`` where ``info`` carries ``iters``, ``primal_res``
        and ``dual_res`` -- useful for showing the computational cost of a
        control step honestly rather than claiming it is free.
        """
        q = np.asarray(q, dtype=float).reshape(self.n)
        lb = np.broadcast_to(np.asarray(lb, dtype=float), (self.n,))
        ub = np.broadcast_to(np.asarray(ub, dtype=float), (self.n,))
        if np.any(lb > ub):
            raise ValueError("lb must not exceed ub")

        rho = self.rho
        x = np.zeros(self.n) if x0 is None else np.asarray(x0, dtype=float).copy()
        z = np.clip(x, lb, ub)
        u = np.zeros(self.n)

        iters = 0
        primal_res = np.inf
        dual_res = np.inf
        for iters in range(1, self.max_iter + 1):
            x = self._factor_solve(rho * (z - u) - q)
            z_prev = z
            z = np.clip(x + u, lb, ub)
            u += x - z

            primal_res = float(np.linalg.norm(x - z))
            dual_res = float(np.linalg.norm(rho * (z - z_prev)))

            scale_p = 1.0 + max(np.linalg.norm(x), np.linalg.norm(z))
            scale_d = 1.0 + np.linalg.norm(rho * u)
            if primal_res <= self.tol * scale_p and dual_res <= self.tol * scale_d:
                break

        info = {"iters": iters, "primal_res": primal_res, "dual_res": dual_res}
        return z, info


def solve_box_qp(P, q, lb, ub, **kwargs):
    """Convenience wrapper around :class:`BoxQP` for one-shot use."""
    return BoxQP(P, **kwargs).solve(q, lb, ub)
