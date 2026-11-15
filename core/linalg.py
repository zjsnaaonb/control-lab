"""Tiny numerical helpers that keep ``core`` free of SciPy.

Only two things are needed beyond NumPy: a matrix exponential (for zero-order
hold discretisation) and the discrete algebraic Riccati equation.  Both are
implemented here so the package stays installable with a single dependency.
"""

from __future__ import annotations

import numpy as np


def expm(M):
    """Matrix exponential via scaling-and-squaring with a Taylor series.

    Accurate to roughly machine precision for the modest matrices used here
    (n <= 8).  Scaling keeps ``||M||_1 <= 0.5`` so the first twenty series
    terms are more than enough.
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("expm expects a square matrix")

    n = M.shape[0]
    norm1 = np.linalg.norm(M, 1)
    squares = 0
    while norm1 > 0.5:
        norm1 /= 2.0
        squares += 1

    A = M / float(2 ** squares)
    result = np.eye(n)
    term = np.eye(n)
    for k in range(1, 20):
        term = term @ A / k
        result = result + term

    for _ in range(squares):
        result = result @ result
    return result


def dare(A, B, Q, R, max_iter: int = 3000, tol: float = 1e-12):
    """Solve the discrete algebraic Riccati equation by value iteration.

    ``P = A'PA - A'PB (B'PB + R)^{-1} B'PA + Q``

    Returns ``(P, K, iters)`` with ``K`` the optimal state-feedback gain,
    ``u = -Kx``.  Raises when the recursion diverges or fails to settle, which
    is the honest outcome for an unstabilisable pair.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)

    P = Q.copy()
    for i in range(1, max_iter + 1):
        AtP = A.T @ P
        BtP = B.T @ P
        S = BtP @ B + R
        K = np.linalg.solve(S, BtP @ A)
        P_next = AtP @ A - (AtP @ B) @ K + Q
        P_next = 0.5 * (P_next + P_next.T)

        if not np.all(np.isfinite(P_next)):
            raise RuntimeError("Riccati iteration diverged; is the pair stabilisable?")

        gap = np.max(np.abs(P_next - P))
        if gap <= tol * (1.0 + np.max(np.abs(P_next))):
            # Recompute the gain from the converged P so that P and K agree.
            K = np.linalg.solve(B.T @ P_next @ B + R, B.T @ P_next @ A)
            return P_next, K, i
        P = P_next

    raise RuntimeError(f"Riccati iteration did not converge in {max_iter} steps")
