"""Controllers: PID, state-feedback LQR and constrained MPC.

All three expose the same tiny interface::

    u, info = controller.compute(x_hat, y_ref)

``x_hat`` is the estimated state (length n), ``y_ref`` the reference for the
measured outputs.  ``info`` carries solver statistics so the UI can show what
a control step actually costs.
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from .linalg import dare
from .qp import BoxQP

# State weights used as the default Q for LQR and MPC.  The pendulum needs a
# much heavier penalty on angle than on cart position, otherwise it trades
# stabilisation for station keeping and topples over.
DEFAULT_WEIGHTS: Dict[str, List[float]] = {
    "double_integrator": [10.0, 1.0],
    "mass_spring": [10.0, 1.0],
    "dc_motor": [10.0, 1.0],
    "inverted_pendulum": [6.0, 0.2, 30.0, 0.2],
}


def state_weights(plant_key: str, q_scale: float = 1.0) -> np.ndarray:
    w = np.asarray(DEFAULT_WEIGHTS.get(plant_key, [1.0]), dtype=float)
    return w * float(q_scale)


# Reasonable single-loop PID gains per plant.  The pendulum pair is deliberately
# tuned for angle regulation only: it keeps theta near zero but has no idea where
# the cart ends up -- the clearest demonstration of what a single loop cannot do.
PID_DEFAULTS: Dict[str, Dict[str, float]] = {
    "double_integrator": {"kp": 4.0, "ki": 1.0, "kd": 1.5, "direction": 1.0},
    "mass_spring": {"kp": 3.0, "ki": 0.8, "kd": 0.9, "direction": 1.0},
    "dc_motor": {"kp": 4.0, "ki": 2.0, "kd": 0.6, "direction": 1.0},
    # The cart-pole is a reverse-acting plant: pushing right tips the pole
    # left, so a naive PID with positive gains fights its own sign and diverges.
    # ``direction = -1`` is what a commissioning engineer would set by hand.
    "inverted_pendulum": {"kp": 16.0, "ki": 0.0, "kd": 4.0, "direction": -1.0},
}


def pid_defaults(plant_key: str) -> Dict[str, float]:
    return dict(PID_DEFAULTS.get(plant_key, {"kp": 2.0, "ki": 1.0, "kd": 0.2, "direction": 1.0}))


def reference_state(C: np.ndarray, y_ref) -> np.ndarray:
    """Map an output reference onto the state space (minimum-norm lift).

    Unmeasured states get zero reference, which is what a regulator wants.
    """
    y_ref = np.asarray(y_ref, dtype=float)
    return np.linalg.pinv(C) @ y_ref


class Controller:
    """Common base: holds the discretised plant and bookkeeping."""

    name = "base"

    def __init__(self, Ad, Bd, C, Ts, u_min, u_max):
        self.Ad = np.asarray(Ad, dtype=float)
        self.Bd = np.asarray(Bd, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.Ts = float(Ts)
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.n = self.Ad.shape[0]
        self.m = self.Bd.shape[1]

    def _clip(self, u):
        return np.minimum(np.maximum(u, self.u_min), self.u_max)

    def compute(self, x_hat, y_ref):  # pragma: no cover - interface only
        raise NotImplementedError


class PID(Controller):
    """Single-loop PID on one measured channel, with clamping anti-windup.

    Only one output is regulated, which is exactly the limitation this demo
    is meant to show: on the cart-pole it keeps the angle near zero while the
    cart drifts away, because nothing constrains the unmeasured state.
    """

    name = "pid"

    def __init__(self, Ad, Bd, C, Ts, u_min, u_max, kp=2.0, ki=1.0, kd=0.2,
                 channel: int = -1, d_filter: float = 0.15, direction: float = 1.0):
        super().__init__(Ad, Bd, C, Ts, u_min, u_max)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.channel = int(channel)
        self.alpha = float(d_filter)
        self.direction = float(direction)   # +1 or -1: plant sign convention
        self._integral = 0.0
        self._prev_e = 0.0
        self._d_state = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_e = 0.0
        self._d_state = 0.0

    def compute(self, x_hat, y_ref):
        y = self.C @ x_hat
        e = float(y_ref.reshape(-1)[self.channel] - y[self.channel])

        # Derivative on error, low-pass filtered to keep noise from exploding.
        d_raw = (e - self._prev_e) / self.Ts
        self._d_state = (1.0 - self.alpha) * self._d_state + self.alpha * d_raw
        self._prev_e = e

        candidate = self._integral + e * self.Ts
        raw_test = self.kp * e + self.ki * candidate + self.kd * self._d_state
        u_test = self.direction * raw_test
        # Freeze the integrator while the actuator is saturated: this is
        # clamping anti-windup, the standard fix for slow recovery after a limit.
        if u_test > self.u_max or u_test < self.u_min:
            candidate = self._integral
        self._integral = candidate

        raw = self.kp * e + self.ki * self._integral + self.kd * self._d_state
        u = self._clip(self.direction * raw)
        return np.array([u]), {"iters": 0, "solve_ms": 0.0}


class LQR(Controller):
    """Optimal state feedback from the discrete Riccati solution.

    Unconstrained by construction: this is the baseline the MPC comparison
    should be measured against.
    """

    name = "lqr"

    def __init__(self, Ad, Bd, C, Ts, u_min, u_max, weights, r=1.0):
        super().__init__(Ad, Bd, C, Ts, u_min, u_max)
        Q = np.diag(np.asarray(weights, dtype=float))
        R = np.eye(self.m) * float(r)
        self.P, self.K, self.riccati_iters = dare(self.Ad, self.Bd, Q, R)

    def compute(self, x_hat, y_ref):
        x_ref = reference_state(self.C, y_ref)
        u = -self.K @ (x_hat - x_ref)
        u = self._clip(u)
        # The Riccati solve happened once at construction; the online step is
        # a single matrix-vector product, so it is honestly reported as ~0 ms.
        return u, {"iters": 0, "solve_ms": 0.0, "setup_iters": int(self.riccati_iters)}


class MPC(Controller):
    """Condensed constrained MPC with a box on the input.

    The QP is built once per parameter set -- only the linear term moves with
    the state -- so the Cholesky factor of the quadratic term is cached and
    each control step costs a handful of triangular solves.  Warm start uses
    the previous solution shifted by one step.
    """

    name = "mpc"

    def __init__(self, Ad, Bd, C, Ts, u_min, u_max, weights, r=1.0,
                 horizon: int = 30, rho: float = 0.5, terminal_weight: bool = True):
        super().__init__(Ad, Bd, C, Ts, u_min, u_max)
        self.N = int(horizon)
        if self.N < 2:
            raise ValueError("horizon must be at least 2")
        Q = np.diag(np.asarray(weights, dtype=float))
        R = np.eye(self.m) * float(r)
        self._build_prediction(Q, R, bool(terminal_weight))
        self._u_shift = np.zeros(self.N * self.m)

    def _build_prediction(self, Q, R, terminal_weight):
        n, m, N = self.n, self.m, self.N

        # F: (N+1)n x n, G: (N+1)n x Nm   --  X = F x0 + G U
        powers = [np.eye(n)]
        for _ in range(N):
            powers.append(self.Ad @ powers[-1])

        F = np.zeros(((N + 1) * n, n))
        G = np.zeros(((N + 1) * n, N * m))
        for k in range(N + 1):
            F[k * n:(k + 1) * n, :] = powers[k]
            for j in range(k):
                G[k * n:(k + 1) * n, j * m:(j + 1) * m] = powers[k - j - 1] @ self.Bd

        Qbar = np.kron(np.eye(N + 1), Q)
        if terminal_weight:
            # Infinite-horizon cost as a terminal penalty; this is what makes
            # a horizon of a few tenths of a second viable on an unstable plant.
            P, _, _ = dare(self.Ad, self.Bd, Q, R)
            Qbar[-n:, -n:] += P

        Rbar = np.kron(np.eye(N), R)

        self.Qbar = Qbar
        self.F = F
        self.G = G
        self.H = G.T @ Qbar @ G + Rbar
        self.H = 0.5 * (self.H + self.H.T)

        self.lb = np.full(N * m, self.u_min)
        self.ub = np.full(N * m, self.u_max)
        self.qp = BoxQP(self.H, rho=0.5, max_iter=150, tol=1e-7)

    def compute(self, x_hat, y_ref):
        x_ref = reference_state(self.C, y_ref)
        # Constant reference: the deviation obeys the homogeneous dynamics,
        # so the predicted error stack is simply F * (x - x_ref).
        x_err = x_hat - x_ref
        q = self.G.T @ (self.Qbar @ (self.F @ x_err))

        t0 = time.perf_counter()
        U, info = self.qp.solve(q, self.lb, self.ub, x0=self._u_shift)
        solve_ms = (time.perf_counter() - t0) * 1e3

        u = U[:self.m].copy()
        # Shift the horizon for the next warm start, repeating the last input.
        self._u_shift = np.concatenate([U[self.m:], U[-self.m:]])

        info = {
            "iters": int(info["iters"]),
            "solve_ms": float(solve_ms),
            "primal_res": float(info["primal_res"]),
        }
        return u, info


def build_controller(kind: str, plant, Ts: float, params: dict, u_range=None):
    """Factory used by the server; keeps parameter coercion in one place."""
    Ad, Bd, C, _ = plant.discrete(Ts)
    u_lim = float(plant.u_limit if u_range is None else u_range)
    weights = state_weights(plant.key, params.get("q_scale", 1.0))
    r = float(params.get("r", 1.0))
    common = dict(Ad=Ad, Bd=Bd, C=C, Ts=Ts, u_min=-u_lim, u_max=u_lim)

    kind = (kind or plant.default_controller).lower()
    if kind == "pid":
        defaults = pid_defaults(plant.key)
        return PID(
            **common,
            kp=float(params.get("kp", defaults["kp"])),
            ki=float(params.get("ki", defaults["ki"])),
            kd=float(params.get("kd", defaults["kd"])),
            direction=float(params.get("direction", defaults["direction"])),
        )
    if kind == "lqr":
        return LQR(**common, weights=weights, r=r)
    if kind == "mpc":
        return MPC(**common, weights=weights, r=r,
                   horizon=int(params.get("horizon", 30)))
    raise ValueError(f"unknown controller '{kind}'")
