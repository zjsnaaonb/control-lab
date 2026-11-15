"""Simulation session: plant + controller + observer, stepped in real time."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from core.controllers import build_controller
from core.linalg import dare
from core.models import get_plant

TS = 0.02  # 50 Hz control loop, fixed so the UI can reason about time


class Simulation:
    """One run of one plant under one controller.

    The session is deliberately stateful: changing controller parameters keeps
    the plant state, so the effect of the change is visible immediately instead
    of restarting from the initial condition.
    """

    MAX_STEPS = 120

    def __init__(self, config: Optional[dict] = None):
        self.config = {
            "plant": "inverted_pendulum",
            "controller": "mpc",
            "params": {},
            "ref": None,
            "noise": 0.0,
            "observer": False,
            "u_limit": None,
            "speed": 1.0,
            "paused": False,
        }
        if config:
            self.config.update(config)

        self.rng = np.random.default_rng(7)
        self._build(reuse_state=False)
        self.reset()
        self._pulse_steps = 0
        self._pulse_amp = 0.0
        self.diverged = 0
        self._solve_ms_avg: Optional[float] = None

    # ------------------------------------------------------------------ setup
    def _build(self, reuse_state: bool):
        plant_key = self.config["plant"]
        self.plant = get_plant(plant_key)
        self.Ad, self.Bd, self.C, _ = self.plant.discrete(TS)

        u_lim = self.config.get("u_limit") or self.plant.u_limit
        self.u_limit = float(u_lim)
        self.controller = build_controller(
            self.config["controller"], self.plant, TS,
            dict(self.config.get("params") or {}), u_range=self.u_limit,
        )

        ref = self.config.get("ref")
        if not ref or len(ref) != self.plant.n_outputs:
            ref = [0.0] * self.plant.n_outputs
            self.config["ref"] = ref
        self.ref = np.asarray(ref, dtype=float)

        self.noise = float(self.config.get("noise") or 0.0)
        self.use_observer = bool(self.config.get("observer"))
        if self.use_observer:
            # Luenberger gain from the dual Riccati problem.
            measure_var = max(self.noise * self.noise, 1e-6)
            _, K, _ = dare(self.Ad.T, self.C.T,
                           np.eye(self.plant.n_states),
                           np.eye(self.plant.n_outputs) * measure_var)
            self.L = K.T
        else:
            self.L = None

        if not reuse_state:
            self.x = self.plant.x0.copy()
            self.x_hat = self.x.copy()

    def reset(self):
        self.config["paused"] = False
        self.x = self.plant.x0.copy()
        self.x_hat = self.x.copy()
        self.t = 0.0
        self.u_prev = np.zeros(self.plant.n_inputs)
        self.diverged = 0
        self._pulse_steps = 0
        self._pulse_amp = 0.0
        self._solve_ms_avg = None

    def apply_config(self, config: dict):
        """Merge a new configuration.

        The plant state survives a parameter change so the user can watch
        LQR -> MPC hand over mid-flight, but a plant or reference change
        restarts from the initial condition.
        """
        restart = any(
            key in config and config[key] != self.config.get(key)
            for key in ("plant",)
        )
        self.config.update(config)
        ref_changed = "ref" in config
        self._build(reuse_state=not (restart or ref_changed))
        if restart or ref_changed:
            self.reset()

    def inject_disturbance(self, amplitude: float, duration: float = 0.12):
        """Push the cart: a bounded external force pulse on the input channel."""
        self._pulse_amp = float(amplitude)
        self._pulse_steps = max(1, int(round(duration / TS)))

    # ------------------------------------------------------------------- step
    def step(self) -> Dict:
        y_true = self.C @ self.x
        y_meas = y_true + self.noise * self.rng.standard_normal(self.plant.n_outputs)

        state_for_control = self.x_hat if self.use_observer else self.x
        u, info = self.controller.compute(state_for_control, self.ref)
        u = np.asarray(u, dtype=float)

        # Disturbance acts on the plant, not on the controller's decision.
        u_total = u.copy()
        if self._pulse_steps > 0:
            u_total = u_total + self._pulse_amp
            self._pulse_steps -= 1

        self.x = self.Ad @ self.x + self.Bd @ u_total
        if self.use_observer:
            innovation = y_meas - self.C @ self.x_hat
            self.x_hat = self.Ad @ self.x_hat + self.Bd @ u + self.L @ innovation
        else:
            self.x_hat = self.x
        self.u_prev = u
        self.t += TS

        solve_ms = float(info.get("solve_ms") or 0.0)
        if solve_ms > 0.0:
            self._solve_ms_avg = solve_ms if self._solve_ms_avg is None else \
                0.9 * self._solve_ms_avg + 0.1 * solve_ms

        # Numerical escape hatch: a badly tuned loop must not poison the stream
        # with inf/nan. Record it honestly and restart.
        exploded = not np.all(np.isfinite(self.x)) or float(np.max(np.abs(self.x))) > 1e4
        if exploded:
            self.diverged += 1
            self.x = self.plant.x0.copy()
            self.x_hat = self.x.copy()
            if hasattr(self.controller, "reset"):
                self.controller.reset()

        return {
            "t": round(self.t, 4),
            "y": y_true.round(6).tolist(),
            "u": float(u[0]),
            "x": self.x.round(6).tolist(),
            "ref": self.ref.tolist(),
            "iters": int(info.get("iters") or 0),
            "diverged": bool(exploded),
        }

    def advance(self, n_steps: int) -> List[Dict]:
        n = int(max(0, min(n_steps, self.MAX_STEPS)))
        return [self.step() for _ in range(n)]

    # ------------------------------------------------------------------ meta
    def describe(self) -> Dict:
        p = self.plant
        return {
            "plant": {
                "key": p.key,
                "name": p.name,
                "state_names": p.state_names,
                "output_names": p.output_names,
                "input_name": p.input_name,
                "u_limit": self.u_limit,
                "x0": p.x0.tolist(),
            },
            "controller": self.controller.name,
            "config": self.config,
            "ts": TS,
            "has_solver": self.controller.name in ("mpc", "lqr"),
        }

    def stats(self) -> Dict:
        return {
            "solve_ms": None if self._solve_ms_avg is None else round(self._solve_ms_avg, 3),
            "diverged": self.diverged,
            "x": [round(float(v), 5) for v in self.x],
            "u": float(self.u_prev[0]),
            "t": round(self.t, 3),
        }
