"""Plant library: linear time-invariant models with zero-order hold discretisation.

Every model is written in continuous state-space form so the numbers can be
checked against a textbook, then discretised at the simulation rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .linalg import expm


@dataclass(frozen=True)
class Plant:
    """A continuous-time LTI model plus the metadata the UI needs."""

    key: str
    name: str
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    state_names: List[str]
    output_names: List[str]
    input_name: str
    x0: np.ndarray
    u_limit: float
    default_controller: str

    @property
    def n_states(self) -> int:
        return self.A.shape[0]

    @property
    def n_inputs(self) -> int:
        return self.B.shape[1]

    @property
    def n_outputs(self) -> int:
        return self.C.shape[0]

    def discrete(self, Ts: float):
        """Zero-order hold discretisation through the augmented matrix."""
        n, m = self.A.shape[0], self.B.shape[1]
        aug = np.zeros((n + m, n + m))
        aug[:n, :n] = self.A
        aug[:n, n:] = self.B
        E = expm(aug * Ts)
        return E[:n, :n], E[:n, n:], self.C.copy(), self.D.copy()


def _double_integrator() -> Plant:
    """Unit-mass point mass on a frictionless track: input is force."""
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    C = np.array([[1.0, 0.0]])
    return Plant(
        key="double_integrator",
        name="双积分器（无阻尼质量块）",
        A=A, B=B, C=C, D=np.zeros((1, 1)),
        state_names=["位置 x (m)", "速度 v (m/s)"],
        output_names=["位置 x (m)"],
        input_name="推力 u (N)",
        x0=np.array([1.0, 0.0]),
        u_limit=4.0,
        default_controller="mpc",
    )


def _mass_spring_damper() -> Plant:
    """Classic second-order oscillator, m = 1, k = 1, c = 0.25."""
    m, k, c = 1.0, 1.0, 0.25
    A = np.array([[0.0, 1.0], [-k / m, -c / m]])
    B = np.array([[0.0], [1.0 / m]])
    C = np.array([[1.0, 0.0]])
    return Plant(
        key="mass_spring",
        name="弹簧-质量-阻尼（欠阻尼振子）",
        A=A, B=B, C=C, D=np.zeros((1, 1)),
        state_names=["位移 x (m)", "速度 v (m/s)"],
        output_names=["位移 x (m)"],
        input_name="外力 u (N)",
        x0=np.array([0.0, 0.0]),
        u_limit=5.0,
        default_controller="lqr",
    )


def _dc_motor() -> Plant:
    """Armature-controlled DC motor, position output, first-order lag in speed.

    Simplified to a second-order model: theta'' = -a*theta' + b*u, with the
    mechanical time constant folded into ``a``.
    """
    a, b = 5.0, 5.0
    A = np.array([[0.0, 1.0], [0.0, -a]])
    B = np.array([[0.0], [b]])
    C = np.array([[1.0, 0.0]])
    return Plant(
        key="dc_motor",
        name="直流电机（位置伺服）",
        A=A, B=B, C=C, D=np.zeros((1, 1)),
        state_names=["转角 θ (rad)", "角速度 ω (rad/s)"],
        output_names=["转角 θ (rad)"],
        input_name="电枢电压 u (V)",
        x0=np.array([1.0, 0.0]),
        u_limit=5.0,
        default_controller="pid",
    )


def _inverted_pendulum() -> Plant:
    """Cart-pole, linearised about the upright equilibrium.

    Derived from
        (M+m)x'' + m*l*theta'' = u
        x'' + l*theta'' = g*theta
    which gives
        x''     = (-m*g*theta + u) / M
        theta'' = ((M+m)*g*theta - u) / (M*l)

    The unstable pole sits near sqrt((M+m)g/(M l)) ~ 5.4 rad/s, i.e. a
    doubling time of about 0.13 s -- fast enough that a short horizon visibly
    struggles, which is the point of the demo.
    """
    M, m, l, g = 1.0, 0.2, 0.4, 9.81
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -m * g / M, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, (M + m) * g / (M * l), 0.0],
    ])
    B = np.array([[0.0], [1.0 / M], [0.0], [-1.0 / (M * l)]])
    C = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    return Plant(
        key="inverted_pendulum",
        name="倒立摆（小车-摆杆，线性化）",
        A=A, B=B, C=C, D=np.zeros((2, 1)),
        state_names=["小车位置 x (m)", "小车速度 (m/s)", "摆角 θ (rad)", "角速度 θ̇ (rad/s)"],
        output_names=["小车位置 x (m)", "摆角 θ (rad)"],
        input_name="水平推力 u (N)",
        x0=np.array([0.0, 0.0, 0.15, 0.0]),
        u_limit=2.0,
        default_controller="mpc",
    )


_FACTORIES = {
    "double_integrator": _double_integrator,
    "mass_spring": _mass_spring_damper,
    "dc_motor": _dc_motor,
    "inverted_pendulum": _inverted_pendulum,
}


def get_plant(key: str) -> Plant:
    try:
        factory = _FACTORIES[key]
    except KeyError:
        raise KeyError(f"unknown plant '{key}'; available: {sorted(_FACTORIES)}") from None
    return factory()


def list_plants() -> List[Dict]:
    """Serialisable summary for the HTTP layer."""
    out = []
    for key, factory in _FACTORIES.items():
        p = factory()
        out.append({
            "key": p.key,
            "name": p.name,
            "n_states": p.n_states,
            "n_outputs": p.n_outputs,
            "state_names": p.state_names,
            "output_names": p.output_names,
            "input_name": p.input_name,
            "u_limit": p.u_limit,
            "default_controller": p.default_controller,
            "unstable": bool(np.max(np.real(np.linalg.eigvals(p.A))) > 1e-9),
        })
    return out


PLANT_KEYS = tuple(_FACTORIES)
