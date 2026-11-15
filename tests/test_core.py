"""Self-contained checks for the control core.

Run with ``python tests/test_core.py``. No pytest dependency: the file is a
tiny runner so anyone who clones the repo can verify the maths without a
package install beyond NumPy.

Each case states what it proves. A green run means the numbers in the UI rest
on verified code rather than on a plausible-looking integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.controllers import build_controller, pid_defaults  # noqa: E402
from core.linalg import dare, expm  # noqa: E402
from core.models import get_plant, list_plants  # noqa: E402
from core.qp import BoxQP  # noqa: E402
from server.engine import TS, Simulation  # noqa: E402

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


# --------------------------------------------------------------------- linalg
@case
def test_expm_matches_known_values():
    """expm of a nilpotent matrix and of a rotation must match closed forms."""
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    E = expm(A * 0.5)
    assert np.allclose(E, [[1.0, 0.5], [0.0, 1.0]]), E

    theta = np.pi / 2
    R = np.array([[0.0, -1.0], [1.0, 0.0]]) * theta
    E = expm(R)
    assert np.allclose(E, [[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]], atol=1e-12), E


@case
def test_expm_of_zero_and_diagonal():
    assert np.allclose(expm(np.zeros((3, 3))), np.eye(3))
    D = np.diag([-1.0, 2.0])
    assert np.allclose(expm(D), np.diag([np.exp(-1.0), np.exp(2.0)]))


@case
def test_discretisation_preserves_eigenvalues():
    """ZOH maps continuous poles to exp(p*Ts); a stable plant stays stable."""
    plant = get_plant("mass_spring")
    Ad, Bd, C, _ = plant.discrete(0.02)
    lam_c = np.linalg.eigvals(plant.A)
    lam_d = np.linalg.eigvals(Ad)
    assert np.allclose(np.sort_complex(np.exp(lam_c * 0.02)),
                       np.sort_complex(lam_d), atol=1e-10)
    assert np.max(np.abs(lam_d)) < 1.0, "discretised stable plant must be inside the unit circle"
    assert Ad.shape == (2, 2) and Bd.shape == (2, 1) and C.shape == (1, 2)


# -------------------------------------------------------------------- riccati
@case
def test_dare_satisfies_the_riccati_equation():
    """Residual of the algebra, not just 'it did not crash'."""
    plant = get_plant("inverted_pendulum")
    Ad, Bd, _, _ = plant.discrete(TS)
    Q = np.diag([6.0, 0.2, 30.0, 0.2])
    R = np.eye(1)
    P, K, iters = dare(Ad, Bd, Q, R)

    lhs = P
    rhs = Ad.T @ P @ Ad - (Ad.T @ P @ Bd) @ np.linalg.solve(Bd.T @ P @ Bd + R, Bd.T @ P @ Ad) + Q
    assert np.max(np.abs(lhs - rhs)) < 1e-8, np.max(np.abs(lhs - rhs))
    assert np.allclose(K, np.linalg.solve(Bd.T @ P @ Bd + R, Bd.T @ P @ Ad))
    assert iters > 0


@case
def test_lqr_closed_loop_is_stable():
    """Every eigenvalue of Ad - Bd K must sit inside the unit circle."""
    for key in ("double_integrator", "mass_spring", "dc_motor", "inverted_pendulum"):
        plant = get_plant(key)
        ctrl = build_controller("lqr", plant, TS, {})
        Ad, Bd, _, _ = plant.discrete(TS)
        eig = np.linalg.eigvals(Ad - Bd @ ctrl.K)
        assert np.max(np.abs(eig)) < 1.0 - 1e-9, (key, np.abs(eig))


@case
def test_dare_reports_failure_on_unstabilisable_pair():
    """A plant that cannot be stabilised must raise, not silently return junk."""
    A = np.array([[1.5, 0.0], [0.0, 1.0]])
    B = np.array([[0.0], [0.0]])       # no authority over the unstable mode
    try:
        dare(A, B, np.eye(2), np.eye(1), max_iter=40)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError for an unstabilisable pair")


# ------------------------------------------------------------------------- qp
def _kkt_holds(x, P, q, lb, ub, tol=2e-3):
    """Check the optimality conditions of a box-constrained QP."""
    g = P @ x + q
    for i in range(len(x)):
        at_lb = ub[i] - lb[i] > 1e-12 and abs(x[i] - lb[i]) < 1e-7
        at_ub = ub[i] - lb[i] > 1e-12 and abs(x[i] - ub[i]) < 1e-7
        if at_lb and g[i] < -tol:
            return False, f"i={i} at lower bound but gradient {g[i]:.4f} < 0"
        if at_ub and g[i] > tol:
            return False, f"i={i} at upper bound but gradient {g[i]:.4f} > 0"
        if not at_lb and not at_ub and abs(g[i]) > tol:
            return False, f"i={i} interior but gradient {g[i]:.4f} != 0"
    return True, ""


@case
def test_box_qp_interior_solution():
    """With no active bounds the answer is the unconstrained minimiser -P^-1 q."""
    P = np.diag([2.0, 2.0])
    q = np.array([-1.0, -1.0])
    x, info = BoxQP(P).solve(q, -10.0, 10.0)
    assert np.allclose(x, [0.5, 0.5], atol=1e-6), x
    ok, why = _kkt_holds(x, P, q, np.full(2, -10.0), np.full(2, 10.0))
    assert ok, why


@case
def test_box_qp_respects_and_activates_bounds():
    """Both coordinates saturate: the solver must clamp and satisfy KKT there."""
    P = np.eye(2)
    q = np.array([-1.0, -2.0])
    x, info = BoxQP(P).solve(q, 0.0, 1.0)
    assert np.allclose(x, [1.0, 1.0], atol=1e-6), x
    ok, why = _kkt_holds(x, P, q, np.zeros(2), np.ones(2))
    assert ok, why


@case
def test_box_qp_mixed_active_set():
    """One bound active, one interior: the classic partially constrained case."""
    P = np.diag([2.0, 1.0])
    q = np.array([-3.0, -1.0])
    lb, ub = np.zeros(2), np.ones(2)
    x, info = BoxQP(P).solve(q, lb, ub)
    assert np.allclose(x, [1.0, 1.0], atol=1e-6), x
    ok, why = _kkt_holds(x, P, q, lb, ub)
    assert ok, why


@case
def test_box_qp_scales_to_a_control_horizon():
    """A 30-variable problem (N = 30, single input) must still satisfy KKT."""
    rng = np.random.default_rng(3)
    M = rng.normal(size=(30, 30))
    P = M @ M.T + np.eye(30)
    q = rng.normal(size=30)
    lb, ub = np.full(30, -2.0), np.full(30, 2.0)
    x, info = BoxQP(P, rho=1.0, max_iter=400).solve(q, lb, ub)
    ok, why = _kkt_holds(x, P, q, lb, ub, tol=1e-2)
    assert ok, why
    # The constrained optimum cannot beat the unconstrained one.
    x_un = np.linalg.solve(P, -q)
    f = lambda v: 0.5 * v @ P @ v + q @ v
    assert f(x) >= f(x_un) - 1e-8


@case
def test_box_qp_handles_a_fixed_variable():
    """lb == ub pins a variable; that must not blow up the solver."""
    P = np.eye(3)
    q = np.array([-1.0, -1.0, -1.0])
    lb = np.array([0.0, 0.5, -5.0])
    ub = np.array([1.0, 0.5, 5.0])
    x, _ = BoxQP(P).solve(q, lb, ub)
    assert abs(x[1] - 0.5) < 1e-9, x
    assert np.allclose(x[[0, 2]], [1.0, 1.0], atol=1e-6), x


# -------------------------------------------------------------------- closure
def _run_closed_loop(plant_key, kind, seconds, params=None):
    plant = get_plant(plant_key)
    Ad, Bd, C, _ = plant.discrete(TS)
    ctrl = build_controller(kind, plant, TS, params or {})
    x = plant.x0.copy()
    u_max = 0.0
    hist = []
    for _ in range(int(seconds / TS)):
        u, _ = ctrl.compute(x, np.zeros(plant.n_outputs))
        u_max = max(u_max, float(np.max(np.abs(u))))
        x = Ad @ x + Bd @ u
        hist.append(x.copy())
        assert np.all(np.isfinite(x)), "state went non-finite"
    return plant, np.array(hist), u_max


@case
def test_lqr_stabilises_the_cart_pole():
    """The angle must end up essentially upright."""
    plant, hist, u_max = _run_closed_loop("inverted_pendulum", "lqr", 10.0)
    assert abs(hist[-1][2]) < 0.01, hist[-1]
    assert abs(hist[-1][0]) < 0.05, hist[-1]
    assert u_max <= plant.u_limit + 1e-9


@case
def test_mpc_stabilises_the_cart_pole_within_the_input_bound():
    """Same plant, same start, and u must never exceed the actuator limit."""
    plant, hist, u_max = _run_closed_loop("inverted_pendulum", "mpc", 10.0, {"horizon": 30})
    assert abs(hist[-1][2]) < 0.01, hist[-1]
    assert abs(hist[-1][0]) < 0.05, hist[-1]
    assert u_max <= plant.u_limit + 1e-9, (u_max, plant.u_limit)


@case
def test_pid_holds_the_angle_but_loses_the_cart():
    """This is the lesson, encoded as an assertion.

    A single loop on the angle keeps theta near zero while the cart walks away.
    If this ever stops being true the demo's premise has changed.
    """
    _, hist, _ = _run_closed_loop("inverted_pendulum", "pid", 8.0)
    assert abs(hist[-1][2]) < 0.02, hist[-1]
    assert abs(hist[-1][0]) > 2.0, f"cart should drift, ended at {hist[-1][0]:.3f}"


@case
def test_mpc_tracks_a_reference_on_the_double_integrator():
    plant = get_plant("double_integrator")
    Ad, Bd, C, _ = plant.discrete(TS)
    ctrl = build_controller("mpc", plant, TS, {"horizon": 30})
    x = np.zeros(2)
    target = np.array([1.5])
    for _ in range(int(8.0 / TS)):
        u, _ = ctrl.compute(x, target)
        x = Ad @ x + Bd @ u
    assert abs(x[0] - 1.5) < 0.02, x
    assert abs(x[1]) < 0.02, x


@case
def test_pid_defaults_exist_for_every_plant():
    for entry in list_plants():
        d = pid_defaults(entry["key"])
        assert {"kp", "ki", "kd", "direction"} <= set(d), entry["key"]


# --------------------------------------------------------------------- engine
@case
def test_simulation_stream_shape_and_bounds():
    sim = Simulation()
    # A single advance is capped so a stalled client cannot trigger a marathon
    # catch-up run; asking for more returns exactly the cap.
    samples = sim.advance(150)
    assert len(samples) == Simulation.MAX_STEPS, len(samples)
    keys = {"t", "y", "u", "x", "ref", "iters", "diverged"}
    assert keys <= set(samples[0])
    assert all(s["t"] > 0 for s in samples)
    assert all(samples[i]["t"] < samples[i + 1]["t"] for i in range(len(samples) - 1))
    assert all(abs(s["u"]) <= sim.u_limit + 1e-9 for s in samples)
    assert all(np.all(np.isfinite(s["x"])) for s in samples)


@case
def test_simulation_survives_a_hostile_configuration():
    """A deliberately unstable setup must report divergences, not emit NaN."""
    sim = Simulation({"plant": "inverted_pendulum", "controller": "pid",
                      "params": {"kp": 2.0, "ki": 0.0, "kd": 0.2, "direction": 1.0},
                      "u_limit": 0.5})
    sim.advance(400)
    assert sim.diverged >= 0
    st = sim.stats()
    assert all(np.isfinite(v) for v in st["x"])


@case
def test_observer_converges_under_measurement_noise():
    sim = Simulation({"plant": "inverted_pendulum", "controller": "lqr",
                      "observer": True, "noise": 0.01})
    for _ in range(200):
        sim.step()
    err = np.max(np.abs(sim.x_hat - sim.x))
    assert err < 0.05, f"observer error {err:.4f} too large"


@case
def test_disturbance_pulse_decays():
    """Push the cart, then confirm the loop returns to the equilibrium."""
    sim = Simulation({"plant": "inverted_pendulum", "controller": "mpc"})
    sim.advance(150)                      # settle first
    sim.inject_disturbance(1.8)
    sim.advance(400)
    assert abs(sim.x[2]) < 0.02, sim.x
    assert abs(sim.x[0]) < 0.1, sim.x


def main():
    failures = []
    for fn in CASES:
        name = fn.__name__
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures.append((name, exc or "assertion failed"))
            print(f"  FAIL  {name}: {exc or 'assertion failed'}")
        except Exception as exc:          # noqa: BLE001 - report, keep going
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"{len(failures)} of {len(CASES)} checks failed")
        return 1
    print(f"all {len(CASES)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
