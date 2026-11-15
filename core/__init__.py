"""Pure-Python control core: models, controllers, QP and Riccati solvers."""

from .models import Plant, get_plant, list_plants
from .controllers import build_controller, PID, LQR, MPC
from .qp import BoxQP, solve_box_qp
from .linalg import expm, dare

__all__ = [
    "Plant", "get_plant", "list_plants",
    "build_controller", "PID", "LQR", "MPC",
    "BoxQP", "solve_box_qp", "expm", "dare",
]
