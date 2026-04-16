import numpy as np
import cvxpy as cp
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CoordinateBounds:
    index: int
    min_value: float
    max_value: float
    min_point: np.ndarray
    max_point: np.ndarray


@dataclass
class EpochBoundsResult:
    epoch: np.ndarray
    tolerance: float
    s_min: float
    s_max: float
    bounds: List[CoordinateBounds]


def _validate_inputs(epoch: np.ndarray, s_min: float, s_max: float, tolerance: float) -> None:
    if epoch.ndim != 1:
        raise ValueError("epoch must be a 1D vector")
    if len(epoch) == 0:
        raise ValueError("epoch must be non-empty")
    if np.linalg.norm(epoch) == 0:
        raise ValueError("epoch vector must be non-zero")
    if s_min < 0:
        raise ValueError("s_min must be >= 0")
    if s_max < s_min:
        raise ValueError("s_max must be >= s_min")
    if not (0 <= tolerance <= 1):
        raise ValueError("tolerance must be in [0, 1]")


def _solve_single_objective_socp(
    epoch: np.ndarray,
    s_min: float,
    s_max: float,
    tolerance: float,
    coord_index: int,
    maximize: bool,
    solver: str = "ECOS",
) -> tuple[float, np.ndarray]:
    """
    Solves:
        min/max x_i
        subject to:
            x >= 0
            s_min <= sum(x) <= s_max
            dot(epoch, x) >= tolerance * ||epoch|| * ||x||

    Implemented as SOCP with auxiliary scalar t:
        ||x||_2 <= t
        tolerance * ||epoch|| * t <= dot(epoch, x)
    """
    n = len(epoch)
    epoch_norm = float(np.linalg.norm(epoch))

    x = cp.Variable(n, nonneg=True)
    t = cp.Variable(nonneg=True)

    constraints = [
        cp.sum(x) >= s_min,
        cp.sum(x) <= s_max,
        cp.norm(x, 2) <= t,
        tolerance * epoch_norm * t <= epoch @ x,
    ]

    objective = cp.Maximize(x[coord_index]) if maximize else cp.Minimize(x[coord_index])
    problem = cp.Problem(objective, constraints)

    problem.solve(solver=solver)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(
            f"SOCP solve failed for coordinate {coord_index}, maximize={maximize}. "
            f"Status: {problem.status}"
        )

    return float(x.value[coord_index]), np.array(x.value, dtype=float)


def solve_epoch_bounds_socp(
    epoch: List[float],
    s_min: float,
    s_max: float,
    tolerance: float,
    solver: str = "ECOS",
) -> EpochBoundsResult:
    """
    Compute min/max for every coordinate for one epoch in the continuous case.
    """
    epoch_arr = np.asarray(epoch, dtype=float)
    _validate_inputs(epoch_arr, s_min, s_max, tolerance)

    bounds: List[CoordinateBounds] = []

    for i in range(len(epoch_arr)):
        min_value, min_point = _solve_single_objective_socp(
            epoch=epoch_arr,
            s_min=s_min,
            s_max=s_max,
            tolerance=tolerance,
            coord_index=i,
            maximize=False,
            solver=solver,
        )

        max_value, max_point = _solve_single_objective_socp(
            epoch=epoch_arr,
            s_min=s_min,
            s_max=s_max,
            tolerance=tolerance,
            coord_index=i,
            maximize=True,
            solver=solver,
        )

        bounds.append(
            CoordinateBounds(
                index=i,
                min_value=min_value,
                max_value=max_value,
                min_point=min_point,
                max_point=max_point,
            )
        )

    return EpochBoundsResult(
        epoch=epoch_arr,
        tolerance=tolerance,
        s_min=s_min,
        s_max=s_max,
        bounds=bounds,
    )

# epoch = [23, 3 ,13, 16]
epoch = [10,0,0,0]
s_max = sum(epoch)
s_min = 0.8 * s_max

tolerance = 0.99

result = solve_epoch_bounds_socp(
    epoch=epoch,
    s_min=s_min,
    s_max=s_max,
    tolerance=tolerance,
    solver="ECOS",   # или "SCS"
)

for item in result.bounds:
    print(f"Coordinate {item.index}")
    print(f"  min = {int(item.min_value)}")
    print(f"  max = {int(item.max_value)}")

    min_point_int = [int(x) for x in item.min_point]
    max_point_int = [int(x) for x in item.max_point]

    print(f"  argmin x = {min_point_int}")
    print(f"  argmax x = {max_point_int}")

    print(f"  argmin x = {item.min_point}")
    print(f"  argmax x = {item.max_point}")

    print()

for item in result.bounds:
    print()
    print(f"{[int(x) for x in item.max_point]}, ")
