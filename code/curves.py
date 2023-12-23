import numpy as np
from scipy.interpolate import CubicSpline

import constants


# Function that creates a step reference curve between two points
def step(*, start=None, end=None, steps=constants.TT):
    _check_point(start)
    _check_point(end)

    if start.shape[0] != end.shape[0]:
        raise ValueError("start and end must have the same size")

    # Initialization of the reference curve
    ref = np.zeros((start.shape[0], steps))

    # Definition of the reference curve
    for i in range(steps):
        if i < (steps / 2):
            ref[:, i] = start
        else:
            ref[:, i] = end

    return ref


# Function that creates a smooth reference curve using two equilibrium points
def cubic_spline(*, start=None, end=None, steps=constants.TT):
    _check_point(start)
    _check_point(end)

    if start.shape[0] != end.shape[0]:
        raise ValueError("start and end must have the same size")

    final_time = 3
    x = np.array([0, final_time])

    ref = np.zeros((start.shape[0], steps))
    for i in range(start.shape[0]):
        # Given points
        y = start[i], end[i]

        # Interpolate using CubicSpline
        interp = CubicSpline(x, y, bc_type="clamped")

        # Generate points for the interpolated polynomial
        x_interp = np.linspace(0, final_time, steps // 10)
        y_interp = interp(x_interp)

        ref1 = np.tile(start[i], (steps // 2) - (len(y_interp) // 2))
        ref2 = np.tile(end[i], (steps // 2) - (len(y_interp) // 2))
        ref[i, :] = np.concatenate((ref1, y_interp, ref2), axis=0)

    return ref


def _check_point(point):
    if point is None:
        raise ValueError("point must be specified")

    if not isinstance(point, np.ndarray):
        raise TypeError("point must be a numpy array")

    if point.ndim != 1:
        raise ValueError("point must be a 1D array")
