"""
Utils for plotting

"""
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

from . import models


def no_mixing(
    axis: plt.Axes,
    val: float,
    fmt: str = "r--",
    label: str = None,
) -> None:
    """
    Plot no mixing "fit" on an axis - i.e. a horizontal line at val

    :param axis: axis to plot on
    :param ratios:
    :param fmt: format string for the plot
    :param label: label to add to legend

    """
    pts = np.linspace(*axis.get_xlim())
    axis.plot(pts, val * np.ones_like(pts), fmt, label=label)


def no_constraints(
    axis: plt.Axes,
    params: Tuple[float, float, float],
    fmt: str = "r--",
    label: str = None,
) -> None:
    """
    Plot unconstrained mixing fit on an axis.
    Uses the existing axis limits as the plotting range

    :param axis: axis to plot on
    :param params: parameters (a, b, c) as defined in models.no_constraints
    :param fmt: format string for the plot
    :param label: label to add to legend

    """
    pts = np.linspace(*axis.get_xlim())
    axis.plot(pts, models.no_constraints(pts, *params), fmt, label=label)
