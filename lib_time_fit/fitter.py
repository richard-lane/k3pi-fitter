"""
Decay time fitters

"""
from typing import Tuple
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares

from . import util, models


def no_mixing(ratio: np.ndarray, err: np.ndarray) -> Tuple[float, float]:
    """
    Find the best value that describes our dataset if we assume no mixing;
    i.e. that the ratio is described by r_D^2.

    We don't actually need to do a fit here - we can find the optimal
    \chi^2 value analytically - it's just the weighted mean

    :param ratio: binned ratio of DCS/CF decay times
    :param err: error on the ratio

    :returns: the value of r_D that best describes the data
    :returns: the error on r_D

    """
    assert len(ratio) == len(err)

    weights = 1 / err ** 2

    weighted_mean = np.average(ratio, weights=weights)
    err_on_mean = 1 / np.sum(weights)

    amp_ratio = np.sqrt(weighted_mean)
    amp_ratio_err = 0.5 * err_on_mean / np.sqrt(weighted_mean)

    return amp_ratio, amp_ratio_err
