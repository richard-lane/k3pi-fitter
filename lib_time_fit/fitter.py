"""
Decay time fitters

"""
from typing import Tuple
import numpy as np
from iminuit import Minuit

from . import util, models


def weighted_mean(points: np.ndarray, errors: np.ndarray) -> Tuple[float, float]:
    """
    Weighted mean and its error of some points

    :param points: values
    :param errors: errors on the values
    :returns: the mean
    :returns: error on the mean

    """
    weights = 1 / errors**2

    return np.average(points, weights=weights), 1 / np.sum(weights)


def no_mixing(ratio: np.ndarray, err: np.ndarray) -> Tuple[float, float]:
    r"""
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

    mean, err_on_mean = weighted_mean(ratio, err)

    amp_ratio = np.sqrt(mean)
    amp_ratio_err = 0.5 * err_on_mean / np.sqrt(mean)

    return amp_ratio, amp_ratio_err


def no_constraints(
    ratio: np.ndarray,
    errs: np.ndarray,
    bins: np.ndarray,
    initial_guess: util.MixingParams,
) -> Minuit:
    """
    Fit WS/RS decay time ratio with no constraints on parameters

    Uses models.no_constraints as the mode.

    :param ratio: ratio of WS/RS decay times in each time bin
    :param errs: error on the ratio of WS/RS decay times in each time bin
    :param bins: bins used when calculating the ratio.
                 Should contain each left bin edge, plus the rightmost.
    :param initial_guess: inital guess at the parameters when fitting.

    :returns: Minuit fitter after the fit


    """
    assert len(ratio) == len(bins) - 1

    chi2 = models.NoConstraints(ratio, errs, bins)

    minimiser = Minuit(chi2, **initial_guess._asdict())

    minimiser.migrad()

    return minimiser
