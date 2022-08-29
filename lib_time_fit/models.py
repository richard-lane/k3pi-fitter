"""
Fit models

"""
import numpy as np
from iminuit import Minuit
from iminuit.util import make_func_code


def no_mixing(amplitude_ratio: float) -> float:
    """
    Model for no mixing - i.e. the ratio will be a constant (amplitude_ratio^2)
    Independent of time

    :param amplitude_ratio: ratio of DCS/CF amplitudes

    :returns: amplitude_ratio ** 2

    """

    return amplitude_ratio ** 2


def no_constraints(times: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Model for WS/RS time ratio; allows D0 mixing but does not constrain the mixing
    parameters to their previously measured values.
    Low time/small mixing approximation

    ratio = a^2 + abt + ct^2

    :param times: times to evaluate the ratio at, in lifetimes
    :param a: amplitude ratio
    :param b: mixing parameter from interference
    :param c: mixing parameter

    :returns: ratio at each time

    """
    return a ** 2 + a * b * times + c * times ** 2


def rs_integral(bins: np.ndarray) -> np.ndarray:
    """
    The integral of the RS model in each bin

    (Proportional to how many RS events we expect in each bin)

    :param bins: leftmost edge of each bin, plus the rightmost edge of the last bin. In lifetimes.
    :returns: integral of e^-t in each bin

    """
    return np.exp(-bins[:-1]) - np.exp(-bins[1:])


def _ws_integral_dispatcher(
    times: np.ndarray, a: float, b: float, c: float
) -> np.ndarray:
    """
    Indefinite integral evaluated at each time - constant term assumed to be 0

    """
    return -np.exp(-times) * (
        (a ** 2) + a * b * (times + 1) + (times * (times + 2) + 2) * c
    )


def ws_integral(bins: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    The integral of the WS model in each bin

    (Proportional to how many WS events we expect in each bin -
    same constant of proportionality as the RS integral)

    :param bins: leftmost edge of each bin, plus the rightmost edge of the last bin. In lifetimes.
    :returns: integral of e^-t in each bin

    """
    return _ws_integral_dispatcher(bins[1:], a, b, c) - _ws_integral_dispatcher(
        bins[:-1], a, b, c
    )


class NoConstraints:
    """
    Cost function for the fitter without constraints

    """

    errordef = Minuit.LEAST_SQUARES

    def __init__(self, ratio: np.ndarray, error: np.ndarray, bins: np.ndarray):
        """
        Set parameters for doing a fit without constraints

        :param ratio: WS/RS ratio
        :param error: error in ratio
        :param bins: bins used when finding the ratio

        """
        self.ratio = ratio
        self.error = error
        self.bins = bins

        # We need to tell Minuit what our function signature is explicitly
        self.func_code = make_func_code(["a", "b", "c"])

        # Denominator (RS) integral doesn't depend on the params
        # so we only need to evaluate it once
        self._expected_rs_integral = rs_integral(bins)

    def _expected_ws_integral(self, a: float, b: float, c: float) -> np.ndarray:
        """
        Given our parameters, find the expected WS integral

        """
        return ws_integral(self.bins, a, b, c)

    def __call__(self, a: float, b: float, c: float):
        """
        Evaluate the chi2 given our parameters

        """
        expected_ratio = (
            self._expected_ws_integral(a, b, c) / self._expected_rs_integral
        )

        return np.sum(((self.ratio - expected_ratio) / self.error) ** 2)
