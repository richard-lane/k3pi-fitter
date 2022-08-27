"""
Fit models

"""
import numpy as np


def no_mixing(amplitude_ratio: float) -> float:
    """
    Model for no mixing - i.e. the ratio will be a constant (amplitude_ratio^2)

    :param amplitude_ratio: ratio of DCS/CF amplitudes

    :returns: amplitude_ratio ** 2

    """

    return amplitude_ratio ** 2
