"""Functions for extracting features from lower-level features."""
from scipy import optimize
import numpy as np
import pandas as pd


def frequency_modulation_amplitude(time, signal, frequency):
    """Return the amplitude of frequency modulation at the specified frequency."""
    signal = signal - signal.mean()

    def sin_at_freq(time, offset, amplitude):
        return np.sin((time * 2 * np.pi * frequency) + offset) * amplitude

    popt, _ = optimize.curve_fit(sin_at_freq, time, signal)  # pylint: disable=W0632
    amp = popt[1]
    return np.abs(amp)


# pylint: disable=fixme
# TODO: test
def g_OSI_signal(response, ori_degrees, groupby):  # pylint: disable=invalid-name
    """Return the generalized orientation selectivity index signal.

    Arguments:
        response: array of neural responses
        ori_degrees: array of stimulus orientations in degrees
        groupby: grouping variable (e.g. cell ids)

    Returns:
        Series of orientation selectivity index values per group
    """
    df = pd.DataFrame({'resp': response,
                       'ori': np.deg2rad(ori_degrees)})
    df['scaling'] = np.exp(2 * 1j * df['ori'])
    df['scaled_resp'] = df['resp'] * df['scaling']
    grouped = df.groupby(groupby)
    return (grouped['scaled_resp'].sum() / grouped['resp'].sum()).abs()
