from scipy import optimize
import numpy as np
import pandas as pd


def frequency_modulation_amplitude(time, signal, frequency):
    """Returns the amplitude of frequency modulation at the specified frequency"""
    signal = signal - signal.mean()
    def sin_at_freq(time, offset, amplitude):
        return np.sin((time * 2 * np.pi * frequency) + offset) * amplitude
    (offst, amp), _ = optimize.curve_fit(sin_at_freq, time, signal)
    return np.abs(amp)


# TODO: test
def g_OSI_signal(response, ori_degrees, groupby, null_shuffle=False):
    df = pd.DataFrame({'resp': response,
                       'ori': np.deg2rad(ori_degrees)})
    df['scaling'] = np.exp(2 * 1j * df['ori'])
    df['scaled_resp'] = df['resp'] * df['scaling']
    grouped = df.groupby(groupby)
    return (grouped['scaled_resp'].sum() / grouped['resp'].sum()).abs()
    