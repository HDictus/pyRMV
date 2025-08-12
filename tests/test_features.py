import numpy as np
import pandas as pd
from analysis_neuro.features import frequency_modulation_amplitude, g_OSI_signal


def test_frequency_modulation_reproduces_amp():
    t = np.linspace(0, 10, 100)
    assert np.allclose(
        frequency_modulation_amplitude(t, 3*np.sin(t * np.pi * 2), 1),
        3)


def test_frequency_modulation_accounts_for_offset():
    t = np.linspace(0, 10, 100)
    assert np.allclose(
        frequency_modulation_amplitude(t, np.sin(t * np.pi * 2 + np.pi/4), 1),
        1, atol=0.01)


def test_frequency_modulation_low_for_wrong_freq():
    t = np.linspace(0, 10, 100)
    assert frequency_modulation_amplitude(t, np.sin(t * np.pi * 2), 2) < 1


def test_frequency_modulation_dstinguishes_noise():
    t = np.linspace(0, 10, 100)
    f1 = t * np.pi * 2
    amp1, amp2, amp3 = 1, 2, 0.5
    freq1, freq2, freq3 = 1, 2.5, 4
    signal = amp1 * np.sin(f1 * freq1) + amp2 * np.sin(f1 * freq2) + amp3 * np.sin(f1 * freq3)
    assert np.allclose(frequency_modulation_amplitude(t, signal, freq1), amp1)
    assert np.allclose(frequency_modulation_amplitude(t, signal, freq2), amp2)
    assert np.allclose(frequency_modulation_amplitude(t, signal, freq3), amp3, atol=0.001)


def test_gosi_distinguishes_sel_from_nonsel():
    pd.testing.assert_series_equal(
        g_OSI_signal([0, 1, 0, 1, 1, 1, 1, 1], [0, 90, 180, 270, 0, 90, 180, 270], [1, 1, 1, 1, 2, 2, 2, 2]),
        pd.Series({1: 1., 2: 0.}))
