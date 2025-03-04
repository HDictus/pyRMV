from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from analysis_neuro.io import load_result, save_result
from analysis_neuro.testing import assert_results_equal



def test_save_and_load_figures_and_measurement():
    result, measurement = create_test_results()
    with TemporaryDirectory() as tdir:
        savepath = Path(tdir) / "result"
        save_result(result, savepath)
        loaded_result = load_result(savepath)
        assert_results_equal(result, loaded_result)
    return


def test_equality_assertion_ignores_ordering():
    result, measurement = create_test_results()
    with TemporaryDirectory() as tdir:
        savepath = Path(tdir) / "result"
        save_result(result, savepath)
        loaded_result = load_result(savepath)
        result['measurement'] = measurement.loc[np.flipud(measurement.index)]
        assert_results_equal(result, loaded_result)
        result['measurement'] = measurement[np.flipud(measurement.columns)]
        assert_results_equal(result, loaded_result)


def create_test_results():
    measurement = pd.DataFrame(
        {"aparam": ["a", "b", "c", "d", "e"], "ameasurement": [1, 2, 3, 4, 5]}
    )
    fig, ax = plt.subplots()
    measurement.hist("ameasurement", ax=ax)
    result = {
        "measurement": measurement,
        "figures": {"afigure": fig},
        "Introduction": "abcdefg",
        "stats": {'a hypothesis with (\arbitr$ary ch@ract/ers':
                  measurement.assign(pvalue=1)},
    }
    return result, measurement