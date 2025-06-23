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

def test_save_and_load_dfpointer():
    result, measurement = create_test_results()
    measurement['a df pointer'] = pd.DataFrame({'avalue': [3, 4]}).pointer()
    with TemporaryDirectory() as tdir:
        savepath = Path(tdir) / "result"
        save_result(result, savepath)
        loaded_result = load_result(savepath)
        assert_results_equal(result, loaded_result)

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


# TODO: there is a case where this doesn't quite work: duplicate index within different datasets
def test_equality_assertion_handles_duplicate():
    result, measurement = create_test_results()
    measurement = pd.concat([
        measurement, 
        measurement.loc[np.flipud(measurement.index)].reset_index()
    ])
    result['measurement'] = measurement
    with TemporaryDirectory() as tdir:
        savepath = Path(tdir) / 'result'
        save_result(result, savepath)
        result['measurement'] = measurement.iloc[np.flipud(np.arange(len(measurement)))]
        loaded_result = load_result(savepath)
        assert_results_equal(result, loaded_result)


# TODO: to allow models to 'volunteer' information, we need to be able to handle nans in this comparison


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