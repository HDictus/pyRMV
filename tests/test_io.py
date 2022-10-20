from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import pandas as pd

from analysis_neuro.io import load_result, save_result
from analysis_neuro.testing import assert_results_equal


def test_save_and_load_figures_and_measurement():
    measurement = pd.DataFrame(
        {"aparam": ["a", "b", "c", "d", "e"], "ameasurement": [1, 2, 3, 4, 5]}
    )
    fig, ax = plt.subplots()
    measurement.hist("ameasurement", ax=ax)
    result = {"measurement": measurement, "figures": {"afigure": fig},
              'Introduction': 'abcdefg'}
    with TemporaryDirectory() as tdir:
        savepath = Path(tdir) / "result"
        save_result(result, savepath)
        loaded_result = load_result(savepath)
        assert_results_equal(result, loaded_result)
    return
