"""Tools for the testing of analyses."""
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _assert_figure_equal(new, old_path):
    with TemporaryDirectory() as tmpd:
        newfile = Path(tmpd) / "new.png"
        new.savefig(newfile)
        diff = plt.imread(newfile) - plt.imread(old_path)
        rms = np.mean(diff**2)
        are_same = rms < 0.02
        if not are_same:
            savedpath = Path() / ".testing"
            savedpath.mkdir(exist_ok=True)
            shutil.copy(old_path, savedpath / "old.png")
            shutil.copy(newfile, savedpath / "new.png")
            raise AssertionError(f"plots are not the same, see: {savedpath}")


def assert_results_equal(new_result, stored_result):
    """assert that some result dict is equal to a stored one"""
    print(new_result, stored_result)
    assert set(new_result.keys()) == set(stored_result.keys())
    for k, val in stored_result.items():
        if isinstance(val, dict):
            assert_results_equal(new_result[k], val)
            continue
        if isinstance(val, str):
            if val.endswith(".csv"):
                pd.testing.assert_frame_equal(new_result[k], pd.read_csv(val, index_col=0))
                continue
            if val.endswith(".png"):
                _assert_figure_equal(new_result[k], val)
                continue
        assert type(val) == type(new_result[k])
        assert val == new_result[k], f"{val} != {new_result[k]}"
