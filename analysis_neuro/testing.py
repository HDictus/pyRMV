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
        oldim = plt.imread(old_path)
        dpi = int(np.ceil((oldim.shape[1] / new.get_size_inches())[0]))
        new.savefig(newfile, dpi=dpi)
        errmsg = ""
        try:
            diff = plt.imread(newfile) - oldim
            rms = np.mean(diff**2)
            are_same = rms < 0.02
        except ValueError as verr:
            are_same = False
            errmsg = str(verr)
        if not are_same:
            savedpath = Path() / ".testing"
            savedpath.mkdir(exist_ok=True)
            shutil.copy(old_path, savedpath / "old.png")
            shutil.copy(newfile, savedpath / "new.png")
            raise AssertionError(
                f"{errmsg}\nplots are not the same, see: {savedpath.resolve()}"
            )


# pylint: disable=C0123
def assert_results_equal(new_result, stored_result):
    """Assert that some result dict is equal to a stored one."""
    assert set(new_result.keys()) == set(stored_result.keys())
    for k, val in stored_result.items():
        if isinstance(val, dict):
            assert_results_equal(new_result[k], val)
            continue
        if isinstance(val, str):
            if val.endswith(".csv"):
                new = new_result[k]
                _compare_dataframe(new, val)
                continue
            if val.endswith(".png"):
                _assert_figure_equal(new_result[k], val)
                continue
        # noqa
        assert type(val) == type(
            new_result[k]
        ), f"{type(val)} =/= {type(new_result[k])}"
        assert val == new_result[k], f"{val} != {new_result[k]}"


def _compare_dataframe(new, val):
    nlevels = new.index.nlevels
    if nlevels > 1:
        old = pd.read_csv(val, index_col=list(range(nlevels)))
        old.index = pd.MultiIndex.from_arrays(
            [
                old.index.get_level_values(n).astype(
                    new.index.get_level_values(n).dtype
                )
                for n in range(nlevels)
            ],
            names=new.index.names,
        )
    else:
        old = pd.read_csv(val, index_col=0)
    new = new.sort_index(axis=1).reset_index(drop=False)
    old = old.sort_index(axis=1).reset_index(drop=False)
    new = new.sort_values(list(new.columns)).reset_index(drop=True)
    old = old.sort_values(list(old.columns)).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        new, old
    )
