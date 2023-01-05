"""Tools for saving and loading results."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _append_path(path, resultdict):
    out = {}
    for key, value in resultdict.items():
        if isinstance(value, dict):
            out[key] = _append_path(path, value)
            continue
        if isinstance(value, str):
            if value.endswith(".csv") or value.endswith(".png"):
                out[key] = str(path / value)
                continue
        out[key] = value

    return out


def load_result(path):
    """Load an analysis report from the directory provided."""
    path = Path(path)
    with open(path / "dict.json", "r", encoding="utf8") as jsonfile:
        resultdict = json.load(jsonfile)
    resultdict = _append_path(path, resultdict)
    return resultdict


def _prepare_dict(result, path):
    resultdict = {}
    for key, value in result.items():
        if isinstance(value, pd.DataFrame):
            savepath = path / f"{key}.csv"
            value.to_csv(savepath)
            resultdict[key] = str(savepath.name)
        elif isinstance(value, plt.Figure):
            savepath = path / f"{key}.png"
            value.savefig(savepath)
            resultdict[key] = str(savepath.name)
        elif isinstance(value, dict):
            resultdict[key] = _prepare_dict(value, path)
        else:
            resultdict[key] = value
    return resultdict


def save_result(result, path):
    """Save an analysis report to the directoy provided."""
    path = Path(path)
    jsonpath = path / "dict.json"
    path.mkdir(exist_ok=True)
    resultdict = _prepare_dict(result, path)
    with open(jsonpath, "w", encoding="utf8") as jsonfile:
        json.dump(resultdict, jsonfile)
