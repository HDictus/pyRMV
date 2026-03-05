"""Tools for saving and loading results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import dataframe_pointer
import pandas as pd
from rst2pdf.createpdf import RstToPdf

DPI = 256


def _make_safe_path(filename):
    return "".join(
        [c for c in filename if c.isalpha() or c.isdigit() or c == " "]
    ).rstrip()


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
            name = _save_dataframe(path, key, value)
            resultdict[key] = name
        elif isinstance(value, plt.Figure):
            safe_path = path / f"{_make_safe_path(key)}.png"
            value.savefig(safe_path, dpi=DPI)
            resultdict[key] = str(safe_path.name)
        elif isinstance(value, dict):
            resultdict[key] = _prepare_dict(value, path)
        else:
            resultdict[key] = value
    return resultdict


def _save_dataframe(path, key, df):
    safe_path = path / f"{_make_safe_path(key)}.csv"
    df = df.copy()
    _replace_df_pointers(df, path)
    df.to_csv(safe_path)
    return safe_path.name


def _replace_df_pointers(df, path):
    for col in df.columns:
        if df[col].dtype == object:
            for i, v in df[col].items():
                if isinstance(v, dataframe_pointer.DFPointer):
                    safe_path = path / f"{_make_safe_path(repr(v))}.csv"
                    df.loc[i, col] = str(safe_path.name)
                    v.df.to_csv(safe_path)


def save_result(result, path):
    """Save an analysis report to the directory provided."""
    path = Path(path)
    jsonpath = path / "dict.json"
    path.mkdir(exist_ok=True)
    resultdict = _prepare_dict(result, path)
    print(resultdict)
    with open(jsonpath, "w", encoding="utf8") as jsonfile:
        json.dump(resultdict, jsonfile, indent=4)


def _rst_table(df):
    return "TODO: convert dataframes to tables"


def save_pdf(result, path):
    figures = ""# "\n".join([ for name in result['figures']])
    for name, fig in result['figures'].items():
        fn = path.parent / f'{name}.png'
        fig.savefig(fn)
        figures += f".. image:: {str(fn.absolute())}\n"
    text = f"""
Introduction
============
{result["Introduction"]}

Results
=======
{_rst_table(result['measurement'])}

{figures}


Conclusions
===========
{_rst_table(result['stats'])}
"""

    RstToPdf().createPdf(text=text, output=str(path))
