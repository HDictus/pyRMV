"""Plotting tools."""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis_neuro import terminology as terms
from analysis_neuro.stats import _iter_compare


def crossplot(data, dependent, independent, compare):
    """Compare datasets with a crossplot.

    Plot the dependent variable for each of the compared datasets
    along a different axis, showing the level of agreement between them by
    how close they are to the diagonal.
    """
    # pylint: disable=invalid-name
    # pylint: disable=unused-argument
    figs = {}
    comparevalues = data[compare].unique()

    if len(comparevalues) < 2:
        raise ValueError(
            "a crossplot only makes sense when comparing data. "
            "It is not defined for less than 2 datasets."
        )

    for l1, data1, l2, data2 in _iter_compare(data, compare):
        fig, ax = plt.subplots()
        figs[f"{l1}-{l2}"] = fig
        data1 = data1.sort_values(independent)
        data2 = data2.sort_values(independent)
        x = data1[dependent].values

        y = data2[dependent].values
        ax.scatter(x, y)
        plt.xlabel(l1)
        plt.ylabel(l2)
        plt.title(dependent)
        minimum = max(ax.get_xlim()[0], ax.get_ylim()[0])
        maximum = min(ax.get_xlim()[1], ax.get_ylim()[1])
        plt.plot((minimum, maximum), (minimum, maximum), color="gray", linestyle="--")

    return figs


def pathway_heatmap(
    data: pd.DataFrame, dependent: str, independent: List[str], compare: str
):
    """Create a heatmap for connectivity data.

    data: all measurements
    dependent: dependent variable to plot
    independent: independent variables, must include some prefixed with PRESYNAPTIC and POSTSYNAPTIC
    compare: variable to compare between. Creates a separate plot for each value of this variable.
    """
    out = {}
    for label, dataframe in data.groupby(compare):
        pre_cols = [col for col in independent if col.startswith(terms.PRESYNAPTIC)]
        post_cols = [col for col in independent if col.startswith(terms.POSTSYNAPTIC)]
        pivot = dataframe.pivot_table(
            index=pre_cols, columns=post_cols, values=dependent
        )
        shape = np.array(pivot.shape)
        fig, axes = plt.subplots(figsize=np.max([shape / 4, [3, 3]], axis=0))

        out[label] = fig
        sns.heatmap(pivot, ax=axes)
        axes.set_title(" ".join([label, dependent]))
    return out


def hist(data, dependent, independent, compare):
    """Create histograms comparing datasets.

    Creates a histogram for each unique combination of independent variables.
    The histogram overlays the distribution of the depdendent variable for
    each compared dataset.
    """
    figs = {}
    if len(independent) == 0:
        independent = np.zeros(len(data))
    for indvars, alldata in data.groupby(independent):
        f = plt.figure()
        bins = np.linspace(
            np.nanmin(alldata[dependent]), np.nanmax(alldata[dependent]), 100
        )
        plt.title(str(indvars))
        for label, dataset in alldata.groupby(compare):
            plt.hist(
                dataset[dependent], density=True, label=label, bins=bins, alpha=0.5
            )
        ymax = plt.gca().get_ylim()[1]
        plt.gca().set_prop_cycle(None)
        means = alldata.groupby(compare)[dependent].mean()
        for mean in means:
            plt.vlines(mean, ymin=0, ymax=ymax, linestyle="dashed")
        figs[str(indvars)] = f
        plt.xlabel(dependent)
        if terms.MEAN + dependent in alldata:
            plt.vlines(
                alldata[terms.MEAN + dependent].unique(),
                ymin=0,
                ymax=ymax,
                color="gray",
                linestyle="dashed",
                label="experimental mean",
            )
        plt.legend()
    return figs
