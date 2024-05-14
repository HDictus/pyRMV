"""Plotting tools."""
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from analysis_neuro import terminology as terms


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

    # if len(comparevalues) < 2:
    #     raise ValueError(
    #         "a crossplot only makes sense when comparing data. "
    #         "It is not defined for less than 2 datasets."
    #     )
    # if len(comparevalues) > 2:
    #     raise NotImplementedError(
    #         "currently crossplot comparisons between more than 2 "
    #         "datasets is not implemented, please make a pull request"
    #     )
    from analysis_neuro.stats import _iter_compare
    for l1, data1, l2, data2 in _iter_compare(data, compare):
        # TODO: ensure this is usable for sampled data as well
        # TODO: add reusable test for plotter that it handles unordered data
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


def pathway_heatmap(data: pd.DataFrame, dependent: str, independent: List[str], compare: str):
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
            index=pre_cols,
            columns=post_cols,
            values=dependent)
        shape = np.array(pivot.shape)
        fig, axes = plt.subplots(figsize=np.max([shape / 4, [3, 3]], axis=0))

        out[label] = fig
        sns.heatmap(pivot, ax=axes)
        axes.set_title(" ".join([label, dependent]))
    return out


