"""Plotting tools."""

from typing import List, Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pyrmv import terminology as terms
from pyrmv.stats import _iter_compare


def crossplot(data: pd.DataFrame, dependent: str, independent: List[str],
              compare: str) -> Dict[str, plt.Figure]:
    """Compare datasets with a crossplot using averaged data.

    Groups data by independent variables, computes means, then creates
    scatter plots comparing between different datasets.

    Args:
        data: DataFrame with the data to plot
        dependent: column name for the dependent variable
        independent: list of column names for independent variables
        compare: column name for dataset comparison

    Returns:
        Dictionary mapping comparison names to matplotlib figures
    """
    figs = {}

    group_cols = [compare] + independent
    averaged = data.groupby(group_cols)[dependent].mean().reset_index()

    for l1, data1, l2, data2 in _iter_compare(averaged, compare):
        fig = _crossplot_one(l1, l2, data1, data2, dependent)
        figs[f"{l1}-{l2}"] = fig
    return figs


def _crossplot_one(l1, l2, data1, data2, dependent):
    fig = plt.figure()

    one = data1[dependent].values
    other = data2[dependent].values

    minimum = min(one.min(), other.min())
    maximum = max(one.max(), other.max())

    plt.scatter(one, other)
    plt.plot([minimum, maximum], [minimum, maximum], color='gray')
    plt.xlabel(str(l1))
    plt.ylabel(str(l2))
    return fig


def pathway_heatmap(data: pd.DataFrame, dependent: str,
                    independent: List[str], compare: str) -> Dict[str, plt.Figure]:
    """Create a heatmap for connectivity data.

    Args:
        data: DataFrame with all measurements
        dependent: dependent variable to plot
        independent: independent variables, must include some prefixed with
                    PRESYNAPTIC and POSTSYNAPTIC
        compare: variable to compare between. Creates a separate plot for each
                value of this variable.

    Returns:
        Dictionary mapping comparison values to matplotlib figures
    """
    out = {}
    for label, dataframe in data.groupby(compare):
        pre_cols = [col for col in independent
                    if col.startswith(terms.PRESYNAPTIC)]
        post_cols = [col for col in independent
                     if col.startswith(terms.POSTSYNAPTIC)]
        pivot = dataframe.pivot_table(
            index=pre_cols, columns=post_cols, values=dependent
        )
        shape = np.array(pivot.shape)
        fig, axes = plt.subplots(figsize=np.max([shape / 4, [3, 3]], axis=0))

        out[label] = fig
        sns.heatmap(pivot, ax=axes)
        axes.set_title(" ".join([label, dependent]))
    return out


def hist(data: pd.DataFrame, dependent: str, independent: List[str],
         compare: str) -> Dict[str, plt.Figure]:
    """Create histograms comparing datasets.

    Creates a histogram for each unique combination of independent variables.
    The histogram overlays the distribution of the dependent variable for
    each compared dataset.

    Args:
        data: DataFrame with the data to plot
        dependent: column name for the dependent variable
        independent: list of column names for independent variables
        compare: column name for dataset comparison

    Returns:
        Dictionary mapping variable combinations to matplotlib figures
    """
    figs = {}
    independent_vars = independent if len(independent) > 0 else np.zeros(len(data))
    for indvars, alldata in data.groupby(independent_vars):
        fig = plt.figure()
        bins = np.linspace(
            np.nanmin(alldata[dependent]), np.nanmax(alldata[dependent]), 50
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
        figs[str(indvars)] = fig
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


def wide_barplot(x: List[Any], y: List[float], hue: Optional[List[Any]]
                 ) -> plt.Axes:
    """Create wide bar plots with automatic figure sizing.

    Creates bar plots with figure width scaled based on the number of
    unique values in x and hue variables.

    Args:
        x: List of x-axis category values
        y: List of y-axis values
        hue: List of hue category values (can contain None values)

    Returns:
        Matplotlib axes object containing the plot
    """
    if hue is not None:
        hue_values = [h for h in hue if h is not None]
        if len(hue_values) > 0:
            hue_unique_count = len(np.unique(hue_values))
        else:
            hue_unique_count = 1
            hue = None
    else:
        hue_unique_count = 1

    _, axis = plt.subplots(
        figsize=(len(np.unique(x)) * hue_unique_count / 3, 5)
    )
    return sns.barplot(x=x, y=y, hue=hue, ax=axis)


def pathway_barplot(data: pd.DataFrame, dependent: str,
                    independent: List[str], compare: str
                    ) -> Dict[str, plt.Figure]:
    """Create bar plots grouped by pathway characteristics.

    Creates separate subplots for different vertical distance ranges,
    with bars grouped by horizontal distance and colored by dataset.

    Args:
        data: DataFrame with the data to plot
        dependent: column name for the dependent variable
        independent: list of column names for independent variables
        compare: column name for dataset comparison

    Returns:
        Dictionary mapping pathway names to matplotlib figures
    """
    figs = {}

    pathways = [ind for ind in independent
                if terms.INTERSOMATIC_DISTANCE not in ind]

    for pway, pway_group in data.groupby(pathways):
        figs[str(pway)] = _barplot_for_pathway(pway, pway_group, dependent, compare)

    return figs


def _barplot_for_pathway(_pway, pway_group, dependent, compare):
    vertical_cols = [
        terms.MIN + terms.VERTICAL + terms.INTERSOMATIC_DISTANCE,
        terms.MAX + terms.VERTICAL + terms.INTERSOMATIC_DISTANCE
    ]
    grouped_by_vertical_distance = pway_group.groupby(vertical_cols)

    ngroups = grouped_by_vertical_distance.ngroups
    if ngroups == 0:
        return None

    fig, axes = plt.subplots(
        ngroups, 1, sharex=True, figsize=(8, 3 * ngroups)
    )
    if ngroups == 1:
        axes = [axes]

    for i, ((min_vert, max_vert), group) in enumerate(
        grouped_by_vertical_distance
    ):
        sns.barplot(
            data=group,
            x=terms.MIN + terms.HORIZONTAL + terms.INTERSOMATIC_DISTANCE,
            y=dependent,
            hue=compare,
            ax=axes[i]
        )
        axes[i].set_title(f'Vertical distance: {min_vert}-{max_vert}')

    return fig


def scatter_with_binned_mean(data: pd.DataFrame, dependent: str,
                             independent: List[str], compare: str) -> Dict[str, plt.Figure]:
    """Create scatter plots with binned mean overlay.

    Plots scatter data with a binned mean line overlay showing the
    relationship between independent and dependent variables.

    Args:
        data: DataFrame with the data to plot
        dependent: column name for the dependent variable
        independent: list of column names for independent variables
        compare: column name for dataset comparison

    Returns:
        Dictionary mapping dataset labels to matplotlib figures
    """
    figs = {}

    if len(independent) == 0:
        return figs

    x_col = independent[0]

    for label, dataset in data.groupby(compare):
        fig = plt.figure()

        plt.scatter(
            dataset[x_col], dataset[dependent], alpha=0.1
        )

        x_range = np.linspace(
            dataset[x_col].min(), dataset[x_col].max(), 50
        )
        mean_y = (
            dataset[dependent]
            .groupby(pd.cut(dataset[x_col], x_range))
            .mean()
        )

        centers = []
        values = []
        for interval, value in mean_y.items():
            if not pd.isna(interval) and not pd.isna(value):
                centers.append(np.mean([interval.left, interval.right]))
                values.append(value)

        if len(centers) > 0:
            plt.plot(centers, values, color="black", linewidth=2)
        plt.xlabel(x_col)
        plt.ylabel(dependent)

        figs[label] = fig

    return figs
