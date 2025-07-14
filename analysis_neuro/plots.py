"""Plotting tools."""

from typing import List, Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis_neuro import terminology as terms
from analysis_neuro.stats import _iter_compare


def crossplot(data: pd.DataFrame, dependent: str, independent: str,
              compare: str) -> Dict[str, plt.Figure]:
    """Compare datasets with a crossplot.

    Plot the dependent variable for each of the compared datasets
    along a different axis, showing the level of agreement between them by
    how close they are to the diagonal.

    Args:
        data: DataFrame with the data to plot
        dependent: column name for the dependent variable
        independent: column name for the independent variable
        compare: column name for dataset comparison

    Returns:
        Dictionary mapping comparison names to matplotlib figures

    Raises:
        ValueError: If fewer than 2 datasets are provided
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
        fig, axis = plt.subplots()
        figs[f"{l1}-{l2}"] = fig
        data1 = data1.sort_values(independent)
        data2 = data2.sort_values(independent)
        x_values = data1[dependent].values

        y_values = data2[dependent].values
        axis.scatter(x_values, y_values)
        plt.xlabel(l1)
        plt.ylabel(l2)
        plt.title(dependent)
        minimum = max(axis.get_xlim()[0], axis.get_ylim()[0])
        maximum = min(axis.get_xlim()[1], axis.get_ylim()[1])
        plt.plot((minimum, maximum), (minimum, maximum),
                 color="gray", linestyle="--")

    return figs


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
    # Handle None values in hue by filtering them out for sizing calculation
    if hue is not None:
        hue_values = [h for h in hue if h is not None]
        if len(hue_values) > 0:
            hue_unique_count = len(np.unique(hue_values))
        else:
            hue_unique_count = 1
            hue = None  # If all hue values are None, treat as no hue
    else:
        hue_unique_count = 1

    _, axis = plt.subplots(
        figsize=(len(np.unique(x)) * hue_unique_count / 3, 5)
    )
    return sns.barplot(x=x, y=y, hue=hue, ax=axis)


def unified_histogram(data: pd.DataFrame, dependent: str,
                      independent: List[str], compare: str,
                      reference_data: Optional[np.ndarray] = None,
                      reference_label: Optional[str] = None,
                      n_bins: int = 100) -> Dict[str, plt.Figure]:
    """Create histograms with optional grouping and reference data.

    Combines functionality for grouped histograms and reference data plotting.
    Can handle both grouped data and external reference curves.

    Args:
        data: DataFrame with the data to plot
        dependent: column name for the dependent variable
        independent: list of column names for independent variables (for grouping)
        compare: column name for dataset comparison
        reference_data: optional 2D array with [x, y] reference data
        reference_label: label for the reference data
        n_bins: number of bins for the histogram

    Returns:
        Dictionary mapping group names to matplotlib figures
    """
    figs = {}

    if len(independent) == 0:
        # No grouping - single histogram
        fig = plt.figure()

        # Calculate bins based on all data
        bins = np.linspace(
            np.nanmin(data[dependent]), np.nanmax(data[dependent]), n_bins
        )

        # Plot histogram for each dataset
        for label, dataset in data.groupby(compare):
            plt.hist(
                dataset[dependent], bins=bins, density=True,
                label=label, alpha=0.6
            )

        # Add reference data if provided
        if reference_data is not None:
            # Normalize reference data
            ref_x, ref_y = reference_data[:, 0], reference_data[:, 1]
            ref_y = ref_y / np.trapz(ref_y, ref_x)
            plt.plot(ref_x, ref_y, label=reference_label or "Reference")

        # Add mean lines
        ymax = plt.gca().get_ylim()[1]
        means = data.groupby(compare)[dependent].mean()
        for mean in means:
            plt.vlines(mean, ymin=0, ymax=ymax, linestyle="dashed")

        # Add experimental mean if available
        if terms.MEAN + dependent in data:
            plt.vlines(
                data[terms.MEAN + dependent].unique(),
                ymin=0, ymax=ymax, color="gray", linestyle="dashed",
                label="experimental mean"
            )

        plt.xlabel(dependent)
        plt.legend()
        figs['hist'] = fig

    else:
        # Grouped histograms
        for indvars, alldata in data.groupby(independent):
            fig = plt.figure()

            bins = np.linspace(
                np.nanmin(alldata[dependent]),
                np.nanmax(alldata[dependent]), n_bins
            )

            plt.title(str(indvars))

            # Plot histogram for each dataset
            for label, dataset in alldata.groupby(compare):
                plt.hist(
                    dataset[dependent], bins=bins, density=True,
                    label=label, alpha=0.6
                )

            # Add reference data if provided
            if reference_data is not None:
                ref_x, ref_y = reference_data[:, 0], reference_data[:, 1]
                ref_y = ref_y / np.trapz(ref_y, ref_x)
                plt.plot(ref_x, ref_y, label=reference_label or "Reference")

            # Add mean lines
            ymax = plt.gca().get_ylim()[1]
            means = alldata.groupby(compare)[dependent].mean()
            for mean in means:
                plt.vlines(mean, ymin=0, ymax=ymax, linestyle="dashed")

            # Add experimental mean if available
            if terms.MEAN + dependent in alldata:
                plt.vlines(
                    alldata[terms.MEAN + dependent].unique(),
                    ymin=0, ymax=ymax, color="gray", linestyle="dashed",
                    label="experimental mean"
                )

            plt.xlabel(dependent)
            plt.legend()
            figs[str(indvars)] = fig

    return figs


def averaged_crossplot(data: pd.DataFrame, dependent: str,
                       independent: List[str], compare: str
                       ) -> Dict[str, plt.Figure]:
    """Create crossplots comparing averaged data between datasets.

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
    compared = []

    # Average data by independent variables
    group_cols = [compare] + independent
    averaged = data.groupby(group_cols)[dependent].mean().reset_index()

    # Create crossplots between all dataset pairs
    for dset1, data1 in averaged.groupby(compare):
        for dset2, data2 in averaged.groupby(compare):
            if dset1 == dset2:
                continue
            if (dset2, dset1) in compared:
                continue

            compared.append((dset1, dset2))

            fig = plt.figure()
            one = data1[dependent].values
            other = data2[dependent].values

            minimum = min(one.min(), other.min())
            maximum = max(one.max(), other.max())

            plt.scatter(one, other)
            plt.plot([minimum, maximum], [minimum, maximum], color='gray')
            plt.xlabel(str(dset1))
            plt.ylabel(str(dset2))

            figs[f'{dset1}-{dset2}'] = fig

    return figs


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

    # Filter out intersomatic distance from pathways
    pathways = [ind for ind in independent
                if terms.INTERSOMATIC_DISTANCE not in ind]

    for pway, pway_group in data.groupby(pathways):
        # Group by vertical distance ranges
        vertical_cols = [
            terms.MIN + terms.VERTICAL + terms.INTERSOMATIC_DISTANCE,
            terms.MAX + terms.VERTICAL + terms.INTERSOMATIC_DISTANCE
        ]
        grouped_by_vertical_distance = pway_group.groupby(vertical_cols)

        ngroups = grouped_by_vertical_distance.ngroups
        if ngroups == 0:
            continue

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

        figs[str(pway)] = fig

    return figs


def scatter_with_binned_mean(data: pd.DataFrame, dependent: str,
                             independent: List[str], compare: str,
                             n_bins: int = 20) -> Dict[str, plt.Figure]:
    """Create scatter plots with binned mean overlay.

    Plots scatter data with a binned mean line overlay showing the
    relationship between independent and dependent variables.

    Args:
        data: DataFrame with the data to plot
        dependent: column name for the dependent variable
        independent: list of column names for independent variables
        compare: column name for dataset comparison
        n_bins: number of bins for the mean overlay

    Returns:
        Dictionary mapping dataset labels to matplotlib figures
    """
    figs = {}

    if len(independent) == 0:
        return figs

    x_col = independent[0]

    for label, dataset in data.groupby(compare):
        fig = plt.figure()

        # Create scatter plot
        plt.scatter(
            dataset[x_col], dataset[dependent], alpha=0.1
        )

        # Create binned mean line
        x_range = np.linspace(
            dataset[x_col].min(), dataset[x_col].max(), n_bins
        )
        mean_y = (
            dataset[dependent]
            .groupby(pd.cut(dataset[x_col], x_range))
            .mean()
        )

        # Calculate bin centers and corresponding values, filtering out NaN bins
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
