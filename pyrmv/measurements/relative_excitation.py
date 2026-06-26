"""Relative excitation measurement methods."""

import pandas as pd

import pyrmv.terminology as terms


def _weights_sum(model, parameters):
    """Calculate sum of connection weights per postsynaptic cell.

    Arguments:
        model: must provide a measurement method for CONNECTION_WEIGHT
        parameters: DataFrame of measurement parameters

    Returns:
        Series of summed connection weights grouped by parameters and postsynaptic cell
    """
    from pyrmv.measurements import measure

    edges = measure(model, terms.CONNECTION_WEIGHT, parameters)
    groupcols = list(parameters.columns) + [terms.POSTSYNAPTIC + terms.CELL_ID]
    cond_per_tgid = edges.groupby(groupcols, dropna=False)[
        terms.CONNECTION_WEIGHT
    ].sum()
    return cond_per_tgid


def from_connection_weights(model, parameters):
    """Measure relative excitation from connection weights.

    Arguments:
        model: must provide a measurement method for CONNECTION_WEIGHT
        parameters: DataFrame of measurement parameters including RELATIVE_TO columns

    Returns:
        DataFrame with relative excitation values
    """
    cond_per_tgid = _weights_sum(model, parameters).reset_index()
    relativecols = [col for col in cond_per_tgid if col.startswith(terms.RELATIVE_TO)]
    normalized = []
    for grp, conds in cond_per_tgid.groupby(relativecols, dropna=False):
        relative_params = {
            col.replace(terms.RELATIVE_TO, ''): val
            for col, val in zip(relativecols, grp)
        }
        relative_to = _weights_sum(model, pd.DataFrame(relative_params, index=[0]))
        conds[terms.RELATIVE_EXCITATION] = conds[terms.CONNECTION_WEIGHT] / relative_to.mean()
        normalized.append(conds.drop(columns=[terms.CONNECTION_WEIGHT]))
    return pd.concat(normalized)
