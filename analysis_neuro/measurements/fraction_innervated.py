"""Fraction innervated measurement methods."""

import pandas as pd

import analysis_neuro.terminology as terms


def from_pair_weights(model, parameters):
    """Measure fraction innervated based on edge weights.

    See terms.FRACTION_INNERVATED for definition of term.

    Arguments:
        model: must provide a measurement method for PAIR_WEIGHT
        parameters: DataFrame of measurement parameters

    Returns:
        DataFrame with fraction innervated and sample size
    """
    from analysis_neuro.measurements import measure

    edges = measure(model, terms.PAIR_WEIGHT, parameters)
    edges['conn'] = edges[terms.PAIR_WEIGHT] > 0
    innervated = edges.groupby(
        list(parameters.columns)
        + [terms.POSTSYNAPTIC + terms.CELL_ID],
        dropna=False
    )['conn'].any()

    grouped_by_parameters = innervated.reset_index().groupby(
        list(parameters.columns), dropna=False
    )['conn']
    finner = grouped_by_parameters.mean()
    ncells = grouped_by_parameters.count()
    return pd.DataFrame({
        terms.FRACTION_INNERVATED: finner,
        terms.SAMPLE_SIZE: ncells
    }).reset_index()
