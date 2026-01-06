"""Fraction innervated measurement methods."""
import pandas as pd
import numpy as np

import analysis_neuro.terminology as terms


def from_pair_weights(model, parameters):
    """Measure fraction innervated based on pair weights.

    See terms.FRACTION_INNERVATED for definition of term.
    This method tends to be especially slow for large, sparse networks
    Consider using from_connection_weights instead.

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


def from_connection_weights(model, parameters):
    """Measure FRACTION_INNERVATED from cell ids and connections."""
    from analysis_neuro.measurements import measure, pre_post_params
    out = []
    for i, row in parameters.iterrows():
        _, post_params = pre_post_params(row)
        post_ids = measure(model, terms.CELL_ID, pd.DataFrame([post_params]))
        conns = measure(model, terms.CONNECTION_WEIGHT, parameters.loc[[i]])
        frac = np.isin(post_ids[terms.CELL_ID], conns[terms.POSTSYNAPTIC + terms.CELL_ID]).mean()
        out.append({
            terms.FRACTION_INNERVATED: frac,
            terms.SAMPLE_SIZE: len(post_ids),
             **row}
        )
    return pd.DataFrame(out)
