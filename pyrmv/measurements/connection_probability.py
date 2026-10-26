"""Connection probability measurement methods."""

import pandas as pd

import pyrmv.terminology as terms


def from_pair_weights(model, parameters):
    """Measure terms.CONNECTION_PROBABILITY from pair weights.

    Arguments:
        model: must support measuring terms.PAIR_WEIGHT
        parameters: DataFrame of measurement parameters

    Returns:
        DataFrame with connection probability and sample size
    """
    from pyrmv.measurements import measure  # pylint: disable=import-outside-toplevel

    edges = measure(model, terms.PAIR_WEIGHT, parameters)
    edges['conn'] = edges[terms.PAIR_WEIGHT] > 0
    groups = edges.groupby(list(parameters.columns))['conn']
    connprob = pd.DataFrame({
        terms.CONNECTION_PROBABILITY: groups.mean(),
        terms.SAMPLE_SIZE: groups.count()
    }).reset_index()
    return connprob
