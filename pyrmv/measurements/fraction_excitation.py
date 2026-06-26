"""Fraction excitation measurement methods."""
import pyrmv.terminology as terms


def from_connection_weights(model, parameters):
    """Measure fraction excitation per connection from connection weights.

    See terms.FRACTION_EXCITATION_PER_CONNECTION for definition of term.

    Arguments:
        model: must provide a measurement method for CONNECTION_WEIGHT
        parameters: DataFrame of measurement parameters

    Returns:
        DataFrame with fraction excitation per connection
    """
    from pyrmv.measurements import measure

    edges = measure(model, terms.CONNECTION_WEIGHT, parameters)
    edges = edges[edges[terms.CONNECTION_WEIGHT] != 0]
    groupcols = list(parameters.columns) + [terms.POSTSYNAPTIC + terms.CELL_ID]
    tot_exc = edges.groupby(groupcols, dropna=False)[
        terms.CONNECTION_WEIGHT
    ].sum()
    idx_cols = groupcols + [terms.PRESYNAPTIC + terms.CELL_ID]
    fin = edges.set_index(idx_cols)[terms.CONNECTION_WEIGHT] / tot_exc
    fin.name = terms.FRACTION_EXCITATION_PER_CONNECTION
    return fin.reset_index()
