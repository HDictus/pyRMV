"""Parameterization of stimuli used in experiments.

The .stimuli dict has as keys the names of sets of stimuli, and as values a list stimuli.
Each stimulus is a dict with several parameters describing it.

We may later use this module to download e.g. stimulus movies from their source
"""
import pandas as pd
from analysis_neuro import terms
from . import allen_brain_observatory


# TODO: this isn't quite accurate
stimuli = {
    **allen_brain_observatory.stimuli
}


# is this method redundant?
def get(parameters):
    """Create a dataframe of all stimulus conditions for STIMULUS."""
    out = []
    other_columns = [c for c in parameters.columns if c != terms.STIMULUS]
    for i, row in parameters.iterrows():
        out.append(stimuli[row[terms.STIMULUS]].assign(**row[other_columns]))
    return pd.concat(out)
