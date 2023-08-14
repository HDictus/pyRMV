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
    for i, row in parameters.iterrows():
        all_stimuli = stimuli[row[terms.STIMULUS]]
        other_vars = [c for c in parameters.columns if c not in all_stimuli]
        out.append(all_stimuli.assign(**row[other_vars]))
    return pd.concat(out)
