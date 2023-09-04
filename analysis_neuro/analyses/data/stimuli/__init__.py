"""Parameterization of stimuli used in experiments.

The .stimuli dict has as keys the names of sets of stimuli, and as values a list stimuli.
Each stimulus is a dict with several parameters describing it.

We may later use this module to download e.g. stimulus movies from their source
"""
import pandas as pd
from analysis_neuro import terms
from . import allen_brain_observatory
