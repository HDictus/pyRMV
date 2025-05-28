"""Loading of data from Seigle et al. 2019."""

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

import pandas as pd

from analysis_neuro import terminology as terms
from analysis_neuro.analyses.data.stimuli import allen_brain_observatory

osi = pd.read_csv(files("analysis_neuro.analyses.data").joinpath("siegle-osi-2019.csv"))

osi[terms.STIMULUS] = [
    allen_brain_observatory.drifting_gratings for _, __ in osi.iterrows()
]

dsi = pd.read_csv(files("analysis_neuro.analyses.data").joinpath("siegle-dsi-2019.csv"))

dsi[terms.STIMULUS] = [
    allen_brain_observatory.drifting_gratings for _, __ in osi.iterrows()
]


spontaneous = pd.read_csv(
    files("analysis_neuro.analyses.data").joinpath("siegle-spontaneous-2019.csv")
)
spontaneous[terms.STIMULUS] = [
    allen_brain_observatory.gray for _, __ in spontaneous.iterrows()
]

