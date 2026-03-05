"""Loading of data from Seigle et al. 2019."""

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

import pandas as pd

from analysis_neuro import terminology as terms
from analysis_neuro.analyses.data.stimuli import allen_brain_observatory

osi = pd.read_parquet(files("analysis_neuro.analyses.data").joinpath("siegle_tf4.parquet"))
tf4_only = allen_brain_observatory.drifting_gratings[
    allen_brain_observatory.drifting_gratings[terms.VISUAL_STIMULUS + terms.TEMPORAL_FREQUENCY] == 4
]
osi[terms.STIMULUS] = tf4_only.pointer()


spontaneous = pd.read_parquet(
    files("analysis_neuro.analyses.data").joinpath("siegle_spont.parquet")
)

