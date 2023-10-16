"""Loading of data from Seigle et al. 2019."""
try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

import pandas as pd
from analysis_neuro import terminology as terms
from analysis_neuro.analyses.data.stimuli import allen_brain_observatory

# tuples are required for the data in seigle et al. Set them here after loading.
<<<<<<< HEAD
osi = pd.read_csv(files('analysis_neuro.analyses.data').joinpath(
    'siegle-osi-2019.csv'))
osi[terms.STIMULUS] = [allen_brain_observatory.drifting_gratings for _, __ in osi.iterrows()]

spontaneous = pd.read_csv(files('analysis_neuro.analyses.data').joinpath(
    'siegle-spontaneous-2019.csv'))
spontaneous[terms.STIMULUS] = [allen_brain_observatory.gray for _, __ in spontaneous.iterrows()]
=======
osi = pd.read_csv(files('analysis_neuro.analyses.data').joinpath('siegle_osi_2019.csv'))
osi[terms.STIM_ORIENTATION] = [
    (0, 45, 90, 135, 180, 225, 270, 315) for i in range(osi.shape[0])
]
osi[terms.RESOLUTION] = [(240, 120) for i in range(osi.shape[0])]
>>>>>>> master
