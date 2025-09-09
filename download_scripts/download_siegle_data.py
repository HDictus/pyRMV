import pandas as pd
from analysis_neuro import terminology as terms

import os
import shutil

import numpy as np
import pandas as pd
from pathlib import Path
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# TODO: is there a way to safely run allensdk functions alongside analysis-neuro?
data_directory = __file__.parent  / 'data_store' / 'ecephys' # must be updated to a valid directory in your filesystem

manifest_path = os.path.join(data_directory, "manifest.json")

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

metrics = cache.get_unit_analysis_metrics_by_session_type('brain_observatory_1.1')
strict = metrics['isi_violations'] == 0
metrics = metrics[strict]
# TODO: the operations after this point are pretty cheap - maybe we could run everything up to this in a python shell to get the metrics
# then load them in siegle_2019.py and do the following operations alognside setting the stimulus
# that way there are fewer steps involved in getting the data, and it's all transparently included
FS = metrics['waveform_duration'] < 0.5
metrics = metrics.assign(spiking_class=['FS' if at else 'RS' for at in FS])

experiment = metrics[['ecephys_structure_acronym', 'spiking_class', 'g_osi_dg', 'firing_rate', 'firing_rate_dg', ]]

observations = pd.DataFrame({
    terms.FIRING_RATE: experiment['firing_rate'],
    terms.VISUAL_STIMULUS: stimuli.ALLEN_BRAIN_OBSERVATORY + 'gray',
    terms.REGION: experiment['ecephys_structure_acronym'],
    terms.SPIKING_CLASS: experiment['spiking_class'],
    terms.CITATION: 'siegle_survey_2019',
    terms.DATASET: 'Siegle2019'
})
observations.to_csv("analysis-neuro/analysis_neuro/analyses/data/siegle-spontaneous-2019.csv", index=False)

import numpy as np
osi = pd.DataFrame({
    terms.ORIENTATION_SELECTIVITY: experiment['g_osi_dg'],
    terms.VISUAL_STIMULUS: stimuli.ALLEN_BRAIN_OBSERVATORY + 'drifting gratings',
    terms.TEMPORAL_FREQUENCY: 'optimal',
    terms.REGION: experiment['ecephys_structure_acronym'],
    terms.SPIKING_CLASS: experiment['spiking_class'],
    terms.CITATION: 'siegle_survey_2019',
    terms.DATASET: 'Siegle2019'
})
assert not np.any(np.isnan(osi[terms.ORIENTATION_SELECTIVITY]))
osi.to_csv("analysis-neuro/analysis_neuro/analyses/data/siegle-osi-2019.csv", index=False)

optimal = pd.DataFrame({
    terms.FIRING_RATE: experiment['firing_rate_dg'],
    terms.VISUAL_STIMULUS: stimuli.ALLEN_BRAIN_OBSERVATORY + 'drifting gratings',
    terms.TEMPORAL_FREQUENCY: 'optimal',
    terms.STIM_ORIENTATION: 'optimal',
    terms.REGION: experiment['ecephys_structure_acronym'],
    terms.SPIKING_CLASS: experiment['spiking_class'],
    terms.CIRATION: 'siegle_survey_2019',
    terms.DATASET: 'Siegle2019'
})
optimal.to_csv("analysis-neuro/analysis_neuro/analyses/data/siegle-optimal-2019.csv")