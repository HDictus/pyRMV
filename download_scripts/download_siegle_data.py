import pandas as pd
from analysis_neuro import terminology as terms
from analysis_neuro.analyses.data import stimuli
import os
import shutil

import numpy as np
import pandas as pd
from pathlib import Path
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# TODO: is there a way to safely run allensdk functions alongside analysis-neuro?
data_directory = Path(__file__).parent  / 'data_store' / 'ecephys' # must be updated to a valid directory in your filesystem

manifest_path = os.path.join(data_directory, "manifest.json")

cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

metrics = cache.get_unit_analysis_metrics_by_session_type('brain_observatory_1.1')
strict = metrics['isi_violations'] == 0
metrics = metrics[strict]

FS = metrics['waveform_duration'] < 0.5
metrics = metrics.assign(spiking_class=['FS' if at else 'RS' for at in FS])


units = cache.get_units()

xyz = units[[
    'anterior_posterior_ccf_coordinate', 'dorsal_ventral_ccf_coordinate',
    'left_right_ccf_coordinate']
].values

from voxcell.nexus.voxelbrain import Atlas
import voxcell
atlas = Atlas.open("/work/lnmc/visual_cortex/VISp/.atlas")
rmap = atlas.load_region_map()
region_ids = atlas.load_data("brain_regions").lookup(xyz, outer_value=-1)
def try_get_acronym(region_id):
    try:
        return rmap.get(region_id, 'acronym')
    except voxcell.exceptions.VoxcellError:
        return '???'

acronyms = [try_get_acronym(region_id) for region_id in region_ids]

def _get_layer(acronym):
    if acronym[-1] in '123456':
        return f'L{acronym[-1]}'
    if acronym[-2] == '6':
        return f'L{acronym[-2:]}'
    return None

layers = [_get_layer(acronym) for acronym in acronyms]
units['layer'] = layers

experiment = metrics[['ecephys_structure_acronym', 'spiking_class', 'g_osi_dg', 'firing_rate', 'firing_rate_dg', 'firing_rate_ns']]
experiment['layer'] = units['layer']

observations = pd.DataFrame({
    terms.FIRING_RATE: experiment['firing_rate'],
    terms.REGION: experiment['ecephys_structure_acronym'],
    terms.SPIKING_CLASS: experiment['spiking_class'],
    terms.CITATION: 'siegle_survey_2019',
    terms.LAYER: experiment['layer'],
    terms.DATASET: 'Siegle2019'
})
observations.to_csv("analysis_neuro/analyses/data/siegle-spontaneous-2019.csv", index=False)

import numpy as np
osi = pd.DataFrame({
    terms.ORIENTATION_SELECTIVITY: experiment['g_osi_dg'],
    terms.TEMPORAL_FREQUENCY: 'optimal',
    terms.REGION: experiment['ecephys_structure_acronym'],
    terms.SPIKING_CLASS: experiment['spiking_class'],
    terms.CITATION: 'siegle_survey_2019',
    terms.LAYER: experiment['layer'],
    terms.DATASET: 'Siegle2019'
}).reset_index(drop=True)
assert not np.any(np.isnan(osi[terms.ORIENTATION_SELECTIVITY]))
osi.to_csv("analysis_neuro/analyses/data/siegle-osi-2019.csv", index=False)

optimal = pd.DataFrame({
    terms.FIRING_RATE: experiment['firing_rate_dg'],
    terms.TEMPORAL_FREQUENCY: 'optimal',
    terms.STIM_ORIENTATION: 'optimal',
    terms.REGION: experiment['ecephys_structure_acronym'],
    terms.SPIKING_CLASS: experiment['spiking_class'],
    terms.LAYER: experiment['layer'],
    terms.CITATION: 'siegle_survey_2019',
    terms.DATASET: 'Siegle2019'
}).reset_index(drop=True)
optimal.to_csv("analysis_neuro/analyses/data/siegle-optimal-2019.csv")

natural = pd.DataFrame({
    terms.FIRING_RATE: experiment['firing_rate_ns'],
    terms.REGION: experiment['ecephys_structure_acronym'],
    terms.SPIKING_CLASS: experiment['spiking_class'],
    terms.LAYER: experiment['layer'],
    terms.CITATION: 'siegle_survey_2019',
    terms.DATASET: 'Siegle2019'
}).reset_index(drop=True)
natural.to_csv("analysis_neuro/analyses/data/siegle-natural-2019.csv")