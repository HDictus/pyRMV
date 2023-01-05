"""Terminology used in validation."""

REGION = "region"
LAYER = "layer"
NEURON_OR_GLIA = "neuron or glia?"
MTYPE = 'mtype'

CELL_DENSITY = "cell density ($cells/mm^3$)"
CELL_COUNT = "cell count"
REGION_VOLUME = "volume ($mm^3$)"
DATASET = "dataset"
CITATION = "citation"
NOTES = "notes"
HEMISPHERE = "hemisphere"
COLUMN_RADIUS = 'column radius (um)'

SQERROR = "squared error"
STD = 'std '
TSTAT = 't-statistic'
PVALUE = 'p-value'
SAMPLE_SIZE = 'sample size'


CONNECTION_PROBABILITY = 'connection probability'
INTERSOMATIC_DISTANCE = 'interesomatic distance (um)'

PRESYNAPTIC = 'Presynaptic '
POSTSYNAPTIC = 'Postsynaptic '



measurements = {
    CELL_DENSITY: {"method name": "cell_density"},
    CELL_COUNT: {"method name": "cell_count"},
    CONNECTION_PROBABILITY: {'method name': "connection_probability"},
    INTERSOMATIC_DISTANCE: {'method name': 'intersomatic_distance'},
}
