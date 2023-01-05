import pandas as pd
from analysis_neuro import terminology as terms
from analysis_neuro.analyses.data import data_from_histogram

jiang_intersomatic_2015 = pd.DataFrame(
    {
        terms.INTERSOMATIC_DISTANCE: data_from_histogram(
            bin_edges=[220,237, 305, 371, 437,  505, 571, 637, 701, 809, 770, 837, 903, 1013],
            bin_tops=[811, 568, 253, 427, 115, 255, 427, 673, 566, 707, 670, 811],
            horizontal_min=0, horizontal_max=250,
            vertical_min=0, vertical_max=40),
        terms.NOTES: 'artificial data based on supplementary figure S13',
        terms.CITATION: 'jiang_principles_2015',
        terms.MTYPE: 'PC',
        terms.LAYER: 'L5',
        terms.DATASET: 'Jiang2015',
        terms.COLUMN_RADIUS: 75,
        terms.REGION: 'VISp'}
)
    
