"""Data extracted from @ma_visual_2010"""
import pandas as pd
from analysis_neuro import terminology as terms

spontaneous = pd.DataFrame(
    {
        terms.MEAN
        + terms.FIRING_RATE: [
            0.08256880733945562,
            2.889908256880739,
            0.1651376146789022,
            1.6513761467889954,
        ],
        terms.SAMPLE_SIZE: [37, 12, 25, 21],
        terms.STD
        + terms.FIRING_RATE: [
            0.24770642201834883,
            5.201834862385325,
            0.41284403669725106,
            3.7155963302752326,
        ],
        terms.LAYER: ["L4", "L4", "L23", "L23"],
        terms.REGION: "VISp",
        terms.GENE_EXPRESSION: ["Sst", "Pvalb", "Sst", "Pvalb"],
        terms.DATASET: "Ma2010",
        terms.VISUAL_STIMULUS: "gray",
        terms.VISUAL_STIMULUS + terms.ANGLE_AZIMUTH: pd.Interval(10, 80),
        terms.VISUAL_STIMULUS + terms.ANGLE_ELEVATION: pd.Interval(-27, 27),
        terms.NOTES: "Manually digitized from Figure 1F."
    }
)
# std was based on error bar locations - so must subtract the mean to get it
spontaneous[terms.STD + terms.FIRING_RATE] -= spontaneous[
    terms.MEAN + terms.FIRING_RATE
]
