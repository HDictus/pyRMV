import pandas as pd
from pathlib import Path
from analysis_neuro import Analysis
from analysis_neuro import terminology as terms
from analysis_neuro import stats
import seaborn as sns
import matplotlib.pyplot as plt


schuz_density_1989 = Analysis(
    observations=pd.read_csv(Path(__file__).parent / 'data'
                             / 'schuz_neuron_density_1989.csv'),
    measurement=terms.CELL_DENSITY,
    plotter=sns.barplot,
    stats=stats.squared_error,
    doc="""
    We evaluate the similarity of the total neuron density in the primary
    visual cortex of the model to in-vivo values by comparing to the 
    mean density observed in @schuz_density_1989""")


keller_density_2018 = Analysis(
    observations=pd.read_csv(Path(__file__).parent / 'data' / 'keller_2018.csv'),
    measurement=terms.CELL_DENSITY,
    plotter=sns.barplot,
    stats=stats.TTest(),
    verdict=stats.PooledPvalueThreshold(threshold=0.05))

    
