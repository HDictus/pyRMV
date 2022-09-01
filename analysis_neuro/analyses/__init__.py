import pandas as pd
from pathlib import Path
import analysis_neuro as ana
import seaborn as sns
import matplotlib.pyplot as plt


def schuz_density_1989(*models):
    """
    Compare the neuron density of models to those reported in @schuz_density_1989
    """
    schuz_data = pd.read_csv(Path(__file__).parent / 'data' / 'schuz_neuron_density_1989.csv')
    parameters = schuz_data[[ana.REGION, ana.LAYER, ana.NEURON_OR_GLIA]]
    outdata = [schuz_data]
    for model in models:
        outdata.append(model.cell_density(parameters)
                       .assign(dataset=model.label))
    measurement = pd.concat(outdata, axis=0)
    f, a = plt.subplots()
    sns.barplot(x=measurement[ana.LAYER], y=measurement[ana.CELL_DENSITY],
                ax=a)
    
    return {'measurement': measurement, 'figures': {'bar plot': f}}
