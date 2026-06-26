import pandas as pd
from pyrmv import terminology as terms
import numpy as np

sm_edges = pd.read_csv("inhibitory_edgelist.csv")
print(sm_edges.head())
# from Fig. 3
layer_boundaries = pd.DataFrame({
    'lower_bound': [
        75.41589648798515,
        122.73567467652504,
        254.34380776340117,
        394.82439926062847,
        539.7412199630314,
        770.4251386321627,
    ],
    'layer': ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
}).set_index('layer')


def assign_layer(row):                                                                                                                                                                                   
    """Based on fig. 3"""   
    if row['m_type'].startswith('L'):
        return row['m_type'][:2]
    position = row['pt_position_um_y']                                                                                                                                                                                              
    for l, b in layer_boundaries['lower_bound'].items():                                                                                                                                                                                 
        if position < b:                                                                                                                                                                                 
            return l                                                                                                                                                                              
    return l

cell_types = pd.read_csv('cell_types.csv').set_index('cell_id')
cell_types['layer'] = [assign_layer(row) for _, row in cell_types.iterrows()]

# There is a mismatch between the edges file and the cell_types file
# edges file is newer: assume it is correct
assert np.all(sm_edges.groupby(['post_soma_id'])['m_type_post'].nunique() == 1)
sm_edges = sm_edges[np.isin(sm_edges['post_soma_id'], cell_types.index.values)]
types = sm_edges.groupby(['post_soma_id'])['m_type_post'].apply(lambda s: s.iloc[0])
cell_types['m_type'] = types
sm_edges['pre_layer'] = [
    mt[:2] if mt.startswith('L') else l
    for l, mt in zip(
        cell_types.loc[sm_edges['pre_soma_id'].values, 'layer'].values,
        sm_edges['m_type_pre']
    )
]

sm_edges = sm_edges[np.isin(sm_edges['post_soma_id'], cell_types.index.values)]
sm_edges['post_layer'] = [
    mt[:2] if mt.startswith('L') else l
    for l, mt in zip(
        cell_types.loc[sm_edges['post_soma_id'].values, 'layer'].values,
        sm_edges['m_type_post']
    )
]
nsyn = pd.DataFrame({
    terms.PRESYNAPTIC + terms.LAYER: sm_edges['pre_layer'],
    terms.POSTSYNAPTIC + terms.LAYER: sm_edges['post_layer'],
    terms.PRESYNAPTIC + terms.SMIZELL_TYPE: sm_edges['m_type_pre'],
    terms.POSTSYNAPTIC + terms.SMIZELL_TYPE: sm_edges['m_type_post'],
    terms.SYNAPSES_PER_CONNECTION: sm_edges['num_syn'],
    terms.COLUMN_RADIUS: 50,
})

cell_counts = sm_edges.groupby(['m_type_post', 'post_layer'])['post_soma_id'].nunique()
print(cell_counts)
connprobs = []
for (prel, posl, prem, posm), edges in nsyn.groupby([
    terms.PRESYNAPTIC + terms.LAYER,
    terms.POSTSYNAPTIC + terms.LAYER,
    terms.PRESYNAPTIC + terms.SMIZELL_TYPE,
    terms.POSTSYNAPTIC + terms.SMIZELL_TYPE]):

    total_pre = cell_counts[(prem, prel)]
    total_post = cell_counts[(posm, posl)]
    if (prel, prem) == (posl, posm):
        npairs = total_pre**2 - total_pre
    else:
        npairs = total_pre * total_post
    nconn = len(edges)
    connprobs.append({
        terms.PRESYNAPTIC + terms.LAYER: prel,
        terms.POSTSYNAPTIC + terms.LAYER: posl,
        terms.PRESYNAPTIC + terms.SMIZELL_TYPE: prem,
        terms.POSTSYNAPTIC + terms.SMIZELL_TYPE: posm,
        terms.SAMPLE_SIZE: npairs,
        terms.COLUMN_RADIUS: 50,
        terms.CONNECTION_PROBABILITY: nconn / npairs
    })

connprob = pd.DataFrame(connprobs)

nsyn[terms.DATASET] = 'Schneider-Mizell2024'
nsyn[terms.CITATION] = 'schneider-mizell_cell-type_2024'
connprob[terms.DATASET] = 'Schneider-Mizell2024'
connprob[terms.CITATION] = 'schneider-mizell_cell-type_2024'

nsyn.to_csv("../../../pyrmv/analyses/data/schneider-mizell-nsyn.csv")
connprob.to_csv("../../../pyrmv/analyses/data/schneider-mizell-connprob.csv")
