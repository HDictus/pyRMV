import h5py as h5
import numpy as np
import conntility
import matplotlib.pyplot as plt
import analysis_neuro as an
from analysis_neuro import terms
from analysis_neuro import io
import pandas as pd

# Downloaded from https://zenodo.org/records/13849415
# A more accessible format for the connectome, helpfully provided by Michael Reimann
M = conntility.ConnectivityMatrix.from_h5("microns_mm3_connectome_v1181.h5", 'condensed')

vert_positions = M.vertices[['x_nm', 'y_nm', 'z_nm']]#[['L' not in ct for ct in M.vertices['cell_type']]]
center = vert_positions[['x_nm', 'z_nm']].mean(axis=0)
distances = np.linalg.norm(vert_positions[['x_nm', 'z_nm']].values-center.values, axis=1)
COLUMN_RADIUS = 125
vertices_in = distances < COLUMN_RADIUS * 1000

# TODO: not sure this is right way around
print(M.matrix.shape)
posns_in = vert_positions[vertices_in]

submatrix = M.submatrix(posns_in.index)
print(submatrix)
in_degree = np.array(submatrix.sum(axis=0)).flatten()
out_degree = np.array(submatrix.sum(axis=1)).flatten()

print("calced", in_degree.shape, out_degree.shape)

plt.hist(in_degree, bins=300)
plt.savefig('indegree.png')
print('saved')
plt.clf()
plt.hist(out_degree, bins=300)
plt.savefig('outdegree.png')
plt.clf()
plt.scatter(in_degree, out_degree)
plt.xlabel('indegree')
plt.ylabel('outdegree')
plt.savefig('degree_corr.png')
plt.clf()
print("Saved degree corr")

microns_indegree = pd.DataFrame({
    terms.REGION: 'VIS',
    terms.COLUMN_RADIUS: COLUMN_RADIUS,
    terms.IN_DEGREE: in_degree,
})

microns_outdegree = pd.DataFrame({
    terms.REGION: 'VIS',
    terms.COLUMN_RADIUS: COLUMN_RADIUS,
    terms.OUT_DEGREE: out_degree,
})


posns_in['cell_type'] = M.vertices['cell_type'].loc[posns_in.index]

horizontal_distances = pd.DataFrame(
    np.linalg.norm(
        posns_in[['x_nm', 'z_nm']].values[..., np.newaxis, :] - 
        posns_in[['x_nm', 'z_nm']].values[np.newaxis],
        axis=-1
    ),
    index=pd.Series(posns_in.index, name='pregid'),
    columns=pd.Series(posns_in.index, name='postgid')
)

horizontal_distances = horizontal_distances.melt(value_name='horizontal', ignore_index=False).reset_index().set_index(['pregid', 'postgid'])['horizontal']

vertical_distances = pd.DataFrame(
    np.linalg.norm(
        posns_in[['y_nm']].values[..., np.newaxis, :] - 
        posns_in[['y_nm']].values[np.newaxis],
        axis=-1
    ),
    index=pd.Series(posns_in.index, name='pregid'),
    columns=pd.Series(posns_in.index, name='postgid')
).melt(value_name='vertical', ignore_index=False).reset_index().set_index(['pregid', 'postgid'])['vertical']

connected = pd.Series(0, index=horizontal_distances.index)
connected_idx = pd.MultiIndex.from_arrays(np.nonzero(M.matrix))
connected_idx = connected_idx.intersection(connected.index)

connected[connected_idx] = 1
pre = horizontal_distances.index.get_level_values(0)
post = horizontal_distances.index.get_level_values(1)
pairs = pd.DataFrame({
    'connected': connected,
    'horizontal': horizontal_distances,
    'vertical': vertical_distances,
    'pre_type': M.vertices.loc[pre, 'cell_type'].values,
    'post_type': M.vertices.loc[post, 'cell_type'].values
})
pairs = pairs[np.logical_and(
    pairs['horizontal'] != 0,
    pairs['vertical'] != 0
)]


#import pdb; pdb.set_trace()
pairs.to_parquet('microns_pairs.parquet')
# TODO: This binning into horizontal and vertical bins, rather than leaving the distances in directly,
#  (something done in aisynphys as well)
# reflects an assumption that the model explicitly represents pairs of neurons with set distances, rather than predicting a 
# connprob on the basis of a specific distance
# To properly accomodate such a model, we should be providing the pairs with connected/unconnected
# then it is up to the model to decide if it wants to predict the connection probabilities 
# by binning distances
# the model can then also choose the appropriate bin size for itself
# However, this is terribly inconvenient when it comes to plotting
# and we would need to rework our statistical analysis
# hmm... would we per se? if we say terms.CONNECTION_PROBABILITY: 1 or 0,
#  then the model must provide a prediction for each individual pair
#  that is much more in keeping with our validation philosophy of providing predicted properties as opposed to performing experimental procedures
# It is an improvement for later - really don't have time right now
# in fact, include it as limitation and future work
pairs['horizontal_bin'] = pd.cut(pairs['horizontal'], np.arange(0, pairs['horizontal'].max(), 25))
pairs['vertical_bin'] = pd.cut(pairs['vertical'], np.arange(pairs['vertical'].min(), pairs['vertical'].max(), 25))
grouped = pairs.groupby(['pre_type', 'post_type', 'vertical_bin', 'horizontal_bin'], observed=True)
npairs = grouped['connected'].count()
connprob = grouped['connected'].mean().reset_index()

microns_connprob = pd.DataFrame({
    terms.PRESYNAPTIC + terms.SMIZELL_TYPE: connprob['pre_type'].values,
    terms.POSTSYNAPTIC + terms.SMIZELL_TYPE: connprob['post_type'].values,
    terms.COLUMN_RADIUS: COLUMN_RADIUS,
    terms.MAX + terms.HORIZONTAL + terms.INTERSOMATIC_DISTANCE: [bn.right for bn in connprob['horizontal_bin']],
    terms.MIN + terms.HORIZONTAL + terms.INTERSOMATIC_DISTANCE: [bn.left for bn in connprob['horizontal_bin']],
    terms.MAX + terms.VERTICAL + terms.INTERSOMATIC_DISTANCE: [bn.right for bn in connprob['vertical_bin']],
    terms.MIN + terms.VERTICAL + terms.INTERSOMATIC_DISTANCE: [bn.left for bn in connprob['vertical_bin']],
    terms.CONNECTION_PROBABILITY: connprob['connected'].values,
    terms.SAMPLE_SIZE: npairs.values
})
microns_connprob.to_feather("analysis_neuro/analyses/data/microns-connprob-2021.feather")

alledges = submatrix.edges.set_index(connected_idx)
intersection = alledges.index.intersection(pairs.index)
connected_pairs = pairs.loc[intersection]
connected_pairs[['count', 'total_size']] = alledges.loc[intersection, ['count', 'total_size']]

microns_nsyn = pd.DataFrame({
    terms.SYNAPSES_PER_CONNECTION: connected_pairs['count'].values,
    terms.COLUMN_RADIUS: COLUMN_RADIUS,
    terms.MAX + terms.HORIZONTAL + terms.INTERSOMATIC_DISTANCE: [bn.right for bn in connected_pairs['horizontal_bin']],
    terms.MIN + terms.HORIZONTAL + terms.INTERSOMATIC_DISTANCE: [bn.left for bn in connected_pairs['horizontal_bin']],
    terms.MAX + terms.VERTICAL + terms.INTERSOMATIC_DISTANCE: [bn.right for bn in connected_pairs['vertical_bin']],
    terms.MIN + terms.VERTICAL + terms.INTERSOMATIC_DISTANCE: [bn.left for bn in connected_pairs['vertical_bin']],
    terms.PRESYNAPTIC + terms.SMIZELL_TYPE: connected_pairs['pre_type'].values,
    terms.POSTSYNAPTIC + terms.SMIZELL_TYPE: connected_pairs['post_type'].values,
})

# relative connection strength to create distribution
# TODO: which property of the model are we concerned with here?
#  in a model without anatomical PSD area, we are concened with synaptic weight or synaptic conductance
#  but we may well decide to use synapse size
microns_norm_weight = pd.DataFrame({
    terms.NORMALIZED_SYNAPSE_SIZE: connected_pairs['total_size'] / connected_pairs['total_size'].mean(),
    terms.COLUMN_RADIUS: COLUMN_RADIUS,
    terms.MAX + terms.HORIZONTAL + terms.INTERSOMATIC_DISTANCE: [bn.right for bn in connected_pairs['horizontal_bin']],
    terms.MIN + terms.HORIZONTAL + terms.INTERSOMATIC_DISTANCE: [bn.left for bn in connected_pairs['horizontal_bin']],
    terms.MAX + terms.VERTICAL + terms.INTERSOMATIC_DISTANCE: [bn.right for bn in connected_pairs['vertical_bin']],
    terms.MIN + terms.VERTICAL + terms.INTERSOMATIC_DISTANCE: [bn.left for bn in connected_pairs['vertical_bin']],
    terms.PRESYNAPTIC + terms.SMIZELL_TYPE: connected_pairs['pre_type'].values,
    terms.POSTSYNAPTIC + terms.SMIZELL_TYPE: connected_pairs['post_type'].values,
})

# TODO: may be better to store multiple measurements with same parameterization in single df and load each one

microns_indegree.to_feather("analysis_neuro/analyses/data/microns-indegree-2021.feather")
microns_outdegree.to_feather("analysis_neuro/analyses/data/microns-outdegree-2021.feather")
microns_connprob.to_feather("analysis_neuro/analyses/data/microns-connprob-2021.feather")
microns_nsyn.to_feather("analysis_neuro/analyses/data/microns-nsyn-2021.feather")
microns_norm_weight.to_feather("analysis_neuro/analyses/data/microns-normweight-2021.feather")
