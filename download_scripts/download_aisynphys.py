import pandas as pd
import numpy as np
import aisynphys
from aisynphys.database import SynphysDatabase
from analysis_neuro import terms
from tqdm import tqdm
from pathlib import Path
import datetime
aisynphys.config.cache_path = "/work/lnmc/visual_cortex/synphys_cache"#"./synphys_cache"
db = SynphysDatabase.load_current('full')

pairs = db.pair_query(project_name=db.mouse_projects)




# TODO: come up with and enforce naming conventio for genes
#  Ideally, pick an existing one
def _format_gene(gene):
    if gene == 'unknown':
        return None
    return ','.join([g.capitalize() for g in gene.split(',')])



def _format_sclass(sclass, gene):
    if sclass is None:
        return None 
    formatted = sclass.replace('ex', 'EXC').replace('in', 'INH')
    if gene == 'unknown':
        return formatted

    return formatted
    

psp_data = []
psc_data = []
conn_data = []
conductance_data = []
stp_data = []

# TODO: enforce region naming convention, so someone could more easily catch this
for pair in tqdm(list(pairs)):
    if pair.pre_cell is None or pair.lateral_distance is None:
        continue
    if pair.experiment.target_region is None:
        continue
    # we choose to ignore pairs where the synapse class could not be determined
    if pair.pre_cell.cell_class == 'mixed':
        continue
    if pair.post_cell.cell_class == 'mixed':
        continue
    if pair.lateral_distance > 0.001 or pair.vertical_distance > 0.01:
        continue
    data = {
        terms.SPECIES: 'mouse',
        terms.REGION: pair.experiment.target_region.replace('VisP', 'VISp'), # breaking your own convetion, aibs... tut-tut
        terms.HORIZONTAL + terms.INTERSOMATIC_DISTANCE: np.abs(pair.lateral_distance) * 1e6,
        terms.VERTICAL + terms.INTERSOMATIC_DISTANCE: np.abs(pair.vertical_distance) * 1e6,

        terms.PRESYNAPTIC + terms.LAYER: f'L{pair.pre_cell.cortical_location.cortical_layer}'.replace('/', ''),
        terms.PRESYNAPTIC + terms.GENE_EXPRESSION: _format_gene(pair.pre_cell.cre_type),
        terms.PRESYNAPTIC + terms.SYNAPSE_CLASS: _format_sclass(pair.pre_cell.cell_class, pair.pre_cell.cre_type),
        terms.PRESYNAPTIC + terms.CELL_ID: pair.pre_cell.id,


        terms.POSTSYNAPTIC + terms.LAYER: f'L{pair.post_cell.cortical_location.cortical_layer}'.replace('/', ''),
        terms.POSTSYNAPTIC + terms.GENE_EXPRESSION: _format_gene(pair.post_cell.cre_type),
        terms.POSTSYNAPTIC + terms.SYNAPSE_CLASS: _format_sclass(pair.post_cell.cell_class, pair.pre_cell.cre_type),
        terms.POSTSYNAPTIC + terms.CELL_ID: pair.post_cell.id,


        terms.DATASET: 'AISynphys',
        terms.NOTES: f"Accessed on {datetime.date.today().strftime('%Y-%m-%d %H:%M:%S7')}",
        'connected': pair.synapse is not None
    }
    conn_data.append(data)
    if pair.synapse is not None and pair.synapse.conductance:
        data_psp = {**data}
        data_psc = {**data}
        data_conductance = {**data}
        data_stp = {**data}
        synapse = pair.synapse
        # TODO: I should create a more flexible, and directly usable stimulus representation
        #  e.g. the actual pulse times!
        vhold = synapse.conductance[0].ideal_holding_potential * 1000  # we choose the ideal, because it is a question of how to replicate, not how to do this specific connection
        if synapse.psp_amplitude:
            data_psp.update(**{
                terms.PSP_AMPLITUDE: synapse.psp_amplitude * 1000,
                terms.PSP_RISE_TIME: synapse.psp_rise_time * 1000,
                terms.PSP_DECAY_TAU: synapse.psp_decay_tau * 1000,
                terms.HOLDING_POTENTIAL: vhold,
            })
            psp_data.append(data_psp)

        if synapse.psc_amplitude:
            data_psc.update(**{
                terms.PSC_AMPLITUDE: synapse.psc_amplitude * 1e9,
                terms.PSC_RISE_TIME: synapse.psc_rise_time * 1000,
                terms.PSC_DECAY_TAU: synapse.psc_decay_tau * 1000,
                terms.VOLTAGE_CLAMP: vhold,
            })
            psc_data.append(data_psc)

        if len(synapse.conductance):
            data_conductance.update(**{
                terms.SYNAPTIC_CONDUCTANCE: synapse.conductance[0].effective_conductance * 1e3, # mS -> nS
                terms.REVERSAL_POTENTIAL: synapse.conductance[0].reversal_potential * 1000,
            })
            conductance_data.append(data_conductance)

        dynamics = pair.dynamics
        data_stp.update(**{
            # from https://portal.brain-map.org/explore/connectivity/synaptic-physiology/synaptic-physiology-analysis-methods/synapse-characterization#stp
            # but note that we could extract more detailed information from the dataset...
            terms.STIM_FREQUENCY: 50, # Hz
            terms.NUM_PULSES: 8,
            terms.INTERBURST_INTERVAL: 250, # ms,
            terms.NUM_BURSTS: 2,
            terms.PAIRED_PULSE_DIFFERENCE: dynamics.stp_initial_50hz,
            # TODO: N are different for STP induction and recovery. Should they be separate dataframes?
            terms.STP_INDUCTION: dynamics.stp_induction_50hz,
            terms.STP_RECOVERY: dynamics.stp_recovery_250ms,
            terms.HOLDING_POTENTIAL: vhold
        })

        stp_data.append(data_stp)



def set_idx(df):
    measured = [terms.PSP_AMPLITUDE, terms.PSC_AMPLITUDE, terms.PSP_RISE_TIME, terms.PSC_RISE_TIME, terms.PSP_DECAY_TAU, terms.PSC_DECAY_TAU,
                terms.PAIRED_PULSE_DIFFERENCE, terms.STP_INDUCTION. terms.STP_RECOVERY]
    return df.set_index([c for c in df.columns if c not in measured])

conns = pd.DataFrame(conn_data)


def partition_distances(df):
    partition_size = 25 # um
    def _partition(df, col):
        print("bins")
        print(df[col].max()+partition_size,             partition_size)
        bins = np.arange(
            0,
            df[col].max()+partition_size,
            partition_size
        )
        print("cut")
        intervals = pd.cut(df[col], bins)
        print("set")
        df[terms.MIN + col] = [ival.left for ival in intervals.values]
        df[terms.MAX + col] = [ival.right for ival in intervals.values]

    _partition(df, terms.HORIZONTAL + terms.INTERSOMATIC_DISTANCE)
    _partition(df, terms.VERTICAL + terms.INTERSOMATIC_DISTANCE)
    return df

print(len(conn_data), "Pairs had usable data")
print("With ", len(psp_data), len(psc_data), len(stp_data), "PSP, PSC, and STP connections")
conn = pd.DataFrame(conn_data)
conn = partition_distances(conn)
print("Grouping...")
grouped_by_pathway_and_distance = conn.groupby([
    terms.REGION,

    terms.PRESYNAPTIC + terms.SYNAPSE_CLASS,
    terms.PRESYNAPTIC + terms.GENE_EXPRESSION,
    terms.PRESYNAPTIC + terms.LAYER,

    terms.POSTSYNAPTIC + terms.SYNAPSE_CLASS,
    terms.POSTSYNAPTIC + terms.GENE_EXPRESSION,
    terms.POSTSYNAPTIC + terms.LAYER,

    terms.MIN + terms.HORIZONTAL + terms.INTERSOMATIC_DISTANCE,
    terms.MAX + terms.HORIZONTAL + terms.INTERSOMATIC_DISTANCE,

    terms.MIN + terms.VERTICAL + terms.INTERSOMATIC_DISTANCE,
    terms.MAX + terms.VERTICAL + terms.INTERSOMATIC_DISTANCE,

    terms.DATASET, terms.NOTES
])
print("Grouped")
connprob = grouped_by_pathway_and_distance['connected'].mean()
connprob.name = terms.CONNECTION_PROBABILITY
connprob = connprob.reset_index()
connprob[terms.SAMPLE_SIZE] = grouped_by_pathway_and_distance['connected'].count().values
print("Saving...")
DATA_DIR = Path('../../../analysis_neuro/analyses/data/')
import datetime

connprob.reset_index(drop=True).to_csv(DATA_DIR / "campagnola_mouse_2022_connectivity.csv")

partition_distances(pd.DataFrame(psp_data)).to_csv(DATA_DIR / "campagnola_mouse_2022_psp.csv")
partition_distances(pd.DataFrame(psc_data)).to_csv(DATA_DIR / "campagnola_mouse_2022_psc.csv")
partition_distances(pd.DataFrame(stp_data)).to_csv(DATA_DIR / "campagnola_mouse_2022_stp.csv")
partition_distances(pd.DataFrame(conductance_data)).to_csv(DATA_DIR / "campagnola_mouse_2022_conductance.csv")
# partition_distances(pd.DataFrame(psp_data))

