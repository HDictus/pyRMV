import pandas as pd
import numpy as np
from analysis_neuro import terms
import analysis_neuro.measurements as test_module
from analysis_neuro.analyses.data import stimuli

          
# TODO: simplify: always optimal with respect to other stimulus parameters.
#  if not, then is different property
def test_measures_osi_with_firing_rate():

    class MockModel:

        label = 'mock'
        
        def firing_rate(self, parameters):
            """Each neuron will have a rate of 1 if either temporal frequency or orientation are non-optimal.
            Else a 2.
            This way, each neuron is orientation-selective only at optimal tf
            
            If terms.MTYPE is 'PC' then only the rates of first neuron are returned.
            """
            nrns = [0, 1, 2]
            pref_tf = [2, 8, 4]
            pref_ori = [0, 90, 270]
            out = []
            for _, row in parameters.iterrows():
                if terms.MTYPE in row and (row[terms.MTYPE] == 'PC'):
                    nrns_loc = [0]
                else:
                    nrns_loc = nrns
                for nrn, tf, ori in zip(nrns_loc, pref_tf, pref_ori):
                    if row[terms.TEMPORAL_FREQUENCY] == tf and row[terms.STIM_ORIENTATION] == ori:
                        out.append({terms.FIRING_RATE: 2, terms.CELL_ID: nrn, **row})
                    else:
                        out.append({terms.FIRING_RATE: 1, terms.CELL_ID: nrn, **row})
            return pd.DataFrame(out)
        
        def orientation_selectivity(self, parameters):
           return test_module.orientation_selectivity.from_firing_rate(self, parameters)

    rates = np.array([1, 1, 1, 2, 1, 1, 1, 1])
    oris = np.array([0, 45, 90, 135, 180, 225, 270, 315])
    expected_OSI = np.abs(np.sum(rates * np.exp(2*1j * np.deg2rad(oris)))) / np.sum(rates)


    parameters = pd.DataFrame(
        {terms.STIMULUS: [stimuli.allen_brain_observatory.drifting_gratings]})
    measured = test_module.measure(MockModel(), terms.ORIENTATION_SELECTIVITY, parameters)
    # measures by averaging over stimulus conditions
    expected_columns = set(list(parameters.columns) + [terms.CELL_ID, terms.ORIENTATION_SELECTIVITY, terms.DATASET])
    assert set(measured.columns) == expected_columns
    assert all(measured[terms.ORIENTATION_SELECTIVITY] > 0)

    parameters[terms.TEMPORAL_FREQUENCY] = 1
    expected_columns = set(list(parameters.columns) + [terms.CELL_ID, terms.ORIENTATION_SELECTIVITY, terms.DATASET])
    measured = test_module.measure(MockModel(), terms.ORIENTATION_SELECTIVITY, parameters)
    # measures just for the specified temporal frequency
    assert set(measured.columns) == expected_columns
    assert np.allclose(measured[terms.ORIENTATION_SELECTIVITY], 0)

    parameters[terms.TEMPORAL_FREQUENCY] = 2
    measured = test_module.measure(MockModel(), terms.ORIENTATION_SELECTIVITY, parameters)
    assert np.allclose(measured[terms.ORIENTATION_SELECTIVITY].values, [expected_OSI, 0, 0])

    parameters[terms.TEMPORAL_FREQUENCY] = 'optimal'
    # measures each cell at its optimal TF
    measured = test_module.measure(MockModel(), terms.ORIENTATION_SELECTIVITY, parameters)
    assert np.allclose(measured[terms.ORIENTATION_SELECTIVITY], expected_OSI)

    parameters = pd.DataFrame(
        {terms.STIMULUS: [stimuli.allen_brain_observatory.drifting_gratings,
                          stimuli.allen_brain_observatory.drifting_gratings],
         terms.TEMPORAL_FREQUENCY: [2, 4]})
    measured = test_module.measure(MockModel(), terms.ORIENTATION_SELECTIVITY, parameters)
    # a separate measurement for each unique TF value
    assert np.allclose(
        measured.set_index(terms.TEMPORAL_FREQUENCY).loc[
            2, terms.ORIENTATION_SELECTIVITY],
        [expected_OSI, 0, 0]
    )
    assert np.allclose(
        measured.set_index(terms.TEMPORAL_FREQUENCY).loc[
            4, terms.ORIENTATION_SELECTIVITY],
        [0, 0, expected_OSI]
    )

    # check that other parameters are passed to firing rate as well!
    
    parameters = pd.DataFrame(
        {terms.STIMULUS: [stimuli.allen_brain_observatory.drifting_gratings],
         terms.MTYPE: 'PC'})
    measured = test_module.measure(MockModel(), terms.ORIENTATION_SELECTIVITY, parameters)

    assert list(measured[terms.CELL_ID]) == [0]

def test_measures_connprob_from_pair_weight():

    edges = pd.DataFrame({
        terms.PRESYNAPTIC + terms.LAYER: [2, 2, 2, 3, 3, 3, 3],
        terms.POSTSYNAPTIC + terms.LAYER: 1,
        terms.PRESYNAPTIC + terms.CELL_ID: [1, 2, 3, 4, 5, 6, 7],
        terms.POSTSYNAPTIC + terms.CELL_ID: [0, 0, 0, 0, 0, 0, 0],
        terms.PAIR_WEIGHT: [0, 1, 2, 0, 0, 1, 0]
    })

    class MockModel:

        label='mock'

        def connection_probability(self, parameters):
            return test_module.connection_probability.from_pair_weights(self, parameters)

        def pair_weight(self, parameters):
            return edges
    
    expected = pd.DataFrame({
        terms.PRESYNAPTIC + terms.LAYER: [2, 3],
        terms.POSTSYNAPTIC + terms.LAYER: [1, 1],
        terms.CONNECTION_PROBABILITY: [2/3, 1/4],
        terms.SAMPLE_SIZE: [3, 4],
        terms.DATASET: 'mock'
    })
    params = pd.DataFrame({
        terms.PRESYNAPTIC + terms.LAYER: [2, 3],
        terms.POSTSYNAPTIC + terms.LAYER: [1, 1]}
    )
    result = test_module.measure(
        MockModel(),
        terms.CONNECTION_PROBABILITY,
        params
    )
    pd.testing.assert_frame_equal(
        result,
        expected
    )


def test_measures_fraction_innervated():
    # TODO: Consider whether pre and post id should be index
    # TODO: here we see another example of where allowing nans in dataframes
    #   leads to unintuitive behavior.
    #   so what do we do when a value is specified for one measurement and not another?
    #   should we define our own non-nan nantype?
    edges = pd.DataFrame({
        terms.POSTSYNAPTIC + terms.LAYER: [np.nan, np.nan, np.nan, 3, 3, 3, 3],
        terms.PRESYNAPTIC + terms.LAYER: 1,
        terms.PRESYNAPTIC + terms.CELL_ID: [5, 6, 5, 5, 5, 5, 5],
        terms.POSTSYNAPTIC + terms.CELL_ID: [0, 1, 1, 2, 3, 3, 4],
        terms.PAIR_WEIGHT: [0, 1, 2, 0, 0, 1, 0],
    })
    class MockModel:

        label='mock'

        def fraction_innervated(self, parameters):
            res = test_module.fraction_innervated.from_pair_weights(self, parameters)
            return res

        def pair_weight(self, parameters):
            return edges

    expected = pd.DataFrame({
        terms.POSTSYNAPTIC + terms.LAYER: [np.nan, 3],
        terms.PRESYNAPTIC + terms.LAYER: [1, 1],
        terms.FRACTION_INNERVATED: [1/2, 1/3],
        terms.SAMPLE_SIZE: [2, 3],
        terms.DATASET: 'mock'
    })
    parameters = pd.DataFrame({
        terms.POSTSYNAPTIC + terms.LAYER: [np.nan, 3],
        terms.PRESYNAPTIC + terms.LAYER: [1, 1],
    })
    result = test_module.measure(
        MockModel(),
        terms.FRACTION_INNERVATED,
        parameters
    )

    pd.testing.assert_frame_equal(
        result.sort_values(list(parameters.columns)).reset_index(drop=True),
        expected.sort_values(list(parameters.columns)).reset_index(drop=True)
    )

# TODO: we can do edge weight from synaptic conductance maybe

def test_measure_relative_excitation_from_conn_weight():
    edges = pd.DataFrame({
        terms.POSTSYNAPTIC + terms.LAYER: [np.nan, np.nan, 3, 3],
        terms.PRESYNAPTIC + terms.LAYER: 1,
        terms.PRESYNAPTIC + terms.CELL_ID: [6, 5, 5, 5],
        terms.POSTSYNAPTIC + terms.CELL_ID: [1, 1, 3, 4],
        terms.CONNECTION_WEIGHT: [1, 2, 1, 2],
    })

    class MockModel:

        label='mock'

        def relative_excitation(self, parameters):
            res = test_module.relative_excitation.from_connection_weights(self, parameters)
            return res

        def connection_weight(self, parameters):
            ewithp = edges.assign(**{
                terms.RELATIVE_TO + terms.PRESYNAPTIC + terms.LAYER: 1,
                terms.RELATIVE_TO + terms.POSTSYNAPTIC + terms.LAYER: np.nan
            })
            paramcols = ewithp.set_index(list(parameters.columns))
            out = []
            for i, row in parameters.iterrows():
                out.append(paramcols.loc[tuple(row.values)].reset_index())
            return pd.concat(out)
    
    parameters = pd.DataFrame({
        terms.POSTSYNAPTIC + terms.LAYER: [np.nan, 3],
        terms.PRESYNAPTIC + terms.LAYER: 1,
        terms.RELATIVE_TO + terms.PRESYNAPTIC + terms.LAYER: 1,
        terms.RELATIVE_TO + terms.POSTSYNAPTIC + terms.LAYER: np.nan,
    })

    expected = pd.DataFrame({
        terms.POSTSYNAPTIC + terms.LAYER: [np.nan, 3, 3],
        terms.PRESYNAPTIC + terms.LAYER: [1, 1, 1],
        terms.RELATIVE_EXCITATION: [1, 1/3, 2/3],
        terms.POSTSYNAPTIC + terms.CELL_ID: [1, 3, 4],
        terms.RELATIVE_TO + terms.PRESYNAPTIC + terms.LAYER: 1,
        terms.RELATIVE_TO + terms.POSTSYNAPTIC + terms.LAYER: np.nan,
        terms.DATASET: 'mock'
    })

    result = test_module.measure(MockModel(), terms.RELATIVE_EXCITATION, parameters)

    pd.testing.assert_frame_equal(
        result.sort_index(axis=1).sort_values(list(result.columns)).reset_index(drop=True),
        expected.sort_index(axis=1).sort_values(list(expected.columns)).reset_index(drop=True)
    )


def test_measure_fraction_excitation_per_connection_from_pair_weights():
    edges = pd.DataFrame({
        terms.POSTSYNAPTIC + terms.LAYER: [np.nan, np.nan, 3, 3],
        terms.PRESYNAPTIC + terms.LAYER: 1,
        terms.PRESYNAPTIC + terms.CELL_ID: [6, 5, 5, 5],
        terms.POSTSYNAPTIC + terms.CELL_ID: [1, 1, 3, 4],
        terms.CONNECTION_WEIGHT: [1, 2, 1, 2],
    })

    class MockModel:

        label='mock'

        def fraction_excitation_per_connection(self, parameters):
            res = test_module.fraction_excitation.from_connection_weights(self, parameters)
            return res

        def connection_weight(self, parameters):
            paramcols = edges.set_index(list(parameters.columns))
            out = []
            for i, row in parameters.iterrows():
                out.append(paramcols.loc[tuple(row.values)].reset_index())
            return pd.concat(out)
    
    parameters = pd.DataFrame({
        terms.POSTSYNAPTIC + terms.LAYER: [np.nan, 3],
        terms.PRESYNAPTIC + terms.LAYER: [1, 1],
    })

    expected = pd.DataFrame({
        terms.POSTSYNAPTIC + terms.LAYER: [np.nan, np.nan, 3, 3],
        terms.PRESYNAPTIC + terms.LAYER: [1, 1, 1, 1],
        terms.FRACTION_EXCITATION_PER_CONNECTION: [1/3, 2/3, 1, 1],
        terms.PRESYNAPTIC + terms.CELL_ID: [6, 5, 5, 5],
        terms.POSTSYNAPTIC + terms.CELL_ID: [1, 1, 3, 4],
        terms.DATASET: 'mock'
    })

    result = test_module.measure(MockModel(), terms.FRACTION_EXCITATION_PER_CONNECTION, parameters)

    res_sort = result.sort_index(axis=1)
    exp_sort = expected.sort_index(axis=1)
    pd.testing.assert_frame_equal(
        res_sort.sort_values(list(res_sort.columns)).reset_index(drop=True),
        exp_sort.sort_values(list(exp_sort.columns)).reset_index(drop=True)
    )


def test_measures_with_method():

    density = terms.Term(
        'density (kg/m^3)',
        '',
        measurement_method='density')
 
    params = pd.DataFrame({'_': [0, 0]})

    class MockModelWithDens:
        
        label = 'mock'

        def density(self, parameters):
            return parameters.assign(**{density: 100})

    dens = test_module.measure(
        MockModelWithDens(), density, params)
    assert dens[density].values[0] == 100


def test_measure_tuple():
    
    thing1 = terms.Term("thing1", measurement_method='thing1')
    thing2 = terms.Term("thing2", measurement_method='thing2')
    aparam = terms.Term("aparam")
    
    class MockModelMeasuresBoth:
        # TODO: the problem with this is that we are relying on the modeler
        #   to ensure that all relevant measures line up
        #   this may not be straightforward for them
        #   and until they themselves implement that specifically
        #   others cannot rely on their model's compound measurements
        label = 'mock'
        
        def thing1(self, parameters):
            return parameters.assign(thing1=np.arange(len(parameters)))
        
        def thing2(self, parameters):
            return parameters.assign(thing2=1 + np.arange(len(parameters)))
        
    measured = test_module.measure(
        MockModelMeasuresBoth(), [thing1, thing2],
        pd.DataFrame({aparam: [1, 2, 3, 4, 5]})
    )
    assert all(measured[thing1] == np.arange(5))
    assert all(measured[thing2] == 1 + np.arange(5))


def test_extract_parameters():
    measurement = pd.DataFrame({
        'param a': [1, 1, 2, 2, 3],
        'param b': [1, 2, 1, 1, 2],
        'measured': [1, 2, 3, 4, 5]})

    # measurement and duplicate parameter sets should be removed
    pd.testing.assert_frame_equal(
        test_module.extract_parameters(measurement, 'measured'),
        pd.DataFrame({
            'param a': [1, 1, 2, 3],
            'param b': [1, 2, 1, 2]}))

    # can work without measurement
    pd.testing.assert_frame_equal(
        test_module.extract_parameters(measurement[['param a', 'param b']]),
        pd.DataFrame({
            'param a': [1, 1, 2, 3],
            'param b': [1, 2, 1, 2]}))

    # doesn't drop na values
    df = pd.DataFrame({
        'a': [1, 2, None, None, None],
        'b': ['b', 'c', 'd', 'e', 'e'],
    })
    pd.testing.assert_frame_equal(
        test_module.extract_parameters(df),
        pd.DataFrame({
            'a': [1, 2, None, None],
            'b': ['b', 'c', 'd', 'e']
        })
    )

