import numpy as np
import analysis_neuro.analyses as ana
import analysis_neuro as an
from analysis_neuro import terminology as terms
import pandas as pd


def test_jiang_connprob():
    """
    sort of just test that it runs. Temporary.
    """

    class MockModel:

        label = "mock"

        def connection_probability(self, params):
            return params.assign(**{terms.CONNECTION_PROBABILITY: 0.1})

    class PerfectModel:

        label='perfect'

        def connection_probability(self, params):
            return ana.jiang_connprob_2015.observations.assign(**{terms.DATASET: self.label})


    result = ana.jiang_connprob_2015(MockModel())
    mockresults = result["measurements"][
        result["measurements"][terms.DATASET] == "mock"
    ]
    assert np.all(mockresults[terms.CONNECTION_PROBABILITY] == 0.1)
    assert result["verdict"] == {
        "The probability connection probability is the same for Jiang2015 as for mock.": "Fail"
    }

    perfect = ana.jiang_connprob_2015(PerfectModel())
    perfect["verdict"] == {
        "The probability connection probability is the same for Jiang2015 as for perfect.": "Pass"
    }



def test_jiang_intersomatic():
    class MockModel:

        label = "mock"

        def intersomatic_distance(self, params):
            return params.assign(**{terms.INTERSOMATIC_DISTANCE: 100})

    result = ana.jiang_intersomatic_2015(MockModel())
    mockresults = result["measurements"][
        result["measurements"][terms.DATASET] == "mock"
    ]
    assert np.all(mockresults[terms.INTERSOMATIC_DISTANCE] == 100)


def test_siegle_osi():
    class MockModel:

        label='mock'

        def orientation_selectivity(self, params):
            return pd.DataFrame([
                dict(**row, **{terms.ORIENTATION_SELECTIVITY: num})
                for _, row in params.iterrows()
                for num in np.linspace(0, 1, 5)])

    results = ana.siegle_osi_2019(MockModel())
    mockmeasurements = results['measurements'][
        results['measurements'][terms.DATASET] == 'mock'
    ]
    print(mockmeasurements)
    assert np.all(
        mockmeasurements[terms.ORIENTATION_SELECTIVITY] == np.concatenate(
            [np.linspace(0, 1, 5)] * 2))


def test_pala_peterson_conprob_2015():
    class MockModel:

        label = "mock"

        def connection_probability(self, parameters):
            return parameters.assign(**{terms.CONNECTION_PROBABILITY: 0.1})

    class PerfectModel:

        label='perfect'

        def connection_probability(self, parameters):
            return ana.pala_peterson_conprob_2015.observations.assign(**{terms.DATASET: self.label})

    result = ana.pala_peterson_conprob_2015(MockModel())
    mockresults = result["measurements"][result["measurements"][terms.DATASET] == "mock"]
    assert np.all(mockresults[terms.CONNECTION_PROBABILITY] == 0.1)
    assert result["verdict"] == {
        "The probability connection probability is the same for PalaPeterson2015 as for mock.": "Fail"
    }

    perfect = ana.pala_peterson_conprob_2015(PerfectModel())

    assert perfect["verdict"] == {
        "The probability connection probability is the same for PalaPeterson2015 as for perfect.": "Pass"
    }


def test_mtype_to_mtype_connectivity():

    class MockModel:
    
        label='mock'
    
        def mtype(self, params=None):
            if params is None:
                params = pd.DataFrame(index=[1])
            return pd.DataFrame([{**row, terms.MTYPE: mt} for mt in an.mtypes.PRIMITIVES 
                                 for i, row in params.iterrows()])
    
        def connection_probability(self, params):
            rng = np.random.default_rng(1)
            return params.assign(**{
                terms.CONNECTION_PROBABILITY: rng.uniform(0, 0.7, size=params.shape[0])})
    
        def synapses_per_connection(self, params):
            rng = np.random.default_rng(1)
            return pd.DataFrame([{
                **row, terms.SYNAPSES_PER_CONNECTION: val
            } for _, row in params.iterrows() for val in rng.poisson(3, size=10) ])
    
        def num_synapses(self, params):
            rng = np.random.default_rng(1)
            return params.assign(**{terms.NUM_SYNAPSES: rng.poisson(1000, size=params.shape[0])})
 
    mock = MockModel()
    
    results = ana.mtype_to_mtype_connectivity(mock)
    assert results[terms.CONNECTION_PROBABILITY]
    assert results[terms.SYNAPSES_PER_CONNECTION]
    assert results[terms.NUM_SYNAPSES]
