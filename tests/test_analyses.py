import numpy as np
import analysis_neuro.analyses as ana
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

    result = ana.jiang_connprob_2015(MockModel())
    mockresults = result["measurements"][
        result["measurements"][terms.DATASET] == "mock"
    ]
    assert np.all(mockresults[terms.CONNECTION_PROBABILITY] == 0.1)


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
                                 
