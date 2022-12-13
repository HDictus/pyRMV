import numpy as np
import analysis_neuro.analyses as ana
from analysis_neuro import terminology as terms


def test_jiang_connprob():
    """
    sort of just test that it runs. Temporary.
    """
    class MockModel:

        label = 'mock'
        
        def connection_probability(self, params):
            return params.assign(**{terms.CONNECTION_PROBABILITY: 0.1})

    result = ana.jiang_connprob_2016(MockModel())
    mockresults = result['measurements'][result['measurements'][terms.DATASET] == 'mock']
    assert np.all(mockresults[terms.CONNECTION_PROBABILITY] == 0.1)
