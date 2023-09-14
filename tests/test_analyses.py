import numpy as np
import pytest as pyt
import analysis_neuro.analyses as ana
import analysis_neuro as an
from analysis_neuro import terminology as terms
import pandas as pd


def test_schuz_density():
    class MockModel:

        label = 'mock'

        def cell_density(self, params):
            return params.assign(**{terms.CELL_DENSITY: 100000})

    # just tests the validation runs
    res = ana.schuz_density_1989(MockModel())
    assert res['stats']
    assert terms.SQERROR in list(res['stats'].values())[0]


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
                for num in np.random.uniform(0, 1, size=300)
            ])

    class Perfect:

        label = 'perfect'

        def orientation_selectivity(self, params):
            return ana.siegle_osi_2019.observations.drop(columns=[terms.DATASET, terms.CITATION])

    results = ana.siegle_osi_2019(MockModel(), Perfect())

    assert results['verdict'][
        f"The underlying distribution of {terms.ORIENTATION_SELECTIVITY}"
        " for Siegle2019 and mock is the same"
    ] == "Fail"
    
    assert results['verdict'][
        f"The underlying distribution of {terms.ORIENTATION_SELECTIVITY}"
        " for Siegle2019 and perfect is the same"
    ] == "Pass"

    assert results['verdict'][
        f"The underlying distribution of {terms.ORIENTATION_SELECTIVITY}"
        " for mock and perfect is the same"
    ] == "Fail"


def test_siegle_spontaneous():

    class MockModel:

        label='mock'

        def firing_rate(self, params):
            return pd.DataFrame([
                dict(**row, **{terms.FIRING_RATE: num})
                for _, row in params.iterrows()
                for num in np.random.uniform(0, 1, size=300)
            ])

    class Perfect:

        label = 'perfect'

        def firing_rate(self, params):
            return ana.siegle_spontaneous_2019.observations.drop(columns=[terms.DATASET, terms.CITATION])

    results = ana.siegle_spontaneous_2019(MockModel(), Perfect())

    assert results['verdict'][
        f"The underlying distribution of {terms.FIRING_RATE}"
        " for Siegle2019 and mock is the same"
    ] == "Fail"
    
    assert results['verdict'][
        f"The underlying distribution of {terms.FIRING_RATE}"
        " for Siegle2019 and perfect is the same"
    ] == "Pass"

    assert results['verdict'][
        f"The underlying distribution of {terms.FIRING_RATE}"
        " for mock and perfect is the same"
    ] == "Fail"


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
            return pd.DataFrame([
                {**row, terms.MTYPE: mt}
                for mt in np.random.choice(an.mtypes.SUPPORTED_MTYPE_LABELS, 2)
                for i, row in params.iterrows()])
    
        def connection_probability(self, params):
            rng = np.random.default_rng(1)
            return params.assign(**{
                terms.CONNECTION_PROBABILITY: rng.uniform(0, 0.7, size=params.shape[0]),
                terms.SAMPLE_SIZE: [25] * params.shape[0]})
    
        def synapses_per_connection(self, params):
            rng = np.random.default_rng(1)
            return pd.DataFrame([{
                **row, terms.SYNAPSES_PER_CONNECTION: val
            } for _, row in params.iterrows() for val in rng.poisson(3, size=10) ])
    
        def num_synapses(self, params):
            rng = np.random.default_rng(1)
            return params.assign(**{terms.NUM_SYNAPSES: rng.poisson(1000, size=params.shape[0])})
 
    mock = MockModel()

    with pyt.warns(DeprecationWarning):
        results = ana.mtype_to_mtype_connectivity(mock)
        assert results[terms.CONNECTION_PROBABILITY]
        assert results[terms.SYNAPSES_PER_CONNECTION]
        assert results[terms.NUM_SYNAPSES]
        mock2 = MockModel()
        mock2.label = 'amock'
        
        results = ana.mtype_to_mtype_connectivity(mock, mock2)

def test_ji_innervation_2016():
    
    class BadModel:
        
        label = 'bad'
        
        def fraction_innervated(self, parameters):
            return parameters.assign(**{
                terms.FRACTION_INNERVATED: 0.2})

    class PerfectModel:
        
        label = 'perfect'
        
        def fraction_innervated(self, parameters):
            return ana.ji_innervation_2016.observations

    report = ana.ji_innervation_2016(BadModel(), PerfectModel())
    verdict = report['verdict']
    for hypothesis, value in verdict.items():
        if 'bad' in hypothesis:
            assert value == 'Fail'
            continue
        if 'perfect' in hypothesis:
            assert value == 'Pass'

    assert report['figures']
    