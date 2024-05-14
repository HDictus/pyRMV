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

def test_cossell_correlation_psp_2015():
    
    
    class MockModel:
        # TODO: it's kind of awkward to ensure that
        #   the two measurements compare ok
        #   we may want to make it so that 
        #   measuring a term is actually done by the model itself
        #   so that you can request a tuple measurement
        #   and the model will align them appropriately
        
        def response_correlation(self, parameters):
            out = []
            for i, p in parameters.iterrows():
                out += [
                    {**p, 
                     terms.CELL_ID: pair,
                     terms.RESPONSE_CORRELATION: corr}
                     for pair, corr in zip(self.pair_ids, self.pair_corrs)
                ]
            return pd.DataFrame(out)
        
        def psp_amplitude(self, parameters):
            out = []
            for i, p in parameters.iterrows():
                out += [
                    {**p,
                     terms.PRESYNAPTIC + terms.CELL_ID: pair[0],
                     terms.POSTSYNAPTIC + terms.CELL_ID: pair[1],
                     terms.PSP_AMPLITUDE: psp}
                    for pair, psp in zip(self.pair_ids, self.pair_psps)
                ]
            return pd.DataFrame(out)
        
    class BadModel(MockModel):
        label = 'bad'
        
        pair_corrs = [0, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9]
        pair_psps = [1, 0, 1, 0, 1, 0, 1, 0]
        pair_ids = [(0, 1), (0, 2), (1, 2), (0, 4)]
        
    class PerfectModel(MockModel):
        label = 'perfect'
        
        pair_corrs = np.linspace(0, 1, 100)
        pair_psps = [0.5 / 93] * 93 + [0.2 / 4] * 4 + [0.3 / 3] * 3
        pair_ids = [(0, num) for num in range(100)]
        

    results = ana.cossell_correlation_psp_2015(BadModel(), PerfectModel())
    print(results)
    assert len(results['verdict'].keys()) == 2
    for hyp, value in results['verdict'].items():
        if 'bad' in hyp:
            assert value == 'Fail'
        else:
            assert value == 'Pass'


def test_cossell_correlation_psp_2015():
    
    
    class MockModel:
        # TODO: it's kind of awkward to ensure that
        #   the two measurements compare ok
        #   we may want to make it so that 
        #   measuring a term is actually done by the model itself
        #   so that you can request a tuple measurement
        #   and the model will align them appropriately
        
        def response_correlation(self, parameters):
            out = []
            for i, p in parameters.iterrows():
                out += [
                    {**p, 
                     terms.CELL_ID: pair,
                     terms.RESPONSE_CORRELATION: corr}
                     for pair, corr in zip(self.pair_ids, self.pair_corrs)
                ]
            return pd.DataFrame(out)
        
        def psp_amplitude(self, parameters):
            out = []
            for i, p in parameters.iterrows():
                out += [
                    {**p,
                     terms.PRESYNAPTIC + terms.CELL_ID: pair[0],
                     terms.POSTSYNAPTIC + terms.CELL_ID: pair[1],
                     terms.PSP_AMPLITUDE: psp}
                    for pair, psp in zip(self.pair_ids, self.pair_psps)
                ]
            return pd.DataFrame(out)
        
    class BadModel(MockModel):
        label = 'bad'
        
        pair_corrs = [0, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9]
        pair_psps = [1, 0, 1, 0, 1, 0, 1, 0]
        pair_ids = [(0, 1), (0, 2), (1, 2), (0, 4)]
        
    class PerfectModel(MockModel):
        label = 'perfect'
        
        pair_corrs = np.linspace(0, 1, 100)
        pair_psps = [0.5 / 93] * 93 + [0.2 / 4] * 4 + [0.3 / 3] * 3
        pair_ids = [(0, num) for num in range(100)]
        

    results = ana.cossell_correlation_psp_2015(BadModel(), PerfectModel())
    print(results)
    assert len(results['verdict'].keys()) == 2
    for hyp, value in results['verdict'].items():
        if 'bad' in hyp:
            assert value == 'Fail'
        else:
            assert value == 'Pass'
            
    
def test_cossell_connprob_corr_2015():
    
    class BadModel:
        
        label = 'bad'
        
        def connection_probability(self, params):
            return params.assign(**{terms.CONNECTION_PROBABILITY: 0.1})
        
    class PerfectModel:
        
        label = 'perfect'
        
        def connection_probability(self, params):
            return ana.cossell_connprob_corr_2015.observations.drop(columns=terms.DATASET)
        
    results = ana.cossell_connprob_corr_2015(BadModel(), PerfectModel())
    print(results)
    assert len(results['verdict'].keys()) == 3
    for hyp, value in results['verdict'].items():
        if 'bad' in hyp:
            assert value == 'Fail'
        else:
            assert value == 'Pass'
            

def test_lee_connprob_2016():
    
    class BadModel:
        
        label = 'bad'
        
        def connection_probability(self, params):
            return params.assign(**{terms.CONNECTION_PROBABILITY: 0.1})
        
    class PerfectModel:
        
        label = 'perfect'
        
        def connection_probability(self, params):
            return ana.lee_connprob_2016.observations.drop(columns=terms.DATASET)
        
    results = ana.lee_connprob_2016(BadModel(), PerfectModel())
    print(results)
    assert len(results['verdict'].keys()) == 3
    for hyp, value in results['verdict'].items():
        if 'bad' in hyp:
            assert value == 'Fail'
        else:
            assert value == 'Pass'
            
def test_cossell_response_correlation_2015():
    assert False

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
    # Check that the axis labels are intelligible and not a mess of Nones
    ax = report['figures'].get_axes()[0]
    labels = [l.get_text() for l in ax.get_xticklabels()]
    assert labels == [
        "L1", 
        "L23 PV", "L23 Sst", "L23 Vip", "L23 EXC",
        "L4 PV", "L4 Sst", "L4 Vip", "L4 EXC",
        "L5 PV", "L5 Sst", "L5 Vip", "L5 EXC",
        "L6 PV", "L6 Sst", "L6 Vip", "L6 EXC"
    ]


def test_ji_relative_excitation():

    class BadModel():

        label = 'bad'

        def relative_excitation(self, parameters):
            return parameters.assign(**{terms.RELATIVE_EXCITATION: 1})

    class PerfectModel():

        label = 'perfect'

        def relative_excitation(self, parameters):
            return ana.ji_relative_2016.observations.copy()
    
    result = ana.ji_relative_2016(BadModel(), PerfectModel())

    for hypothesis, verdict in result['verdict'].items():
        if 'bad' in hypothesis:
            assert verdict == 'Fail'
        else:
            assert verdict == 'Pass'

    assert result['figures']

    
def test_lien_fraction_excitation_2018():
    # TODO: we should consider the possibility that in cases like this,
    #   a model can implement measures of the mean value for a given sample size
    #   but cannot directly provide samples
    #   for instance, if it describes the phenomenon as a probability distribution
    #   It would be better to let MEAN + <measurement> be the actual measurement,
    #   and sample size a parameter
    #   then we can even implement a helper method that can measure 
    #   MEAN + <measurement> for any model which implements <measurement> directly
    #   for any <measurement>
    #   then things can be greatly simplified, we won't even need the bootstrapper.

    class BadModel:
        
        label = 'bad'
        
        def fraction_excitation_per_connection(self, parameters):
            return pd.DataFrame([
                {**row, terms.FRACTION_EXCITATION_PER_CONNECTION: 0.2}
                for _ in range(20)
                for __, row in parameters.iterrows()
            ])
            
    class GoodModel:
        
        label = 'good'
        
        def fraction_excitation_per_connection(self, parameters):
            """This model includes one very strong connection, and several weak ones
            as @cite:ringach_sparse_2021 points out, a distribution in which there are 
            many very weak connections and a few very strong connections, the
            fraction contributed by connections sampled for a measurement will
            often appear very small.
            So although the mean value from this model differs greatly from the 
            experimental data, a proper test should show that
            the likelihood of the experimental results given the model's distribution
            is acutally fairly high.
            """
            return pd.DataFrame([
                {**row, terms.FRACTION_EXCITATION_PER_CONNECTION: frac}
                for frac in [0.8] + [0.01] * 20
                for _, row in parameters.iterrows()  
            ])
        
    results = ana.lien_fraction_excitation_2018(BadModel(), GoodModel())
    
    assert results['verdict']['The result of Lien2018 could be sampled from the same distribution as bad'] == "Fail"
    assert results['verdict']['The result of Lien2018 could be sampled from the same distribution as good'] == "Pass"
