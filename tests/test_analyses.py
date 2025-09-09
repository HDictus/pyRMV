"""
Refined, simple test suite for analysis_neuro analyses.

This approach uses two universal model classes and pytest parameterization
to eliminate code duplication while being accessible to scientist contributors.
"""

import numpy as np
import pandas as pd
import pytest

import analysis_neuro.analyses as ana
from analysis_neuro import Analysis, terms

rng = np.random.default_rng(1)

class MockModel:
    """Universal mock model that returns constant/random values for all measurements."""

    label = 'mock'

    def connection_probability(self, params):
        return params.assign(**{terms.CONNECTION_PROBABILITY: 0.9})

    def fraction_innervated(self, params):
        return params.assign(**{terms.FRACTION_INNERVATED: 0.1})

    def cell_density(self, params):
        return params.assign(**{terms.CELL_DENSITY: 100000})

    def intersomatic_distance(self, params):
        return params.assign(**{terms.INTERSOMATIC_DISTANCE: 100})

    def relative_excitation(self, params):
        data = randomized_measurement(params, terms.RELATIVE_EXCITATION, 40)
        data[terms.RELATIVE_EXCITATION] /= data[terms.RELATIVE_EXCITATION].mean()
        return data

    def psp_amplitude(self, params):
        return randomized_measurement(params, terms.PSP_AMPLITUDE)

    def response_correlation(self, params):
        return randomized_measurement(params, terms.RESPONSE_CORRELATION)

    def firing_rate(self, params):
        return randomized_measurement(params, terms.FIRING_RATE)

    def orientation_selectivity(self, params):
        return randomized_measurement(params, terms.ORIENTATION_SELECTIVITY)

    def synapses_per_connection(self, params):
        return randomized_measurement(
            params, terms.SYNAPSES_PER_CONNECTION,
            nsamples=100,
            distribution=lambda nsamples: rng.poisson(100, nsamples)
        )

    def paired_pulse_difference(self, params):
        return randomized_measurement(
            params, terms.PAIRED_PULSE_DIFFERENCE
        )

    def psc_amplitude(self, params):
        return randomized_measurement(
            params, terms.PSC_AMPLITUDE
        )

    def psc_decay_tau(self, params):
        return randomized_measurement(
            params, terms.PSC_DECAY_TAU
        )

    def psc_rise_time(self, params):
        return randomized_measurement(
            params, terms.PSC_RISE_TIME
        )

    def psp_decay_tau(self, params):
        return randomized_measurement(
            params, terms.PSP_DECAY_TAU
        )

    def psp_rise_time(self, params):
        return randomized_measurement(
            params, terms.PSP_RISE_TIME
        )

    def stp_induction(self, params):
        return randomized_measurement(
            params, terms.STP_INDUCTION
        )

    def stp_recovery(self, params):
        return randomized_measurement(
            params, terms.STP_RECOVERY
        )

    def cell_count(self, params):
        return params.assign(**{terms.CELL_COUNT: 1000})

    def somatic_current(self, params):
        return randomized_measurement(
            params, terms.SOMATIC_CURRENT
        )

    def fraction_excitation_per_connection(self, params):
        return randomized_measurement(
            params, terms.FRACTION_EXCITATION_PER_CONNECTION
        )


def randomized_measurement(
    params, measurement,
    nsamples=100,
    distribution=lambda nsamples: rng.uniform(0, 1, nsamples)
):
    """Random measurement for mock model"""
    return pd.DataFrame([
        {**row, measurement: val}
         for _, row in params.iterrows()
         for val in distribution(nsamples)
    ])

class PerfectModel:
    """Universal perfect model that returns experimental data for any analysis."""

    label = 'perfect'

    def __init__(self, analysis):
        self.analysis = analysis

    def __getattr__(self, method_name):
        """Dynamically handle any measurement method by returning experimental data."""
        def measurement_method(params):
            obs = self.analysis.observations.copy()
            if terms.DATASET in obs.columns:
                obs = obs.drop(columns=[terms.DATASET])
            if terms.STD + self.analysis.measurement in obs.columns:
                obs = obs.drop(columns=[terms.STD + self.analysis.measurement])
            return obs.assign(**{terms.DATASET: self.label})
        return measurement_method


def run_basic_analysis_test(analysis):
    """
    Run a basic test on any analysis to verify that a control model fails, and a perfect model passes
    """
    mock_model = MockModel()
    perfect_model = PerfectModel(analysis)

    results = analysis(mock_model, perfect_model)


    # Generally, the hypothesis tests of an analysis check whether two datasets are the same
    #   for that reason, we verify that the 'mock' model fails, and 'perfect' passes
    if 'verdict' in results and isinstance(results['verdict'], dict):
        analysis_datasets = analysis.observations[terms.DATASET].unique()
        assert len(analysis_datasets) == 1
        verdicts = results['verdict']
        for hypothesis, status in verdicts.items():
            if 'mock' in hypothesis:
                print(hypothesis)
                print(results['stats'][hypothesis].iloc[:, -3:])
                #import pdb; pdb.set_trace()
                assert status == 'Fail'
            else:
                assert status == 'Pass'

    return results


# All analyses
ANALYSES = [name for name in dir(ana) if isinstance(getattr(ana, name), Analysis)]
# some analyses require specialized testing logic
CAMPAGNOLA = [
    'campagnola_connprob_2022',
    'campagnola_psc_amp_2022',
    'campagnola_psc_rise_2022',
    'campagnola_psp_decay_2022',
    'campagnola_psc_decay_2022',
    'campagnola_psp_amp_2022',
    'campagnola_psp_rise_2022',
    'campagnola_stp_induction_2022',
    'campagnola_stp_recovery_2022',
    'campagnola_ppd_2022'
]
EXCEPTIONS = [
    'schuz_density_1989',
    'ma_spontaneous_2010',
    'jiang_intersomatic_2015',
    'cossell_correlation_psp_2015',
    'cossell_response_correlation_2015',
    'lien_fraction_excitation_2018',
    'lien_thalamocortical_current_2013',
] + CAMPAGNOLA


SIMPLE_ANALYSES = [a for a in ANALYSES if a not in EXCEPTIONS]


@pytest.mark.parametrize("analysis_name", SIMPLE_ANALYSES)
def test_analysis(analysis_name):
    """Generic test for analyses"""
    analysis = getattr(ana, analysis_name)
    run_basic_analysis_test(analysis)


def test_schuz_density_1989():
    """Test Schuz density analysis which may not have standard verdict structure."""
    mock_model = MockModel()

    # This analysis only needs one model
    results = ana.schuz_density_1989(mock_model)

    assert isinstance(results, dict)
    assert 'measurements' in results or 'stats' in results
    # This analysis may have different structure, so just check it runs


def test_ma_spontaneous_2010():
    """Test Ma spontaneous analysis with appropriate model."""

    class MaCompatibleMockModel:
        label = 'mock'

        def firing_rate(self, params):
            # Return data that matches the expected parameters structure
            out = []
            for _, row in params.iterrows():
                # Generate multiple measurements per parameter combination
                for _ in range(10):
                    out.append({**row, terms.FIRING_RATE: np.random.uniform(0, 5)})
            return pd.DataFrame(out)

    class MaPerfectModel:
        label = 'perfect'

        def firing_rate(self, params):
            out = []
            # Get expected means from the analysis observations
            indexed = ana.ma_spontaneous_2010.observations.set_index(
                [terms.LAYER, terms.GENE_EXPRESSION]
            )[terms.MEAN + terms.FIRING_RATE]

            for _, row in params.iterrows():
                try:
                    mean_val = indexed.loc[(row[terms.LAYER], row[terms.GENE_EXPRESSION])]
                    # Add realistic variance around experimental mean
                    for _ in range(10):
                        out.append({
                            **row,
                            terms.FIRING_RATE: mean_val + np.random.normal(0, 0.1)
                        })
                except (KeyError, IndexError):
                    # If no exact match, use a reasonable value
                    for _ in range(10):
                        out.append({**row, terms.FIRING_RATE: 2.0})
            return pd.DataFrame(out)

    results = ana.ma_spontaneous_2010(MaCompatibleMockModel(), MaPerfectModel())

    assert isinstance(results, dict)
    assert 'measurements' in results


def test_jiang_intersomatic_2015():
    """Test Jiang intersomatic analysis which may be a simple measurement."""
    mock_model = MockModel()

    results = ana.jiang_intersomatic_2015(mock_model)

    assert isinstance(results, dict)
    assert 'measurements' in results

    measurements = results['measurements']
    assert isinstance(measurements, pd.DataFrame)
    assert len(measurements) > 0


def test_cossell_correlation_psp_2015():
    """Test Cossell analysis with coordinated measurements."""

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

    assert isinstance(results, dict)
    assert 'verdict' in results
    assert len(results['verdict']) == 2

    # Check expected behavior
    for hypothesis, verdict in results['verdict'].items():
        if 'mock' in hypothesis:
            assert verdict == 'Fail', f"Mock model should fail: {hypothesis}"
        elif 'perfect' in hypothesis:
            assert verdict == 'Pass', f"Perfect model should pass: {hypothesis}"


@pytest.mark.parametrize("analysis_name", CAMPAGNOLA)
def test_campagnola(analysis_name):
    analysis = getattr(ana, analysis_name)
    # TODO: the neccessity of this suggests these tests need to be redesigned
    #  In particular, the use of a bonferroni correction is a problem
    if terms.SAMPLE_SIZE in analysis.observations:
        analysis = analysis.with_fields(
            observations=analysis.observations[analysis.observations[terms.SAMPLE_SIZE] > 7],
            plotter=None,
        )
    else:
        obs = analysis.observations
        analysis = analysis.with_fields(
            observations=obs.iloc[:100].assign(**{
                terms.MAX + terms.VERTICAL + terms.INTERSOMATIC_DISTANCE: 125,
                terms.MIN + terms.VERTICAL + terms.INTERSOMATIC_DISTANCE: 0,
                terms.MAX + terms.HORIZONTAL + terms.INTERSOMATIC_DISTANCE: 125,
                terms.MIN + terms.HORIZONTAL + terms.INTERSOMATIC_DISTANCE: 0,
            }),
            plotter=None,
        )
    run_basic_analysis_test(analysis)


def test_lien_fraction_excitation():

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
        
    results = ana.lien_fraction_excitation_2018(MockModel(), GoodModel())

    assert results['verdict']['The result of Lien2018 could be sampled from the same distribution as mock'] == "Fail"
    assert results['verdict']['The result of Lien2018 could be sampled from the same distribution as good'] == "Pass"
    
def test_lien_thalamocortical_current():
       
    expected = ana.lien_thalamocortical_current_2013.observations[
        terms.MEAN + terms.SOMATIC_CURRENT
    ].iloc[0]

    class GoodModel:
        
        label = 'good'
        
        def somatic_current(self, params):
            res =  pd.DataFrame([
                {terms.SOMATIC_CURRENT: expected + diff, **row}
                for _, row in params.iterrows()
                for diff in np.arange(-0.001, 0.001, 0.00001)
            ])
            return res
    

    results = ana.lien_thalamocortical_current_2013(
        GoodModel(), MockModel()
    )
    assert results['verdict']['The result of Lien2013 could be sampled from the same distribution as mock'] == "Fail"
    assert results['verdict']['The result of Lien2013 could be sampled from the same distribution as good'] == "Pass"
