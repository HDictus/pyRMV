from analysis_neuro.measurement_utils import measure
from analysis_neuro import terms
import pandas as pd

"""
def test_osi_optimal_tf():

    osiparams = pd.DataFrame({
        terms.STIMULUS: allen_brain_observatory.drifting_gratings,
        'other variable': ['a', 'a', 'b', 'b'],
        terms.TEMPORAL_FREQUENCY: ['optimal', 2, 'optimal', 2]})

    class MockModel:

        def firing_rate(self, parameters):
            out = []
            nrns = [0, 1, 2]
            pref_tfs = [2, 8, 4]
            pref_oris = [90, 180, 270]
            for i, row in parameters.iterrows():
                for nrn, tf, ori in zip(nrns, pref_tfs, pref_oris):
                    # only show a strong response at preferred TF and ORI
                    if row[terms.TEMPORAL_FREQUENCY] != tf:
                        out.append({**row, terms.CELL_ID: nrn, terms.FIRING_RATE: 1})
                        continue
                    if row[terms.STIM_ORIENTATION] == ori:
                        out.append({**row, terms.CELL_ID: nrn, terms.FIRING_RATE: 2})
            return pd.DataFrame(out)

    osi = measure_OSi(MockModel(), osiparams)
    pd.testing.assert_frame_equal(
        osi,
        pd.DataFrame({
            terms.STIMULUS: allen_brain_observatory.drifting_gratings,
            terns.TEMPORAL_FREQUENCY: ['optimal', 'optimal', 'optimal',
                                       2, 2, 2,
                                       'optimal', 'optimal', 'optimal',
                                       2, 2, 2],
            'other variable': ['a', 'a', 'a',
                               'a', 'a', 'a',
                               'b', 'b', 'b',
                               'b', 'b', 'b'],
            terms.ORIENTATION_SELECTVITY: [
                0.5, 0.5, 0.5,
                0.5, 0, 0,
                0.5, 0.5, 0.5,,
                0.5, 0, 0
                ]}
        )

# other good usecases: siegle TF/SF tuning at optimal orientation / direction


# maybe it's best to just let an analysis object recieve a method that measures a thing?
# then this can be organized in its own way: e.g. a general method for which partials are passed for specific examples
# how does a model choose to overwrite this then? e.g. if they prefer to estimate the variable directly.
        # yeah, this is less than ideal.

class OrientationSelectivity():

    def from_firing_rates(parameters):
        return

    def from_membrane_potential(parameters):
        return

    def etc():
        return

@measures(terms.ORIENTATION_SELECTVITY).using(terms.FIRING_RATE)
def orientation_selectivity(model, parameters):
    return
# and what if there is an analysis where we compare orientation selectivity of subtrheshold to spiking??
# but osi may depend on e.g. deconvolved CA thing rather than firing rate...
# whether to prefer one based on firing rate or ca fluoresence is a matter of parameters, not availability
# but when ca fluoresence is not available then
# orientation selectivity is only meaningful coupled to the response type
# e.g. there is no reason to believe super and subthreshold tuning to be identical.

# advantage is to regard them as one until required otherwise: then we add a parameter as necessary
# e.g. response_type
# the model can group by response type as appropriate... or a helper can do that?
#

# but putting that on the model again increases the work needed to keep up with the capabilities
"""

def test_measures_with_method():

    density = 'density (kg/m^3)'
    params = pd.DataFrame({'_': [0, 0]})

    class MockModelWithDens:

        def density(self, parameters):
            return parameters.assign(**{density: 100})

    dens = measure(
        MockModelWithDens(), density, params,
        measurements_library={density: {'method name': 'density'}})
    assert dens[density].values[0] == 100


def test_measure_on_basis_of_others():

    density = 'density (kgm^-3)'
    volume = 'volume (m^3)'
    mass = 'mass (kg)'

    class MockModel:

        def volume(self, parameters):
            return parameters.assign(**{volume: 100})

        def mass(self, parameters):
            return parameters.assign(**{mass: 200})

    def measure_density(model, parameters):
        vol_mass = measure(model, (volume, mass), parameters)
        rho = vol_mass[mass] / vol_mass[volume]
        return vol_mass.drop(columns=[volume]).assign(**{density: rho})

    measurements_library = {
        volume: {'method name': 'volume'},
        mass: {'method name': 'mass'},
        density: {'method name': 'density',
                  (volume, mass): measure_density}}

    params = pd.DataFrame({'_': [0]})

    dens = measure(
        MockModel(), density, params,
        measurements_library=measurements_library)
    assert dens[density].values[0] == 2

    # ensure that the model method still gets priority
    class MockModelWithDens:

        def density(self, parameters):
            return parameters.assign(**{density: 100})

    dens = measure(
        MockModelWithDens(), density, params,
        measurements_library=measurements_library)
    assert dens[density].values[0] == 100


def test_measures_osi_with_firing_rate():

    class MockModel:

        def firing_rate(self, parameters):
            return

    measure(terms.ORIENTATION_SELECTIVITY, MockModel())
