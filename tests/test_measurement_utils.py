from analysis_neuro.measurement_utils import measure, extract_parameters
import pandas as pd


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

    def measure_density(model, parameters, measurements_library):
        v = measure(model, volume, parameters, measurements_library)
        m = measure(model, mass, parameters, measurements_library)
        rho = m[mass] / v[volume]
        return v.drop(columns=[volume]).assign(**{density: rho})

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


def test_extract_parameters():
    measurement = pd.DataFrame({
        'param a': [1, 1, 2, 2, 3],
        'param b': [1, 2, 1, 1, 2],
        'measured': [1, 2, 3, 4, 5]})

    # measurement and duplicate parameter sets should be removed
    pd.testing.assert_frame_equal(
        extract_parameters(measurement, 'measured'),
        pd.DataFrame({
            'param a': [1, 1, 2, 3],
            'param b': [1, 2, 1, 2]}))

    # can work without measurement
    pd.testing.assert_frame_equal(
        extract_parameters(measurement[['param a', 'param b']]),
        pd.DataFrame({
            'param a': [1, 1, 2, 3],
            'param b': [1, 2, 1, 2]}))

    # doesn't drop na values
    df = pd.DataFrame({
        'a': [1, 2, None, None, None],
        'b': ['b', 'c', 'd', 'e', 'e'],
    })
    pd.testing.assert_frame_equal(
        extract_parameters(df),
        pd.DataFrame({
            'a': [1, 2, None, None],
            'b': ['b', 'c', 'd', 'e']
        })
    )