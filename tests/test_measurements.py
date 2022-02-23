import analysis_library.measurements as msr


def test_cell_density_call():

    class MockCDModel:

        def cell_density(self, params):
            return 123 / 456

    cd = msr.CellDensity()
    assert cd(MockCDModel(), {'param': 'bla'}) == {msr.terms.CELL_DENSITY: 123/456}
