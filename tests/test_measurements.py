import analysis_library.measurements as msr


def test_cell_density_call():

    class MockCDModel:

        def num_cells(self, params):
            return 123


        def region_volume(self, params):
            return 456

    cd = msr.CellDensity()
    assert cd(MockCDModel(), {'param': 'bla'}) == {msr.terms.CELL_DENSITY: 123/456}
