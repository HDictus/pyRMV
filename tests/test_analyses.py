from analysis_neuro import analyses, terms


def test_schuz_density_1989():
    class MockModel:
        def __init__(self, mocked_density, label):
            self.mocked_density = mocked_density
            self.label = label

        def cell_density(self, parameters):
            assert all(parameters[terms.REGION] == "VISp")
            assert list(parameters[terms.LAYER]) == ["L23", "L4", "L5", "L6"]
            out = parameters.copy()
            out[terms.CELL_DENSITY] = self.mocked_density
            return out

    mock1 = MockModel([100, 200, 300, 400], "mock1")
    mock2 = MockModel([1000, 2000, 3000, 4000], "mock2")
    report = analyses.schuz_density_1989(mock1, mock2)
    measured = report["measurements"]
    assert all(
        measured[measured["dataset"] == "mock1"][terms.CELL_DENSITY]
        == [100, 200, 300, 400]
    )
    assert all(
        measured[measured["dataset"] == "mock2"][terms.CELL_DENSITY]
        == [1000, 2000, 3000, 4000]
    )
    assert report["figures"]
