import pandas as pd
from mock import MagicMock

from analysis_neuro import Analysis, measurements, terms

measurements["measured thing"] = {"method name": "thing"}


class MockModel:
    def __init__(self, num):
        self.num = num
        self.label = str(num)

    def thing(self, params):
        return params.assign(**{"measured thing": self.num})


def test_extracts_parameters_from_observations():
    ana = Analysis(
        measurement="measured thing",
        observations=pd.DataFrame(
            {
                "measured thing": [100, 200, 300, 400, 500],
                terms.DATASET: "blabla",
                "a param": ["a", "b", "c", "d", "e"],
                "nother param": ["z", "z", "x", "x", "y"],
                terms.CITATION: "someone_something_someyear",
                terms.NOTES: "yeah just whatever basically",
            }
        ),
    )
    pd.testing.assert_frame_equal(
        ana.parameters,
        pd.DataFrame(
            {
                "a param": ["a", "b", "c", "d", "e"],
                "nother param": ["z", "z", "x", "x", "y"],
            }
        ),
    )


def test_requests_measurements():
    ana = Analysis(
        measurement="measured thing",
        observations=pd.DataFrame(
            {"a param": [1, 2, 3, 4, 5], "another param": [4, 4, 4, 3, 3]}
        ),
    )

    models = MockModel(1), MockModel(2)
    report = ana(*models)

    expected = pd.concat(
        [
            models[0].thing(ana.parameters).assign(dataset="1"),
            models[1].thing(ana.parameters).assign(dataset="2"),
        ]
    )
    pd.testing.assert_frame_equal(report["measurements"], expected)


def test_runs_plotter():
    mockfigure = MagicMock()
    mockaxis = MagicMock()
    mockaxis.get_figure = MagicMock(return_value=mockfigure)
    plotter = MagicMock(return_value=mockaxis)

    observations = pd.DataFrame(
        {
            "measured thing": [100, 200, 300, 400, 500],
            terms.DATASET: "blabla",
            "a param": ["a", "b", "c", "d", "e"],
        }
    )

    ana = Analysis(
        measurement="measured thing", plotter=plotter, observations=observations
    )
    result = ana(MockModel(4))
    assert result["figures"] == mockfigure
    kwargs = plotter.call_args.kwargs
    pd.testing.assert_series_equal(
        kwargs["y"].reset_index(drop=True),
        pd.Series([100, 200, 300, 400, 500, 4, 4, 4, 4, 4], name="measured thing"),
    )
    pd.testing.assert_series_equal(
        kwargs["x"].reset_index(drop=True),
        pd.Series(["a", "b", "c", "d", "e"] * 2, name="a param"),
    )
    pd.testing.assert_series_equal(
        kwargs["hue"].reset_index(drop=True),
        pd.Series(["blabla"] * 5 + ["4"] * 5, name=terms.DATASET),
    )


def test_plots_only_by_varying_params():
    plotter = MagicMock()
    observations = pd.DataFrame(
        {
            "measured thing": [100, 200, 300, 400, 500],
            terms.DATASET: "blabla",
            "a param": ["a", "b", "c", "d", "e"],
            "a constant": "c",
        }
    )

    ana = Analysis(
        measurement="measured thing", plotter=plotter, observations=observations
    )
    ana(MockModel(4))
    kwargs = plotter.call_args.kwargs
    pd.testing.assert_series_equal(
        kwargs["x"].reset_index(drop=True),
        pd.Series(["a", "b", "c", "d", "e"] * 2, name="a param"),
    )
    pd.testing.assert_series_equal(
        kwargs["y"].reset_index(drop=True),
        pd.Series([100, 200, 300, 400, 500, 4, 4, 4, 4, 4], name="measured thing"),
    )
    pd.testing.assert_series_equal(
        kwargs["hue"].reset_index(drop=True),
        pd.Series(["blabla"] * 5 + ["4"] * 5, name=terms.DATASET),
    )


def test_plots_with_multiple_params():
    plotter = MagicMock()
    observations = pd.DataFrame(
        {
            "measured thing": [100, 200, 300, 400, 500],
            terms.DATASET: "blabla",
            "layer": ["L1", "L23", "L4", "L5", "L6"],
            "mtype": ["NGC", "NGC", "MC", "MC", "LBC"],
        }
    )

    ana = Analysis(
        measurement="measured thing", plotter=plotter, observations=observations
    )
    ana(MockModel(4))
    kwargs = plotter.call_args.kwargs
    pd.testing.assert_series_equal(
        kwargs["y"].reset_index(drop=True),
        pd.Series([100, 200, 300, 400, 500, 4, 4, 4, 4, 4], name="measured thing"),
    )
    pd.testing.assert_series_equal(
        kwargs["x"].reset_index(drop=True),
        pd.Series(
            ["L1 NGC", "L23 NGC", "L4 MC", "L5 MC", "L6 LBC"] * 2, name="layer, mtype"
        ),
    )
    pd.testing.assert_series_equal(
        kwargs["hue"].reset_index(drop=True),
        pd.Series(["blabla"] * 5 + ["4"] * 5, name=terms.DATASET),
    )


# TODO: test case where observatioons have measurement but not label
# should raise an error? or just put 'biodata' in place?
