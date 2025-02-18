import pandas as pd
import numpy as np
import pytest as pyt
import warnings
from mock import MagicMock
from analysis_neuro import Analysis, terms, TerminologyError, measurements


MEASURED_THING = terms.Term("measured thing", measurement_method="measured_thing")


class MockModel:
    def __init__(self, num):
        self.num = num
        self.label = str(num)

    def measured_thing(self, params):
        return params.assign(**{MEASURED_THING: self.num})


def test_extracts_parameters_from_observations():
    ana = Analysis(
        measurement=MEASURED_THING,
        observations=pd.DataFrame(
            {
                MEASURED_THING: [100, 200, 300, 400, 500],
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


def test_single_parameter_from_repeated_observations():
    ana = Analysis(
        measurement=MEASURED_THING,
        observations=pd.DataFrame(
            {
                MEASURED_THING: [100, 200, 300, 400, 500],
                terms.DATASET: "blabla",
                "a param": ["a", "a", "a", "a", "a"],
                "nother param": ["b", "b", "b", "c", "c"],
                terms.NOTES: ["wha", "who?", "when?", "why?", "wherefore?"],
            }
        ),
    )
    pd.testing.assert_frame_equal(
        ana.parameters,
        pd.DataFrame(
            {
                "a param": ["a", "a"],
                "nother param": ["b", "c"],
            }
        ),
    )


def test_requests_measurements():
    ana = Analysis(
        measurement=MEASURED_THING,
        observations=pd.DataFrame(
            {"a param": [1, 2, 3, 4, 5], "another param": [4, 4, 4, 3, 3]}
        ),
    )

    models = MockModel(1), MockModel(2)
    report = ana(*models)

    expected = pd.concat(
        [
            models[0].measured_thing(ana.parameters).assign(dataset="1"),
            models[1].measured_thing(ana.parameters).assign(dataset="2"),
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
            MEASURED_THING: [100, 200, 300, 400, 500],
            terms.DATASET: "blabla",
            "a param": ["a", "b", "c", "d", "e"],
        }
    )

    ana = Analysis(
        measurement=MEASURED_THING, plotter=plotter, observations=observations
    )
    result = ana(MockModel(4))
    assert result["figures"] == mockfigure
    kwargs = plotter.call_args.kwargs
    pd.testing.assert_series_equal(
        kwargs["y"].reset_index(drop=True),
        pd.Series([100, 200, 300, 400, 500, 4, 4, 4, 4, 4], name=MEASURED_THING),
    )
    pd.testing.assert_series_equal(
        kwargs["x"].reset_index(drop=True),
        pd.Series(["a", "b", "c", "d", "e"] * 2, name="a param"),
    )
    pd.testing.assert_series_equal(
        kwargs["hue"].reset_index(drop=True),
        pd.Series(["blabla"] * 5 + ["4"] * 5, name=terms.DATASET),
    )


def test_runs_plotter_alternate():
    mockfigure = MagicMock()

    def plotter(data, dependent, independent, compare):
        return mockfigure

    observations = pd.DataFrame(
        {
            MEASURED_THING: [100, 200, 300, 400, 500],
            terms.DATASET: "blabla",
            "a param": ["a", "b", "c", "d", "e"],
        }
    )

    ana = Analysis(
        measurement=MEASURED_THING, plotter=plotter, observations=observations
    )
    assert ana(MockModel(4))["figures"] == mockfigure


def test_plots_only_by_varying_params():
    plotter = MagicMock()
    observations = pd.DataFrame(
        {
            MEASURED_THING: [100, 200, 300, 400, 500],
            terms.DATASET: "blabla",
            "a param": ["a", "b", "c", "d", "e"],
            "a constant": "c",
        }
    )

    ana = Analysis(
        measurement=MEASURED_THING, plotter=plotter, observations=observations
    )
    ana(MockModel(4))
    kwargs = plotter.call_args.kwargs
    pd.testing.assert_series_equal(
        kwargs["x"].reset_index(drop=True),
        pd.Series(["a", "b", "c", "d", "e"] * 2, name="a param"),
    )
    pd.testing.assert_series_equal(
        kwargs["y"].reset_index(drop=True),
        pd.Series([100, 200, 300, 400, 500, 4, 4, 4, 4, 4], name=MEASURED_THING),
    )
    pd.testing.assert_series_equal(
        kwargs["hue"].reset_index(drop=True),
        pd.Series(["blabla"] * 5 + ["4"] * 5, name=terms.DATASET),
    )


def test_plots_with_multiple_params():
    plotter = MagicMock()
    observations = pd.DataFrame(
        {
            MEASURED_THING: [100, 200, 300, 400, 500, 600],
            terms.DATASET: "blabla",
            "layer": ["L1", "L23", "L4", "L5", "L6", "L6"],
            "mtype": ["NGC", "NGC", "MC", "MC", "LBC", None],
        }
    )

    ana = Analysis(
        measurement=MEASURED_THING, plotter=plotter, observations=observations
    )
    ana(MockModel(4))
    kwargs = plotter.call_args.kwargs
    pd.testing.assert_series_equal(
        kwargs["y"].reset_index(drop=True),
        pd.Series([100, 200, 300, 400, 500, 600, 4, 4, 4, 4, 4, 4], name=MEASURED_THING),
    )
    pd.testing.assert_series_equal(
        kwargs["x"].reset_index(drop=True),
        pd.Series(
            ["L1 NGC", "L23 NGC", "L4 MC", "L5 MC", "L6 LBC", "L6"] * 2, name="layer, mtype"
        ),
    )
    pd.testing.assert_series_equal(
        kwargs["hue"].reset_index(drop=True),
        pd.Series(["blabla"] * 6 + ["4"] * 6, name=terms.DATASET),
    )


def test_invalid_observation_supplied():
    observations = 1234
    with pyt.raises(ValueError) as ve:
        ana = Analysis(measurement=MEASURED_THING, observations=observations)
    assert "must be a pandas.DataFrame of the form:" in str(ve.value)


def test_plotter_stats_verdict_signatures():
    invalid_things = ("123", lambda a: a)
    for kw in ["stats", "plotter", "verdict"]:
        with pyt.raises(ValueError) as ve:
            ana = Analysis(
                measurement=MEASURED_THING,
                observations=pd.DataFrame({}),
                **{kw: invalid_things[0]}
            )
        assert "must be a callable of the form:" in str(ve.value)
        with pyt.raises(ValueError) as ve:
            ana = Analysis(
                measurement=MEASURED_THING,
                observations=pd.DataFrame({}),
                **{kw: invalid_things[1]}
            )
        assert "must be a callable of the form:" in str(ve.value)


def test_runs_statistical_tests():
    mockresults = {'hypothesis': 'data-frame'}
    mockstats = MagicMock(return_value=mockresults)

    observations = pd.DataFrame(
        {
            MEASURED_THING: [100, 200, 300, 400, 500],
            terms.DATASET: "blabla",
            "layer": ["L1", "L23", "L4", "L5", "L6"],
            "mtype": ["NGC", "NGC", "MC", "MC", "LBC"],
        }
    )

    ana = Analysis(
        observations=observations, measurement=MEASURED_THING, stats=mockstats
    )

    results = ana(MockModel(4))
    assert results["stats"] == mockresults
    mockstats.assert_called_with(
        data=results["measurements"],
        dependent=MEASURED_THING,
        independent=["layer", "mtype"],
        compare=terms.DATASET,
    )
    

def test_runs_multiple_statistical_tests():
    mockresults1 = {'hypothesis1': 'data-frame'}
    mockresults2 = {'hypothesis2': 'data-frame'}
    mockstats1 = MagicMock(return_value=mockresults1)
    mockstats2 = MagicMock(return_value=mockresults2)

    observations = pd.DataFrame(
        {
            MEASURED_THING: [100, 200, 300, 400, 500],
            terms.DATASET: "blabla",
            "layer": ["L1", "L23", "L4", "L5", "L6"],
            "mtype": ["NGC", "NGC", "MC", "MC", "LBC"],
        }
    )

    ana = Analysis(
        observations=observations, measurement=MEASURED_THING, stats=[mockstats1, mockstats2]
    )

    results = ana(MockModel(4))
    assert results["stats"] == {**mockresults1, **mockresults2}
    mockstats1.assert_called_with(
        data=results["measurements"],
        dependent=MEASURED_THING,
        independent=["layer", "mtype"],
        compare=terms.DATASET,
    )
    mockstats2.assert_called_with(
        data=results["measurements"],
        dependent=MEASURED_THING,
        independent=["layer", "mtype"],
        compare=terms.DATASET,
    )



def test_runs_verdict():
    mockresults = {'hypo': 'somedata'}
    mockstats = MagicMock(return_value=mockresults)
    mockverdict = MagicMock(return_value={"hypo": "Pass"})
    observations = pd.DataFrame(
        {
            MEASURED_THING: [100, 200, 300, 400, 500],
            terms.DATASET: "blabla",
            "layer": ["L1", "L23", "L4", "L5", "L6"],
            "mtype": ["NGC", "NGC", "MC", "MC", "LBC"],
        }
    )

    ana = Analysis(
        observations=observations,
        measurement=MEASURED_THING,
        stats=mockstats,
        verdict=mockverdict,
    )

    results = ana(MockModel(4))
    assert results["verdict"] == {"hypo": "Pass"}
    mockverdict.assert_called_with(mockresults)


def test_warns_invalid_term():
    matchstr = (
        "Column header 'not in terms' is not defined in analysis_neuro.terminology"
    )
    with pyt.warns(Warning, match=matchstr) as wrn:
        Analysis(
            observations=pd.DataFrame({"not in terms": [0, 1, 2, 3]}),
            measurement=terms.CELL_DENSITY,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Analysis(
            observations=pd.DataFrame({"Presynaptic " + "mtype": ["PC"]}),
            measurement=terms.CELL_DENSITY,
        )


def test_raises_error_if_nonstring_col():
    with pyt.raises(ValueError):
        Analysis(
            observations=pd.DataFrame({3451: ["PC"]}), measurement=terms.CELL_DENSITY
        )


def test_raises_error_invalid_measurement():
    with pyt.raises(TerminologyError) as te:
        Analysis(
            observations=pd.DataFrame({terms.LAYER: ["L1", "L2", "L3"]}),
            measurement="lololo",
        )
    assert (
        "lololo is not a defined Term."
    ) in str(te.value)


# TODO: test case where observatioons have measurement but not label
# should raise an error? or just put 'biodata' in place?


def test_measure_checks_valid_return_format():
    class ReturnsNumberMock:

        label = "blabla"

        def cell_density(self, parameters):
            return 0

    with pyt.raises(ValueError):
        Analysis(
            observations=pd.DataFrame({terms.LAYER: ["L1"], terms.REGION: ["VISp"]}),
            measurement=terms.CELL_DENSITY,
        )(ReturnsNumberMock())

    class ReturnsNoMeasurement:

        label = "lololo"

        def cell_density(self, parameters):
            return parameters

    with pyt.raises(ValueError):
        Analysis(
            observations=pd.DataFrame({terms.LAYER: ["L1"], terms.REGION: ["VISp"]}),
            measurement=terms.CELL_DENSITY,
        )(ReturnsNoMeasurement())

    class ReturnsIncompleteParams:

        label = ("hihihi",)

        def cell_density(self, parameters):
            out = parameters[[terms.LAYER]]
            out[terms.CELL_DENSITY] = 10
            return out

    with pyt.raises(ValueError):
        Analysis(
            observations=pd.DataFrame({terms.LAYER: ["L1"], terms.REGION: ["VISp"]}),
            measurement=terms.CELL_DENSITY,
        )(ReturnsIncompleteParams())


def test_uses_copy_of_observations():
    """When testing stuff out, it is possible to accidentally mutate the observations
    dataframe of an Analysis.
    
    If the same dataframe instance that is mutated is used by the analysis, 
    this leads to some hard to track down bugs in testing code.
    To avoid this, we should make sure that the observations property of an Analysis
    is a copy.
    
    Similarly, if a dataframe is used to initialize an analysis and subsequently changed
    this can lead to unexpected behavior by the analysis. 
    So we also check that the .observations of the analysis are not affected
    when we modify the original dataframe
    """
    df = pd.DataFrame({
        'a parameter': [1, 2, 3],
        terms.CELL_DENSITY: [1, 2 , 3],
        terms.DATASET: 'label'
    })
    ana = Analysis(observations=df, measurement=terms.CELL_DENSITY)
    obscopy = ana.observations
    obscopy['a parameter'] = 4
    # the original dataframe should be unchanged
    assert all(df['a parameter'] == [1, 2, 3])
    # when we change the original dataframe
    df['a parameter'] = 3
    # the analysis' observations should be unchanged
    assert all(ana.observations['a parameter'] == [1, 2, 3])
  
  
def test_defaults_dataset():

    df = pd.DataFrame({
        'da': [1, 2, 3],
        terms.CELL_DENSITY: [1, 2 , 3]
    })
    ana = Analysis(observations=df, measurement=terms.CELL_DENSITY)
    assert all(ana.observations[terms.DATASET] == 'experiment')


def test_ignores_unmeasured_values():
    df = pd.DataFrame({
        'parameter': [1, 1, 2, 3, 3, 4, 4],
        MEASURED_THING: 5,
        terms.DATASET: 'experiment'
    })
    ana = Analysis(observations=df, measurement=MEASURED_THING)
    
    class PartialModel:
        
        label = 'partial'
        
        def measured_thing(self, parameters):
            return pd.DataFrame({
                'parameter': [1, 3,  3],
                MEASURED_THING: 2
            })
            
    measured = ana(PartialModel())['measurements']
    pd.testing.assert_frame_equal(
        measured[['parameter', terms.DATASET]].reset_index(drop=True),
        pd.DataFrame({
            'parameter': [1, 1, 3, 3, 1, 3, 3],
            terms.DATASET: ['experiment'] * 4 + ['partial'] * 3,
        })
    )

def test_infer_independent_from_non_varying():
    data = pd.DataFrame({
            MEASURED_THING: 0, 'a': 4, 'b': 6,
            terms.DATASET: ['ah', 'ah'],
            
    })
    analysis = Analysis(
        observations=data, measurement=MEASURED_THING
    )
    assert analysis.independent == ['a', 'b']


def test_set_dependent_independent_compare():
    data = pd.DataFrame({
            MEASURED_THING: 0, 'a': 4,
            terms.DATASET: ['ah']
        })
    analysis = Analysis(
        observations=data, measurement=MEASURED_THING
    )
    assert analysis.independent == ['a']
    assert analysis.dependent == MEASURED_THING
    assert analysis.compare == terms.DATASET
    
    analysis = Analysis(
        observations=data,
        independent=terms.DATASET,
        dependent='a',
        compare=MEASURED_THING,
        measurement=MEASURED_THING
    )
    assert analysis.independent == [terms.DATASET]
    assert analysis.dependent == 'a'
    assert analysis.compare == MEASURED_THING
    

@pyt.mark.xfail
def test_with_fields():
    """Not sure yet how to nicely test this."""
    data = pd.DataFrame({
            MEASURED_THING: 0, 'a': 4,
            terms.DATASET: ['ah']
        })
    analysis = Analysis(
        observations=data, measurement=MEASURED_THING
    )
    fields =['measurement', 
            'observations',
            'plotter',
            'stats',
            'verdict',
            'doc',
            'dependent',
            'independent',
            'compare']
    for field in fields:
        mockobj = MagicMock()
        new_ana = analysis.with_fields(**{field: mockobj})
        assert getattr(new_ana, field) == mockobj