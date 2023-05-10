================================
Analysis-neuro (actual name TBD)
================================


Purpose
=======

Computational models, whether developed iteratively or built from scratch, are ultimately intended to explain and predict observable phenomena.
The precise procedure for validating a model or extracting novel insights from it will often depend on the model.
For us, this means that from one version of a model to the next, evaluating whether previous results still hold often requires manually rewritin.. code, editing config files, and so forth.
This is error-prone and takes up time that is better spent adding new capacities to the model or constructing new analyses.

When it comes to comparing different modeling approaches, the difference is even more stark, and new models are often only validated against a miniscule subset of the observations which they seek to explain.

This library aims to be a growing, standardized library of neuroscientific validations.
All model-specific details are abstracted away so that the validations can be applied to any model in its scope.
This requires the construction of a ``Model`` object which manages the model's implementation details and maps its properties onto scientifically meaningful observables.
For blue brain circuit models, see the `bluebrain-models <https://bbpgitlab.epfl.ch/circuits/personal/bluebrain-models>`_ library for an example.

Although the primary purpose of the library is standardizing validations, in principle, any analysis involving scientifically meaningful properties can be constructed using it, and re-used across models.

Features
========

The following features are provided to aid in the construction of standardized validations:
 - A module of standardized terminology (``analysis_neuro.terminology``).
 - The ``Analysis`` class, which can be initialized with standardized experimental data, statistical hypothesis tests, and plotter objects. This is subsequently called on ``Model`` objects.
 - Statistical hypothesis tests (``analysis_neuro.stats``) and plotting tools (``analysis_neuro.plots``).
 - A library of existing model validations (``analysis_neuro.analyses``).

Usage
=====

A new validation can be defined using the ``Analysis`` class:

.. code-block::

    from analysis_neuro import Analysis, stats, plots
    new_validation_name = Analysis(
       observations=pd.read_csv("path/to/experimental/data.csv"),
       measurement=terms.CONNECITON_PROBABILITY, # the real-world property we are interested in
       plotter=plots.crossplot, # visualize the comparison with a crossplot
       stats=stats.binom_test, # use a binomial test to check whether the model's estimate matches the observed value
       verdict=stats.PValueThreshold(0.05)) # regard the validation as failed if the p-value is below 0.05


This validation can then be run on any appropriate ``Model`` object, e.g.:

.. code-block::

    from bluebrain_models import BBPCircuit

    validation_report = new_validation_name(BBPCircuit("path/to/circuit/config.json"))


The standardized data provided must be a DataFrame (in this example loaded from a .csv file) in which each row corresponds to one measured sample value, and each column to a variable defined in the ``terminology`` module. For example:


+--------------------+-------------------+---------------------+------------------------+------------------------------------+
| presynaptic region | presynaptic layer | postsynaptic region | connection probability | max retinotopic_distance (degrees) |
+====================+===================+=====================+========================+====================================+
| VISp               | L23               | VISlm               |                  0.4   |                               30   |
+--------------------+-------------------+---------------------+------------------------+------------------------------------+
| VISp               | L5                | VISlm               |                  0.1   |                               30   |
+--------------------+-------------------+---------------------+------------------------+------------------------------------+

where each column header is in ``terminology`` either directly (e.g ``terms.CONNECTION_PROBABILITY``) or built from combination of terms (e.g. ``terms.PRESYNAPTIC + terms.REGION``).

If no existing terminology is defined for a relevant variable, the user should define it by adding it to the module:

.. code-block::

    # (inside analysis_neuro/terminology.py)
    CONNECTION_PROBABILITY = Term(
         "connection probability",
         "The probability for a random pair of cells in a pathway to have at least one synapse between them")

    measurements[CONNECTION_PROBABILITY] = {'method_name': 'connection_probability'}


TODO: we plan to make the latter step unnecessary in the future.

The ``Model`` object should define a method for the desired measurement:

.. code-block::

    class MyModel:
       ...
       def connection_probability(self, parameters): # see 'method_name' above
           connprob = ......
           ...
           return parameters.assign(**{terms.CONNECTION_PROBABILITY: connprob})

[TODO Austin: I'm not sure what the below paragraph means. Do you mean this method will be called on the model output's data, the validation data, or both?]

where parameters will be a standardized dataframe containing the parameters of the measurement (e.g. brain region, pre and post synaptic cell parameters, intersomatic distance), one row for each unique parameter combination. The return value is of the same format, with one row per measurement sample.


Contributing
============

All aspects of the library are open to modification as required by user needs.
Any new terminology required should be documented in the terminology module.
Analyses created should be saved within the ``Analysis`` module, and must be a callable accepting one or more ``Model`` objects and returning a dict.

To contribute, simply clone the repository, make your desired changes alongside one or more automated tests for the changes, then create a branch, push, and create a pull request.
After a review process and any necessary changed your improvements will be accepted, and can be used by any other user of the library.


Testing strategy
----------------

We rely on two levels of tests: the first is for whole analyses run on 'mock' models which provide fake data. This works best for analyses that test a hypothesis: provide data that you know will pass the hypothesis test in one case and test that the analysis does so, and pass data that you know will fail the hypothesis test and test that the analysis does so.

The second is unit tests for any reusable or sufficiently complex abstractions created.

