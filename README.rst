================================
Analysis-neuro (actual name TBD)
================================


Purpose
-------

This is the library where we store analyses and validations of neuroscientific models.
Through the use of a separation between model implementation and scientific analysis, analysis code can be reused on past, future, and non-bluebrain models without modification.
This repository is for the analysis code: all model-specific code should go in bluebrain-models 

Any reusable abstractions or tools to make creating new analyses easier can also be created here.


Contributing
============

Whenever you write an analysis with or without using this library, create a pull request with your analysis (even just as a notebook or script).
No need to clean it up first, even if it's rubbish.
We've all written rubbish before and no one will judge.
The sooner it is out there the better, because that way people also know what you are working on and can more easily avoid reinventing your wheels.
When time allows, you can work together with others to turn the analysis provided in the pull request into a reusable piece of code.
With practice, you should be able to write pretty clean analyses on the first try - and do so quickly thanks to the abstractions we create together in this library.


Also, if you have any old analyses or validations which are still potentially useful, throw them in some pull requests too, and we can get round to refactoring them as soon as someone wants something similar.



Testing strategy
----------------

Once an analysis has been created for a model, the model and analysis in question should be added to the growing library of regression tests.
After relevant changes to the framework, these can be run to verify that the analysis still provides the same results.
These also double as integration tests: when we modify the code to support a new kind of model, we can run all applicable analyses which have previously been run on older models using the same set of tests.

Any additional abstractions created should be unit-tested.

