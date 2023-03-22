================================
Analysis-neuro (actual name TBD)
================================


Purpose
-------

This is the library where we store analyses and validations of neuroscientific models.
Through the use of a separation between model implementation and scientific analysis, analysis code can be reused on past, future, and non-bluebrain models without modification.
This repository is for the analysis code: all model-specific code should go in bluebrain-models 

Any reusable abstractions or tools to make creating new analyses easier can also be created here.

Features
========

The terminology module establishes a common terminology used in the dataframes that are used for analysis. the Terms defined there specify a column header and describe the nature of the column contents.

The Analysis class allows the straightforward creation of an analysis by specifying the Term of a the quantity to be measured for the analysis, a set of observations using the terminology to describe the experimental circumstances under which the quantity is to be measured, and any plots or statistical tests desired. The plotting and statistical testing functions follow a fixed format and can be either selected from the plots and stats submodules, or created as needed. Once an analysis is initialized it is a callable and can be called on a model object which implements the necessary measurement methods, generating a dict report containing measurements, documentation, plots, and the results of the statistical tests.

Note that the terminology, and by extension the observations describe the measurement as performed on a real animal, and so do not reference any model-specific elements and can therefoe be used to describe the validation conditions of any model which has them in its scope.


Contributing
============

All aspects of the library are open to modification as required by user needs.
Any new terminology required should be documented in the terminology module.
Analyses created should be saved within the analyses module, and must be a callable accepting one or more model objects and returning a dict.

To contribute, simply clone the repository, make your desired changes alongside one or more automated tests for the changes, then create a branch, push, and create a pull request.
After a review process and any necessary changed your improvements will be accepted, and can be used by any other user of the library.


Testing strategy
----------------

We rely on two levels of tests: the first is for whole analyses run on 'mock' models which provide fake data. This works best for analyses that test a hypothesis: provide data that you know will pass the hypothesis test in once case and test that the analysis does so, and pass data that you know will fail the hypothesis test and test that the analysis does so.

The second is unit tests for any reusable or sufficiently complex abstractions created.

