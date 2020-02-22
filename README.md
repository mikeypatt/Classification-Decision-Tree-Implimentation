## CO395 Introduction to Machine Learning: Coursework 1 (Decision Trees)

### Introduction

This repository contains the skeleton code and dataset files that you need 
in order to complete the coursework.

### Data

The ``data/`` directory contains the datasets you need for the coursework.

The primary datasets are:
- ``train_full.txt``
- ``train_sub.txt``
- ``train_noisy.txt``
- ``validation.txt``

Some simpler datasets that you may use to help you with implementation or 
debugging:
- ``toy.txt``
- ``simple1.txt``
- ``simple2.txt``

The official test set is ``test.txt``. Please use this dataset sparingly and 
purely to report the results of evaluation. Do not use this to optimise your 
classifier (use ``validation.txt`` for this instead). 


### Codes

- ``classification.py``

	* Contains the code ``DecisionTreeClassifier`` class.


- ``eval.py``

	* Contains the code for the ``Evaluator`` class.


- ``main.py``

	* Contains the implementation of the classes in classification.py and eval.py

### Instructions

< The programme can be run by simply running 'python3 main.py' from the command line.
The main is structured in the same order as the brief. Answers to the relevant tasks 
are printed out to the command line and the following files are created: "initial_tree.npy", "simple_tree.npy", "simple_tree_pruned.npy" and "initial_tree_pruned.npy" which contains the created trees. 


For Q3.4 and Q3.5, they are implemented using free
functions "data_split", "cross_validation" and "weighted predict" in the main. >



