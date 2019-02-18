--------------
About Anaconda
--------------

Anaconda is a free and open-source distribution of the Python and R programming languages for scientific computing used in various fields of data science, machine learning, large-scale data processing, predictive analytics, etc. Package versions are managed by the package management system named as "conda". The Anaconda distribution is used by over 6 million users and includes more than 1400 popular data-science packages suitable for Windows, Linux, and MacOS.

Link to install Anaconda: https://www.anaconda.com

------------------------------
Installing Module Dependencies
------------------------------

numpy
scipy
pandas

Note: For Windows system, you can use "pip install <package_name>". Example: "pip install pandas"

----------------------------------------------------
Required files/folders to run the "dtree.py" program
----------------------------------------------------

1. Copy the folders "data_sets1" and "data_sets2" and this file "dtree.py" to the location where you want to run the "dtree.py".
2. From the current folder, run the below commands to generate the decision tree.

----------------------
Command Line Arguments
----------------------

Example:  python dtree.py <L> <K> <training-set> <validation-set> <test-set> <to-print>

#For data_set1, use command: python dtree.py "3" "5" "data_sets1/training_set.csv" "data_sets1/test_set.csv" "data_sets1/validation_set.csv" "yes"
#For data_set2, use command: python dtree.py "3" "5" "data_sets2/training_set.csv" "data_sets2/test_set.csv" "data_sets2/validation_set.csv" "yes"

Options for the above argument values:

<L> - any number (larger number will take more time to simulate)
<K> - any number (larger number will take more time to simulate)
<training-set> - Training Data Set file location with filename (must be .csv file)
<validation-set> - Validation Data Set file location with filename (must be .csv file)
<test-set> - Test Data Set file location with filename (must be .csv file)
<to-print> - "yes" or "no"

----------------------------------------------
Option to save the stdout to a file (Optional)
----------------------------------------------

1. To save the stdout (terminal output) results to a file name "STANDARD_OUTPUT_RESULTS_<number>.txt". Uncomment the block comment from line number 27 to 50 and the line number 463 "sys.stdout.close()" at the end of "dtree.py" program.
2. The below code will create "STANDARD_OUTPUT_RESULTS_.txt" file as the base file and create a new file with incrementing number as "STANDARD_OUTPUT_RESULTS<incrementing_number>.txt", if the file already exists.

------------------------------------------------------------------------------------
List of import packages used to include certain functions used in "dtree.py" program
------------------------------------------------------------------------------------

import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
import ast
import copy
import math
import random

---------------------------------------------------
List of functions/classes implemented in "dtree.py" program
---------------------------------------------------

1. main
2. calculateAccuracy
3. findAttributeWinner
4. removeNode
5. calcEntropy
6. calcVarianceImpurity
7. calcGain
8. pruneTreeFunction

Class Node contains the following:
1. __init__
2. setNodeValue

Class Tree contains the following:
1. __init__
2. createDecisionTree
3. countNumOfNodes
4. countNumOfLeafs
5. printDecisionTreeLevels
6. printDecisionTree
7. predictDecisionTreeLabel
