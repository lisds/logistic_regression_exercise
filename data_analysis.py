# you need to modify this script so that it performs logistic regression analysis
# on the CKD data
import cost_function_utils
import pandas as pd
import numpy as np
from IPython.display import display
from scipy.optimize import minimize
from matplotlib import pyplot as plt

# this just shows how to use your `cost_function_utils` module with the current
# script
cost_function_utils.junk_function()

# generate some toy data
data = pd.DataFrame({'A': np.random.normal(100, 10, 100),
                    'B': np.random.normal(10, 1, 100)})

# show the data
display(data)

# create a plot
data.plot(kind='scatter', x='A', y='B')

ckd_full = pd.read_csv('ckd_full.csv')
ckd_full

ckd_full.plot(kind='scatter', x='Hemoglobin', y='Class')

ckd_full['Class - Binary'] =ckd_full['Class'].replace({'ckd' : 1,'notckd' : 0})

x = ckd_full['Hemoglobin']
y = ckd_full['Class - Binary']
minimized_function = minimize(cost_function_utils.logistic_regression_cost_function, [1, 1], args=(x, y))

minimized_function

import statsmodels.formula.api as smf

model = smf.logit('`Class - Binary` ~ Hemoglobin', data=ckd_full).fit()
