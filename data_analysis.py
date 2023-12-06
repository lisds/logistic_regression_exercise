# you need to modify this script so that it performs logistic regression analysis
# on the CKD data
import cost_function_utils
import pandas as pd
import numpy as np
import matplotlib as plt
from IPython.display import display
import plotly.express as px
from scipy.optimize import minimize 
import cost_function_utils

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

ckd_data = pd.read_csv('CKD full.csv')

ckd_data['Class_Dummy']= ckd_data['Class'].replace(['notckd', 'ckd'], [0, 1])

#plot the data :)
px.scatter(ckd_data['Class'], ckd_data ['Hemoglobin'])

#Minimizaaaation!!!
Minimization_Estimation = minimize(
    cost_function_utils.logistic_regression_cost_function, 
    x0=[-5, 1], 
    args=(ckd_data['Class_Dummy'], ckd_data['Hemoglobin'])
    )

#Stats model fitting 
log_reg_mod = smf.logit('Class_Dummy ~ Hemoglobin', data=ckd_data)
fitted_log_reg_mod = log_reg_mod.fit()

intercept_log_red_mod, slope_log_red_mod = fitted_log_reg_mod.params

# these assertions are not working
assert np.isclose(intercept_log_red_mod, Minimization_Estimation.x[0])
assert np.isclose(slope_log_red_mod, Minimization_Estimation.x[0])

print(Minimization_Estimation.x[0], Minimization_Estimation.x[1])
print(intercept_log_red_mod, slope_log_red_mod)