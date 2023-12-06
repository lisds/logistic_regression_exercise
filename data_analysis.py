# you need to modify this script so that it performs logistic regression analysis
# on the CKD data
import cost_function_utils
import pandas as pd
import numpy as np
from IPython.display import display

# imported modules
import plotly.express as px
from scipy.optimize import minimize
import cost_function_utils
import statsmodels.formula.api as smf

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

ckd_data['Class_Dummy'] = ckd_data['Class'].replace(['notckd', 'ckd'], [0, 1])

# First, this script should plot Class as a function of Hemoglobin.

px.scatter(ckd_data['Class'], ckd_data['Hemoglobin'])

# Fit your model (Class ~ Hemoglobin) using minimize (by importing your cost function cost_function_utils.py).

estimate_minimize = minimize(
    cost_function_utils.logistic_regression_cost_function, 
    x0=[1, -5], 
    args=(ckd_data['Class_Dummy'], ckd_data['Hemoglobin'])
)

# You should then fit the same model using statsmodels.

log_reg_mod = smf.logit('Class_Dummy ~ Hemoglobin', data=ckd_data)
fitted_log_reg_mod = log_reg_mod.fit()

intercept_log_red_mod, slope_log_red_mod = fitted_log_reg_mod.params

# You should write some tests to check that minimize and statsmodels are producing similar parameter estimates (HINT: you may want to investigate np.isclose() and assert - ask us for help with this also).

print(estimate_minimize.x[0], estimate_minimize.x[1])
print(intercept_log_red_mod, slope_log_red_mod)

    # tried to assert that the two values are the same, but they are not
assert np.isclose(intercept_log_red_mod, estimate_minimize.x[0])
assert np.isclose(slope_log_red_mod, estimate_minimize.x[1])

# You should then create a new plot, showing Class ~ Hemoglobin, but also showing the predictions from your logistic regression model.

