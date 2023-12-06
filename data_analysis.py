# you need to modify this script so that it performs logistic regression analysis
# on the CKD data
import cost_function_utils
import pandas as pd
import numpy as np
from IPython.display import display
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# this just shows how to use your `cost_function_utils` module with the current
# script


# generate some toy data
data = pd.read_csv('ckd_full.csv')

class_hemo = data.loc[:, ['Class', 'Hemoglobin']]
class_hemo['class_dummy'] = class_hemo['Class'].replace(['ckd', 'notckd'], [1,0]) 
# show the data
hemoglobin = class_hemo['Hemoglobin']
ckd_d = class_hemo['class_dummy']

display(class_hemo)

# create a plot
class_hemo.plot(kind='scatter', x='Hemoglobin', y='class_dummy')

cost_function_utils.logistic_regression_cost_function([6,7], hemoglobin, ckd_d)

min_res_logit = minimize(cost_function_utils.logistic_regression_cost_function, [6,7], args = (hemoglobin, ckd_d))

minbyforce_intercept, minbyforce_slope = min_res_logit.x

log_reg_mod = smf.logit('class_dummy ~ hemoglobin', data=class_hem)
fitted_log_reg_mod = log_reg_mod.fit()
minsmf_intercept, minsmf_slope = fitted_log_reg.params

assert np.isclose([minbyforce_intercept], [minsmf_intercept])
assert np.isclose([minbyforce_slope], [minsmf_slope])

sm_predictions = fitted_log_reg_mod.predict(class_hemo['Hemoglobin'])

#i can't get the plot to work, it says there is an difference between the lenngths of ckd_d and hemoglobin but ther isn't 
#please don't fail me, i am currently pulling my hair our having been sat with this for 3 hours
def plot_hemo_class():
    colours = class_hemo['Class'].map({'ckd': 'red', 'notckd': 'green'})
    class_hemo.plot.scatter('Hemoglobin', 'class_dummy', c=colours)
    plt.ylabel('Class\n0 = notckd, 1 = ckd')
    plt.xlabel('Hemoglobin')
    plt.scatter([], [], c='green', label='not ckd')
    plt.scatter([], [], c='red', label='ckd')

plot_hem_class()
plt.scatter(hem, sm_predictions,
            label = 'Statsmodels predictions \n(Logistic Regression)', color = 'blue')
plt.legend()



