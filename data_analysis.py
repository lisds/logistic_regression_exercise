# you need to modify this script so that it performs logistic regression analysis
# on the CKD data
import cost_function_utils
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from IPython.display import display

# this just shows how to use your `cost_function_utils` module with the current
# script
#cost_function_utils.inverse_logit(v):
#cost_function_utils.logistic_regression_cost_function(intercept_and_slope, x_values, y_values):


data = pd.read_csv('ckd_full.csv')

# show the data
#display(data)


data['CKDlogit'] = data['Class'].replace({'ckd' : 1,'not ckd' : 0})
hemoglobin = data['Hemoglobin']
ckd = data['CKDlogit']

#I'm not sure why but my minimize is not working...
min_res_logit = minimize(cost_function_utils.logistic_regression_cost_function, [1, 0], args=(hemoglobin, ckd))
min_res_logit

logit_inter, logit_slope = min_res_logit.x
predicted_log_odds = logit_inter + logit_slope * hemoglobin
logit_ss_predicted_prob_of_1 = inverse_logit(predicted_log_odds)

plot_hgb_app()
plt.scatter(hemoglobin, logit_ss_predicted_prob_of_1,
            label='Logistic regression curve \n(from sum of squared error on log odds scale)',
            color='gold')
for i in np.arange(len(hemoglobin)):
    plt.plot([hemoglobin[i], hemoglobin[i]], [logit_ss_predicted_prob_of_1[i], ckd[i]], 'k:')
    # the following code line is just to trick Matplotlib into making a new
    # a single legend entry for the dotted lines.
plt.plot([], [], 'k:', label='Errors ($ \\varepsilon $)')

