# you need to modify this script so that it performs logistic regression analysis
# on the CKD data
import cost_function_utils
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import statsmodels.formula.api as smf

# import data
data = pd.read_csv('ckd_full.csv')

# isolate the only important variables 
class_hem = data.loc[:, ['Hemoglobin', 'Class']]

# rename for ease 
class_hem = class_hem.rename(columns = {'Hemoglobin': 'hemoglobin',
                                    'Class' : 'class' })

# dummy code 1 = has ckd, 0 = notckd
class_hem['class_dummy'] = class_hem['class'].replace(['ckd', 'notckd'], [1,0])

# the x and y variables
hem = class_hem['hemoglobin']
class_d = class_hem['class_dummy']


# create a plot
def plot_hem_class():
    colours = class_hem['class'].map({'ckd': 'red', 'notckd': 'green'})
    class_hem.plot.scatter('hemoglobin', 'class_dummy', c=colours)
    plt.ylabel('Class\n0 = not ckd 1 = ckd')
    plt.xlabel('Hemoglobin')
    plt.scatter([], [], c='green', label='not ckd')
    plt.scatter([], [], c='red', label='ckd')

plot_hem_class();


# now minimize to find best fit 
logisitic_reg_MLL = minimize(cost_function_utils.logistic_regression_cost_function, # Cost function from other file
                             [-7, 1], # initial predictions of intercept and slope 
                             args = (hem,class_d))

#  assigning the parameters to int and slope 
inter_logistic_reg_MLL, slope_logistic_reg_MLL = logisitic_reg_MLL.x

# Now for statsmodel to test 
log_reg_mod = smf.logit('class_dummy ~ hemoglobin', data=class_hem)
fitted_log_reg_mod = log_reg_mod.fit()

#  assigning the parameters to int and slope 
inter_log_red_mod, slope_log_red_mod = fitted_log_reg_mod.params

#  Test to see if the two functions produce the smae intercept and slope 
assert np.isclose([inter_logistic_reg_MLL], [inter_log_red_mod]), "Intercept Error"
assert np.isclose([slope_logistic_reg_MLL], [slope_log_red_mod]), "Slope Error"


# Now to plot with the predictions 

sm_predictions = fitted_log_reg_mod.predict(class_hem['hemoglobin'])

plot_hem_class()
plt.scatter(hem, sm_predictions,
            label = 'Statsmodels predictions \n(Logistic Regression)', color = 'blue')
plt.legend();