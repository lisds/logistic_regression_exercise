# you need to modify this script so that it performs logistic regression analysis
# on the CKD data
import cost_function_utils
import pandas as pd
import numpy as np
from IPython.display import display

#importing scipy to use minimize
from scipy.optimize import minimize

# this just shows how to use your `cost_function_utils` module with the current
# script
cost_function_utils.junk_function()

#creating the dataframe
data = pd.read_csv('ckd_full.csv')



# show the data
display(data)

#coding dummy variable
data['class_dummy'] = data['Class'].replace(['ckd', 'notckd'],
                                                        [0, 1])

# create a plot
data.plot(kind='scatter', x=data['Hemoglobin'], y=data['class_dummy'])




#fitting model using minimize
min_res_logit = minimize(cost_function_utils.logistic_regression_cost_function, [-7, 0.8], args=(data['hemoglobin'], data['class_dummy']))
min_res_logit


# fitting model using statsmodels ??
log_reg_mod = smf.logit('class_dummy ~ hemoglobin', data=data)
# Fit it.
fitted_log_reg_mod = log_reg_mod.fit()
fitted_log_reg_mod.summary()

#trying to make the tests but don't really know how
scipy_params = min_res_logit.x
statsmodels_params = fitted_log_reg_mod.params.values

are_close = np.isclose(scipy_params, statsmodels_params, atol=1e-4)
all_close = np.all(are_close)

if all_close:
    print('Both methods produce similar paramater estimates')
else:
    print('Parameter estimates differ')


#new plot showing Class ~ Hemoglobin maybe...
data['predicted_prob'] = fitted_log_reg_mod.predict(data['Hemoglobin'])
data = data.sort_values(by='Hemoglobin')
plt.scatter(data['Hemoglobin'], data['class_dummy'])
plt.plot(data['Hemoglobin'], data['predicted_prob'])