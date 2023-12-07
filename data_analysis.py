# you need to modify this script so that it performs logistic regression analysis
# on the CKD data
import cost_function_utils
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as sci_opt
import statsmodels as sms

# this just shows how to use your `cost_function_utils` module with the current
# script
cost_function_utils.junk_function()

df = pd.read_csv('ckd_full.csv')
#dropping NaN values
medical_df = df.dropna(subset=['Hemoglobin','Class'])



# generate some toy data
#data = pd.DataFrame({'A': np.random.normal(100, 10, 100),
#                    'B': np.random.normal(10, 1, 100)})

# show the data
#display(data)

# create a plot
#data.plot(kind='scatter', x=medical_df['Hemoglobin'], y=medical_df['Class'])

#plotting class against haemoglobin
plt.scatter(medical_df['Hemoglobin'],medical_df['Class'])
plt.xlabel('Haemoglobin')
plt.ylabel('Class')
plt.show()

hemoglobin = medical_df['Hemoglobin']
#converting class into boolean mask
med_class = medical_df['Class']=='ckd'

min_res_logit = sci_opt.minimize(cost_function_utils.logistic_regression_cost_function, [1, 0], args=(hemoglobin,med_class))
print(min_res_logit.x)
#could not figure out minimize using statsmodels as the syntax wasn't easy to find online

predicted_odds = hemoglobin * min_res_logit.x[1] + min_res_logit.x[0]
predicted_probs = cost_function_utils.inverse_logit(predicted_odds)


plt.scatter(hemoglobin,med_class)
plt.scatter(hemoglobin, predicted_probs, s = 10)
plt.xlabel('Haemoglobin')
plt.ylabel('Probability of CKD')
