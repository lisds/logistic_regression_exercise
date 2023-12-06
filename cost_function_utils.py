# this module contains functions related to the manual implementation of the 
# logistic regression cost function
import numpy as np

# this is just to demonstrate how functions from this file can be used in 
# "showcase_notebook.Rmd" - you can delete this function 

def inverse_logit(y):
    # your code here
    odds = np.exp(y)
    return odds / (odds + 1)

def logistic_regression_cost_function(intercept_and_slope, x, y):
    # your code here
    intercept, slope = intercept_and_slope

    predicted_log_odds = intercept + slope * x

    predicted_prob_of_1 = inverse_logit(predicted_log_odds)

    predicted_prob_of_actual_scores = y * predicted_prob_of_1 + (1 -y) * (1 - predicted_prob_of_1)

    log_likelihood = np.sum(np.log(predicted_prob_of_actual_scores))
    
    return - log_likelihood
