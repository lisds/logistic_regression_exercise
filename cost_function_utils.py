# this module contains functions related to the manual implementation of the 
# logistic regression cost function
import numpy as np

# this is just to demonstrate how functions from this file can be used in 
# "showcase_notebook.Rmd" - you can delete this function 

def inverse_logit(v):
    """Reverse Logit transformation on the array 'v'
    """
    odds = np.exp(v)
    return odds / (odds +1)

# probaly should run tests here? 


def logistic_regression_cost_function(int_and_slope, x, y):
    """ Cost function to finf the max log liklihood 

    Returns the minus of the Log Liklihood
    """
    # Difines the slop and intercept 
    int, slope = int_and_slope

    # Makes predictions on the Logodd scale (the straight one)
    predic_log_odds = int + slope * x

    # Converts the predictions from the log odd scale (straight) and then converts these to the sigmoid (S shaped curve)
    predicted_prob_of_1 = inverse_logit(predic_log_odds)

    # Predicts the probability of getting the score the value actually got (either 0 (poor appetite) or 1 (good appetite))
    predicted_probability_actual_score = y * predicted_prob_of_1 + (1 - y) * (1 - predicted_prob_of_1)

    #  Calulates the liklihhod of that (slop/intercept combo), higher the liklihood the better the fit
    #  Log is taken to mitigate issues of having a too small a no. 
    log_likelihood = np.sum(np.log(predicted_probability_actual_score))
    
    #  - becaue we want to maximise the liklihood turning it negative for minimize func.
    return -log_likelihood