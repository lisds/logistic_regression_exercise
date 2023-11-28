# Logistic regression exercise task

**Do not fork this repository!**. It is for illustrative purposes only.

This is an example repo for the logistic regression exercise task.

It is currently just a skeleton, to show you the sort of file structure your 
exercise repo should have.

Once the scripts were written, the full analysis steps would be:

- run the data cleaning script (this will save a cleaned .csv file)
- run the data analysis script (this will import the cost function from
  cost_function_utils.py and will generate the graphs/perform the analysis)

You could also set up the data analysis script to call the data cleaning script
directly (see here: https://stackoverflow.com/questions/7974849/how-can-i-make-one-python-file-run-another).

The `showcase_notebook.ipynb` is just to run the data analysis script, for
presentation etc. (Normally we would prefer .Rmd files, but this notebook
is only a few lines long, and is only being used to run one other script, so it
is OK in this case).
