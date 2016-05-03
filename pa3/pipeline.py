# Aurhor: Sirui Feng
# Machine Learning Pipeline
# siruif@uchicago.edu

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import statsmodels.api as sm
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn import metrics


def read_data(input_data):
	'''
	Convests a csv file into a df with modified column names. The first column
	is the index number.
	'''
	df = pd.read_csv(input_data, index_col = 0)
	df.columns = [camel_to_snake(col) for col in df.columns]
	return df


def camel_to_snake(column_name):
    '''
    Source: https://github.com/yhat/DataGotham2013/blob/master/notebooks
    	/3%20-%20Importing%20Data.ipynb
    
    Converts a string that is camelCase into snake_case
    Example:
        print camel_to_snake("javaLovesCamelCase")
        > java_loves_camel_case
    See Also:
        http://stackoverflow.com/questions/1175208/elegant-python-functi
        on-to-convert-camelcase-to-camel-case
    '''
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()






if __name__=="__main__":
    if len(sys.argv) != 2:
        print("usage: python3 {} <raw data filename>".format(sys.argv[0]))
        sys.exit(1)

    input_data = sys.argv[1]

    df = read_data(input_data)

    variables = list(df.columns.values)

    for var in variables:
