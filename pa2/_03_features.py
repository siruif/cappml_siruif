# Machine Learning PS2 | Part 3 | Generate Features
# CAPP 30254
# Sirui Feng
# siruif@uchicago.edu

'''
This file discreizes continuous variables and 
creates binary variables from categorical variables.

*** Note: Please have a folder called output in your current directory to pick up all
files that will be generated.
'''

from _01_read_explore import clean_data
import pandas as pd
import numpy as np

# Create a dictionary of continous variables that you want to discretize as keys,
# and values of bins as values. Some meaning bins can be 2(median), 
# 4(quartiles), or
continuous_variables_dictionary = {'age':2}
#Create a list of categorical variables that you want to make dummies of.
categorical_variables = ['age_discrete']

output_filename = 'output/cs-training_features.csv'

##########################################
                                        ##
#PLEASE DO NOT MODIFY THE CODES BELOW.  ##
										##
##########################################

def discretize(df, conti_var, how = 4):
	'''
	Discreizes continuous variables.
	'''
	if len(continuous_variables_dictionary) == 0:
		return df
	labels = list()
	new_col_name = str(conti_var) + "_discrete"
	for i in range(1, how+1):
		labels.append(i)
	discretization = pd.qcut(df[conti_var], how, labels)
	df.insert(1, new_col_name,discretization)
	df.drop(conti_var, axis = 1, inplace = True)
	return df

def get_dummies(df):
	'''
	Creates binary variables from categorical variables.
	Get k-1 dummies out of n categorical levels by removing the first level.
	'''
	if len(categorical_variables) == 0:
		return df
	for categ_var in categorical_variables: 
		dummies = pd.get_dummies(df[categ_var], categ_var, drop_first = True)
		df = df.join(dummies)
		df.drop(categ_var, axis = 1, inplace = True)
	return df

def generate_features(df):
	print("Discretizing continuous variables...")
	for conti_var in continuous_variables_dictionary:
		how = int(continuous_variables_dictionary[conti_var])
		df = discretize(df, conti_var, how)

	print("Making dummies for categorical variables...")
	df = get_dummies(df)

	df.to_csv(output_filename, mode = 'w', index_label = 'ID')
	print("File has been saved to: 'output/cs-training_features.csv'")

	return df