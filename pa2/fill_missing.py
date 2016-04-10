# Machine Learning PS2 | Part 2 | Fill in missing values.
# CAPP 30254
# Sirui Feng
# siruif@uchicago.edu

'''
This file fills in the missing values with attribute's mean.
'''

from read_explore import clean_data
import pandas as pd
import numpy as np


#Create a list of attributs you want to fill in missing values, a direcotry
#for output, and a dictionary  for rounding digits after decimal.

missing_variables = ['monthly_income', 'number_of_dependents']
output_filename = 'output/cs-training_filled_missingvalues.csv'
rounding_dict = {'monthly_income': 0, 'number_of_dependents': 0}

##########################################
                                        ##
#PLEASE DO NOT MODIFY THE CODES BELOW.  ##
										##
##########################################

def fill_with_mean(df):
	'''
	Fills in missing values with column means.
	'''
	for var in missing_variables:
		print("Filling missing values for:", var)
		means = df[var].mean()
		df[var] = df[var].fillna(means)
	df = df.round(rounding_dict)
	df.to_csv(output_filename, mode = 'w', index_label = 'ID')
	print("File has been saved to:", output_filename)

def fill_in_missing_values(input_data):
	df = clean_data(input_data)
	# Fill in missing values with the mean of the values for that attribute.
	fill_with_mean(df)