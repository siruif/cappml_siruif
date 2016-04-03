# Machine Learning PS1 | Problem A | Part 3
# CAPP 30254
# Sirui Feng
# siruif@uchicago.edu


from ps1_1 import clean_data
import pandas as pd
import numpy as np

'''
This file fills in the missing values of Age, GPA, and Days_missed attributes
with different measures.
'''

def fill_with_mean(df, input_data, output_filename, variables):

	for var in variables:
		means = df[var].mean()
		df[var] = df[var].fillna(means)
	df = df.round({'Age':0, 'GPA':0, 'Days_missed':0})
	df.to_csv(output_filename, mode = 'w', index_label = 'ID')

def fill_with_conditional_mean(df, input_data, output_filename, conditions, \
	variables):

	for var in variables:
		means = df.groupby(conditions)[var].mean()
		df = df.set_index(conditions)
		df[var] = df[var].fillna(means)
		df = df.reset_index()
	df = df.round({'Age':0, 'GPA':0, 'Days_missed':0})
	df.to_csv(output_filename, mode = 'w', index_label = 'ID')

def fill_in_missing_values(input_data):

	df = clean_data(input_data)
	variables = ["Age", "GPA", "Days_missed"]

	# Question 3 Part A
	# Fill in missing values with the mean of the values for that attribute.
	fill_with_mean(df, input_data,'output/mock_student_data_mean.csv', \
		variables)

	# Question 3 Part B
	# Fill in missing values with a class-conditional mean(where the class
		# is whether graduated or not).
	fill_with_conditional_mean(df, input_data, \
		'output/mock_student_data_conditional_mean.csv', ['Graduated'], \
		variables)

	# Question 3 Part B
	# A better approach might be filling in values with classes-conditional
		#means (where the classes are whether graduated or not and gender).
	fill_with_conditional_mean(df, input_data, \
		'output/mock_student_data_multi_con.csv', \
		['Graduated', 'Gender'], variables)