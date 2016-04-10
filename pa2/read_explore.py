# Machine Learning PS2 | Part 1 | Read and Explore
# CAPP 30254
# Sirui Feng
# siruif@uchicago.edu

'''
This file reads in the csv dataset and generates summarty statistics description
and histogram.

*** Note: Please have a folder called output in your current directory to pick up all
files that will be generated.
'''

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

'''
['serious_dlqin2yrs', 'revolving_utilization_of_unsecured_lines', 'age', 
'number_of_time30-59_days_past_due_not_worse', 'debt_ratio', 
'monthly_income', 'number_of_open_credit_lines_and_loans', 
'number_of_times90_days_late', 'number_real_estate_loans_or_lines', 
'number_of_time60-89_days_past_due_not_worse', 'number_of_dependents']


histogram_variables = ['age', 'number_of_dependents', 
	'number_of_open_credit_lines_and_loans', 'monthly_income']

bar_variables = ['serious_dlqin2yrs']

['revolving_utilization_of_unsecured_lines', 
'number_of_time30-59_days_past_due_not_worse', 'debt_ratio', 
'number_of_open_credit_lines_and_loans', 
'number_of_times90_days_late', 'number_real_estate_loans_or_lines', 
'number_of_time60-89_days_past_due_not_worse', ]

'''

def explore_data(input_data, output_filename): 
	'''
	Outputs summary statistics into a txt file in the output folder.
	Ouptput histograms graphs into png file in the output folder.
	'''
	# Clean the data
	df = clean_data(input_data)
	row,col = df.shape
	print("This dataset has", row, "entries and", col, "attributes.")
	variables = list(df.columns.values)

	#print(variables)

	# Get summary statistics
	print("Generating summary statistics...")
	output = calculate_summary_stats(df, variables)

	# Write to the output file
	with open(output_filename, "w") as f:
		for summary_stats in output:
			print(summary_stats, file = f)
	
	#Plot
	print("Generating graphs...")

	# Plot histograms, create a list of attributes' names that you
	#want histograms for.
	histogram_variables = ['age', 'number_of_dependents', \
	'number_of_open_credit_lines_and_loans', 'monthly_income']
	for hist_var in histogram_variables:
		plot_histogram(df, hist_var)

	#Plot bar charts, create a list of attributes' names that you
	#want bar charts for.
	bar_variables = ['serious_dlqin2yrs']
	for bar_var in bar_variables:
		plot_bar(df, bar_var)
	
def clean_data(input_data):
	'''
	Convests a csv file into a df with modified column names. The first column is
	the index number.
	'''
	df = pd.read_csv(input_data, index_col = 0)
	#df = pd.read_csv(input_data, index_col='ID', na_values = ['']) 
	#you may define missing values as emplty cells
	df.columns = [camel_to_snake(col) for col in df.columns]
	#df.columns.tolist()
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

def calculate_summary_stats(df, variables):
	'''
	For all numerical attributs, generates summary statistics including mean,
	standard deviation, median. mode and mising valu counts.
	For all attribues (not numerical), generates summary statistics including
	mode and missing value counts.
	'''
	# Compute the summary statistics
	output = list()

	for var in variables:
		output.append("Field Name: " + var)

		if df[var].dtype == 'float64' or df[var].dtype == 'int64':
			output.append("Mean: " + str(round(df[var].mean(),2)))
			output.append("Standard Deviation: " + str(round(df[var].std(),2)))
			output.append("Median: " + str(df[var].median()))
			output.append("Min: " + str(df[var].min()))
			output.append("Max: " + str(df[var].max()))

		output.append("Mode: " + str(df[var].mode()))
		output.append("Non-missing Value Count: " + str(df[var].count()))
		output.append("Missing Value Count: " + str(df[var].isnull().sum()))

		output.append('\n')

	return output

def plot_histogram(df, hist_var):
	'''
	Generate histograms for a specific column of a dataframe.
	'''
	fig = df[hist_var].hist(color = 'pink', bins = 15)
	fig.set_title('Histogram for ' + hist_var)
	plt.draw()
	plt.savefig('output/' + hist_var)
	plt.close()

def plot_bar(df, bar_var):
	'''
	Generate bar charts for a specific column of a dataframe.
	'''
	fig =df.groupby(bar_var).size().plot(kind='bar', color='pink')
	fig.set_xlabel(bar_var) #defines the x axis label
	fig.set_ylabel('Number of Observations') #defines y axis label
	fig.set_title(bar_var+' Distribution') #defines graph title
	plt.draw()
	plt.savefig("output/"+bar_var)
	plt.close('all')

def go(input_data):
	explore_data(input_data,'output/summary_stats.txt')

