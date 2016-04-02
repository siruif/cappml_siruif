# Machine Learning PS1 | Problem A | Part 1
# CAPP 30254
# Sirui Feng
# siruif@uchicago.edu

'''
This file generates summary statistics and plots histograms.
'''


import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# variables = ['First_name', 'Last_name', 'State', 'Gender', 'Age', 'GPA',\
#  'Days_missed', 'Graduated']

# PROBLEM A | PART 1

def clean_data(input_data):

	df = pd.read_csv(input_data, index_col='ID', na_values = [''])

	return df

def get_summary(df, variables):
	# Compute the summary statistics
	output = list()

	for var in variables:
		output.append("Field Name: " + var)

		if df[var].dtype == 'float64':
			output.append("Mean: " + str(round(df[var].mean(),2)))
			output.append("Standard Deviation: " + str(round(df[var].std(),2)))
			output.append("Median: " + str(df[var].median()))

		output.append("Mode: " + str(df[var].mode()))
		output.append("Missing Value Count: " + str(df[var].isnull().sum()))

		output.append('\n')

	return output

def plot_histogram(df, var):

	fig = df[var].hist(color = 'pink')
	fig.set_title(var + ' Distribution')
	plt.draw()
	plt.savefig('output/' + var)
	plt.close()

def gen_summary(input_data, output_filename):

	# Clean the data
	df = clean_data(input_data)
	variables = list(df.columns.values)

	# Get summary statistics
	output = get_summary(df, variables)

	# Write to the output file
	with open(output_filename, "w") as f:
		for summary_stats in output:
			print(summary_stats, file = f)

	# Plot histograms
	for var in variables:
		if df[var].dtype == 'float64':
			plot_histogram(df, var)