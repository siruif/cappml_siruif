# Aurhor: Sirui Feng
# Machine Learning Pipeline - Exploring and preprocessing the data
# siruif@uchicago.edu

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

######################## STEP 1 READ AND EXPLORE ########################

def read_data(input_data):
	'''
	Convests a csv file into a df with modified column names. The first column
	is the index number.
	'''
	df = pd.read_csv(input_data, index_col = 0)
	return df

def calculate_summary_stats(df):
	'''
	Generate summary statistics including mean, count, std, min, median, 25%,
	50%, max and missing value counts.
	'''
	# Compute the summary statistics
	summary_stats = df.describe()
	summary_stats.append(df.mode())
	count_nan = pd.Series(len(df) - df.count(), name = "missing_values").T
	summary_stats = summary_stats.append(count_nan)

	summary_stats = summary_stats.T
	summary_stats.rename(columns = {0:'mode', '50%': 'median'}, inplace = True)
	summary_stats.to_csv('output/summary_stats.csv')
	print('outputing summary statistics...')
	# Compute the correlation
	correlation = df.corr()
	correlation.to_csv('output/correlation.csv')
	print("outputing correlaiton table...")

def plot_histogram(df,attr):

	unique_vals = df[attr].count()
	if unique_vals == 2:
		fig = df.groupby(attr).size().plot(kind='bar')
	elif unique_vals < 10:
		fig = df[attr].hist(xlabelsize=10, ylabelsize=10, bins=unique_vals)
	else:
		fig =df[attr].hist()

	fig.set_xlabel(attr) #defines the x axis label
	fig.set_ylabel('Number of Observations') #defines y axis label
	fig.set_title(attr+' Distribution') #defines graph title
	
	plt.savefig("output/charts/"+attr)
	plt.close('all')

def plot_log(df, attr):

	lb = 0
	ub = 15
	incre = 0.5
	plt.gca().set_xscale('log')
	fig = df[attr].hist(bins = np.exp(np.arange(lb, ub, incre)))
	fig.set_xlabel(attr)

	plt.savefig('output/charts/log_' + attr)
	plt.close()

####################### STEP 2 FILL MISSING VALUES #######################

def impute_missing_values(df, missing_var, method):
	'''
	Imputes missing value with selected method.
	'''
	if method == 'mean':
		mean = df[missing_var].mean()
		return mean

	elif mehtod == 'median':
		median = df[missing_var].median()
		return median

	elif method == 'mode':
		mode = df[missing_var].mode()[0]
		return mode

def replace_missing_values(df, missing_var, value):
	'''
	Fills missing values for one column with specified interger/float.
	'''
	df[missing_var] = df[missing_var].fillna(value)
	return df

def imputation(training, testing, missing_var, method):
	value = impute_missing_values(training, missing_var, method)
	replace_missing_values(testing, missing_var, value)

######################## STEP 3 GENERATE FEATURES ########################

def discretize(df, conti_var, num_bins = 4):
	'''
	Discreizes continuous variables into specified bins.
	'''
	labels = list()
	new_col_name = str(conti_var) + "_discrete"
	for i in range(1, num_bins+1):
		labels.append(i)
	discretization = pd.qcut(df[conti_var], num_bins, labels)
	df.insert(1, new_col_name,discretization)
	return df

def get_dummies(df, categ_var):
	'''
	Creates binary variable from categorical variable.
	Get k-1 dummies out of k categorical levels by removing the first level.
	'''
	dummies = pd.get_dummies(df[categ_var], categ_var, drop_first = True)
	df = df.join(dummies)
	return df

def log_attr(df,attr):
    log_attr ='log_' + attr
    df[log_attr] = np.log1p(df[attr])
    df.drop(attr, axis=1, inplace = True)
    return df

#############################################################################

if __name__=="__main__":

    input_data = '../pa2/cs-training.csv'
    output_data = 'training_cleaned.csv'

    df = read_data(input_data)
    df_original = df.copy()

    attributes = list(df.columns.values)

    calculate_summary_stats(df)

    for attr in attributes:
    	plot_histogram(df,attr)

    missing_variables = ['MonthlyIncome', 'NumberOfDependents']
    for missing_var in missing_variables:
    	value = impute_missing_values(df, missing_var, 'mean')
    	df = replace_missing_values(df,missing_var, value)

    df = log_attr(df,'MonthlyIncome')
    df.to_csv(output_data)