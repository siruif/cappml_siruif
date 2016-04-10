# Machine Learning PS2 | Part 4 | Build Classifier
# CAPP 30254
# Sirui Feng
# siruif@uchicago.edu

from read_explore import clean_data
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

dependent_variable = 'serious_dlqin2yrs'
test_size = 0.2
training_out_filename = 'training.csv'
testing_out_filename = 'testing.csv'

def split_data(df, training_out_filename, testing_out_filename, test_size = 0.2):
	'''
	Splits the dataset into training set and a testing set according to the tsize as
	testing size as a proportion of the entire dataset.
	'''
	train, test = train_test_split(df, test_size = test_size)
	train.to_csv(training_out_filename)
	test.to_csv(testing_out_filename)
	return train, test

def logistic_regression(df):
	print("Splitting dataset into training set and testing set...")
	train, test = split_data(df, training_out_filename, testing_out_filename, 0.2)

	features_cols = df.columns.difference([dependent_variable])
	x = train[features_cols]
	y = train[dependent_variable]
	print("Building logit model...")
	logit = sm.Logit(y, x)
	result = logit.fit()
	print(result.summary())

def build_classifier(df_clean):
	logistic_regression(df_clean)