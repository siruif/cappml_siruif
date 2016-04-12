# Machine Learning PS2 | Part 4 | Build Classifier
# CAPP 30254
# Sirui Feng
# siruif@uchicago.edu

from _01_read_explore import clean_data
from _02_fill_missing import fill_in_missing_values
from _03_features import generate_features
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn import metrics

#Enter your dependent variable/the variable you want to create label for.
dependent_variable = 'serious_dlqin2yrs'
#Enter a test size
test_size = 0.2
#Enter a direcotry for model evaluation report.
model_evaluation_report_out_filename = 'output/model_evaluation_report.txt'
#Enter a file name that you want to predict, leave it as an empty string if
#you do not want to make predictions.
testing_filename = 'cs-test.csv'


##########################################
                                        ##
#PLEASE DO NOT MODIFY THE CODES BELOW.  ##
										##
##########################################

def split_data(df, test_size, dependent_variable, independent_vars):
	'''
	Splits the dataset into training set and a testing set according to the tsize as
	testing size as a proportion of the entire dataset.
	'''
	X_train, X_test, y_train, y_test = \
	train_test_split(df[independent_vars], \
		df[dependent_variable], test_size = test_size, random_state = 0)

	return X_train, X_test, y_train, y_test 

def logistic_regression(df, predict = False):

	independent_vars = df.columns.difference([dependent_variable])

	print("Splitting dataset into training set and testing set...")
	X_train, X_test, y_train, y_test  = \
	split_data(df, 0.2, dependent_variable, independent_vars)
	# x = train[features_cols]
	# y = train[dependent_variable]

	print("Building logit model...")
	model = LogisticRegression()
	model.fit(X_train, y_train)
	predicted_y_test = model.predict(X_test)

	print("Evaluating model and writing model performance to:", \
		model_evaluation_report_out_filename + '...')
	with open (model_evaluation_report_out_filename, "w") as f:
		print("MODEL: LOGISTIC REGRESSION", file = f)
		print ("Metrics accuracy is:", \
			metrics.accuracy_score(y_test, predicted_y_test), file= f)
		print ("Metrics precision is:", \
			metrics.precision_score(y_test, predicted_y_test), file = f)
		print ("Metrics f1 score is (1 is the best):", \
			metrics.f1_score(y_test, predicted_y_test), file = f)
		print('\n', file = f)
		print('\n', file = f)
		print("-"*77, file = f)

	print("Done with logistic regression and evaluation.")

	if predict:
		print()
		print()
		print()
		print("Making Predictions...")
		df_testing = clean_data(testing_filename)
		df_testing = fill_in_missing_values(df_testing)
		df_testing_clean = generate_features(df_testing)
		X = df_testing_clean[independent_vars]
		predicted_y = model.predict(X)

		# I output the data in original format instead of the cleaned version.
		df_testing.insert(len(df_testing.columns), 'Predicted_'+dependent_variable, predicted_y)
		df_testing.to_csv('output/logit_results.csv', mode = 'w', index_label = 'ID')
		print("Predictions have been saved to: output/logit_results.csv")

def build_classifier(df_clean):
	if len(testing_filename) == 0:
		logistic_regression(df_clean)
	else:
		logistic_regression(df_clean, predict = True)