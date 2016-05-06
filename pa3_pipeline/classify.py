# Author: Sirui Feng
# Machine Learning Pipeline - Classification
# siruif@uchicago.edu

'''
The code is a modified version of Rayid Ghani's magicloops.
Source code:
https://github.com/rayidghani/magicloops/blob/master/magicloops.py
'''

from __future__ import division
import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm, metrics, tree, \
decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, \
GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, \
OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
f1_score, roc_auc_score, precision_recall_curve
import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
import time


def define_clfs_params():


    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, \
            max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SVM': svm.LinearSVC(random_state=0, dual = False),
        'KNN': KNeighborsClassifier(n_neighbors=5) 
            }

    grid = { 
    'RF':{'n_estimators': [1,10,100], 'max_depth': [1,5,10,20,50], 'max_features': ['sqrt','log2'], 'min_samples_split': [2,5,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5], 'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1],'penalty':['l1','l2']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'], 'algorithm': ['auto','ball_tree','kd_tree']}
           }

    return clfs, grid

def clf_loop(models_to_run,clfs,grid,X,y):
    clf_log = dict()

    for n in range(1, 2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, \
            test_size=0.2, random_state=0)
        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            start_time = time.time()
            print(models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    print(clf)
                    y_pred_probs = \
                    clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                    #threshold = np.sort(y_pred_probs)[::-1][int(.05*len(y_pred_probs))]
                    #print (threshold)
                    end_time=time.time()
                    print(models_to_run[index], "used:",end_time-start_time)
                    print (precision_at_k(y_test,y_pred_probs,.05))
                    #plot_precision_recall_n(y_test,y_pred_probs,clf)
                    clf_log[clf] = dict()
                    clf_log[clf]['evaluation'] = evaluate(y_test, y_pred_probs)
                    clf_log[clf]['time'] = round(end_time - start_time,2)
                    print()
                except IndexError as e:
                    print ('Error:',e)
                    continue
    print("~"*101)
    print(clf_log)
    return clf_log

def evaluate(y_true, y_predict):
    evaluation = dict()

    try:
        evaluation['accuracy'] = accuracy_score(y_true, y_predict)
        evaluation['precision'] = precision_score(y_true, y_predict)
        evaluation['recall'] = recall_score(y_true, y_predict)
        evaluation['f1'] = f1_score(y_true, y_predict)
        evaluation['area_under_curve'] = roc_auc_score(y_true, y_predict),
        evaluation['precision_at_k'] = precision_at_k(y_true,y_predict, 0.05)

    except:
        print("No metrics.")

    return evaluation

def plot_precision_recall_n(y_true, y_prob, model_name):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, \
    pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()

def precision_at_k(y_true, y_scores, k):
    threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))]
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    return metrics.precision_score(y_true, y_pred)

def get_y_x(df):
    y = df['SeriousDlqin2yrs']
    df.drop('SeriousDlqin2yrs', axis = 1, inplace = True)
    return y, df

def output_clf_log(clf_log):
    with open('output/results.csv','w') as outfile:
        writer = csv.DictWriter(outfile, fieldnames = ['clf', \
                'time', 'accuracy', 'precision', 'recall', 'f1', \
                'area_under_curve', 'precision_at_k'])
        writer.writeheader()

        for k in clf_log:
            try:
                writer.writerow({'clf': k, 
                    'time': clf_log[k]['time'], 
                    'accuracy': clf_log[k]['evaluation']['accuracy'], 
                    'precision': clf_log[k]['evaluation']['precision'], 
                    'recall': clf_log[k]['evaluation']['recall'], 
                    'f1': clf_log[k]['evaluation']['f1'], 
                    'area_under_curve': clf_log[k]['evaluation']['area_under_curve'], 
                    'precision_at_k': clf_log[k]['evaluation']['precision_at_k']})
            except:
                print("Error")


def main(filename): 
    clfs,grid = define_clfs_params()
    models_to_run=['KNN','RF','LR','GB','NB','DT']

    #get X and y
    df = pd.read_csv(filename, index_col = 0)
    y, X = get_y_x(df)

    clf_log = clf_loop(models_to_run,clfs,grid,X,y)

if __name__ == '__main__':
    main('training_cleaned.csv')