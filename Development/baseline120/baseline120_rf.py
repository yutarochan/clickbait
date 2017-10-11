'''
Baseline Model
Simple baseline model based on example features from the instruction sets

Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import sys
import csv
import string
import numpy as np
from nltk import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold

sys.path.append('..')
from util.load_data import JSONData
from util.scores import ScoreReport

import warnings
warnings.filterwarnings("ignore")

''' Constants and Parameters '''
DATA_ROOT = '../../Data/dataset/'       # Root Folder of where Dataset Resides
MODEL_ROOT = '../../Models/dataset/'    # Root Folder of where Model Resides
K_FOLD = 10
SHUFFLE_FOLDS = True
np.random.seed(9892)                    # Seed Parameter for PRNG

report = ScoreReport('Baseline120 Random Forest Classifier')  # Automated Score Reporting Utility

''' Import Data '''
# Load Dataset
data_load = JSONData(DATA_ROOT+'instances_train.jsonl', DATA_ROOT+'truth_train.jsonl', DATA_ROOT+'instances_test.jsonl')
raw_data = csv.reader(open('../../Data/feat.csv', 'rb'), delimiter=',')
X = map(lambda x: map(lambda y: float(y), x), raw_data)
train_Y = data_load.load_train_Y()

# Filter Out Invalid Results
X = filter(lambda x: len(x) == 121, X)

# Filter/Process Adjusting Results
X = np.array(map(lambda x: x[1:], X), dtype='float')
train_Y = map(lambda x: train_Y[int(x[0])], X)
Y = np.array(map(lambda x: 0 if x['truthClass'] == 'no-clickbait' else 1, train_Y))
Y_ = np.array(map(lambda x: 0 if x['truthClass'] == 'no-clickbait' else 1, train_Y))

''' CV Model Training '''
# K-Fold and Score Tracking
kf = StratifiedKFold(n_splits=K_FOLD, shuffle=SHUFFLE_FOLDS)

print('Training Model...')
for i, (train_idx, test_idx) in enumerate(kf.split(X, Y_)):
    print('\n[K = ' + str(i+1) + ']')

    # Train Model
    randforest = RandomForestClassifier(criterion='entropy')
    randforest.fit(X[train_idx], Y[train_idx])

    # Generate Predictions & Confidence Estimates
    y_pred = randforest.predict(X[test_idx])
    y_prob = randforest.predict_proba(X[test_idx])

    # Append to Report
    y_prob = map(lambda x: x[1][x[0]], zip(y_pred, y_prob))
    report.append_result(Y[test_idx].reshape(y_pred.shape), y_pred, y_prob)

# Generate Prediction Reports
report.generate_report()
