'''
Bag-of-Word + Logistic Model
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''

### TODO: Place score computation module!!! and perform complete analysis!

from __future__ import print_function
import sys
import string
import numpy as np
from nltk import word_tokenize
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, f1_score, classification_report

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

# Model Hyper-Parameters
# TODO: Perform Hyperparameter Selection for Best Model - Get Data for Each Dimensional Range
MAX_FEATURES = 10                       # Dimension of Feature Vector

report = ScoreReport('Bag-of-Words '+str(MAX_FEATURES)+' DIM + Naive Bayes')  # Automated Score Reporting Utility

''' Import Data '''
# Load Dataset
data_load = JSONData(DATA_ROOT+'instances_train.jsonl', DATA_ROOT+'truth_train.jsonl', DATA_ROOT+'instances_test.jsonl')
train_X = data_load.load_train_X()
train_Y = data_load.load_train_Y()

''' Preprocess Data '''
# Build Feature Vector
def is_numeric(text):
    try: return type(text2num(text)) == type(0)
    except Exception as e: return False

def preprocess(text):
    text = word_tokenize(text.lower())                                      # Tokenize & Normalize Text
    text = filter(lambda x: x not in string.punctuation, text)              # Remove Punctuation
    return ' '.join(text)

# Perform Feature Preprocessing
X = np.array(map(lambda x: preprocess(x['targetTitle']), train_X))
Y = np.array(map(lambda x: [0] if x['truthClass'] == 'no-clickbait' else [1], train_Y))

# Perform Feature Vectorizer
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = MAX_FEATURES)
X = vectorizer.fit_transform(X).toarray()

''' CV Model Training '''
# K-Fold and Score Tracking
kf = KFold(n_splits=K_FOLD, shuffle=SHUFFLE_FOLDS)

for i, (train_idx, test_idx) in enumerate(kf.split(X)):
    print('\n[K = ' + str(i+1) + ']')
    # Train Model
    gnb = GaussianNB()
    gnb.fit(X[train_idx], Y[train_idx])

    # Generate Predictions & Confidence Estimates
    y_pred = gnb.predict(X[test_idx])
    y_prob = gnb.predict_proba(X[test_idx])

    # Append to Report
    y_prob = map(lambda x: x[1][x[0]], zip(y_pred, y_prob))
    report.append_result(Y[test_idx].reshape(y_pred.shape), y_pred, y_prob)

# Generate Prediction Reports
report.generate_report()
