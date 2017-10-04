'''
Baseline Model
Simple baseline model based on example features from the instruction sets

Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import sys
import string
import numpy as np
from nltk import word_tokenize
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, StratifiedKFold

sys.path.append('..')
from text2num import text2num
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

report = ScoreReport('Baseline Model + StratifiedKFold - FIXED ROC')  # Automated Score Reporting Utility

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

    # Perform Feature Extraction
    word_count = len(text)                                                  # Total Word Count
    avg_word_len = sum(map(lambda x: len(x), text))/float(len(text))        # Average Word Length
    max_word_len = max(map(lambda x: len(x), text))                         # Longest Word Length
    is_number = text[0].isdigit() or is_numeric(text[0])                    # Check if starts with number (Either Numerically or Linguistically)
    start_ws = text[0] in ['who', 'what', 'why', 'where', 'when', 'how']    # Whether it starts with question word

    return [word_count, avg_word_len, max_word_len, int(is_number), int(start_ws)]

# Finalize Feature and Target Vectors
X = np.array(map(lambda x: preprocess(x['targetTitle']), train_X))
Y = np.array(map(lambda x: [0] if x['truthClass'] == 'no-clickbait' else [1], train_Y))
Y_ = np.array(map(lambda x: 0 if x['truthClass'] == 'no-clickbait' else 1, train_Y))

''' CV Model Training '''
# K-Fold and Score Tracking
kf = StratifiedKFold(n_splits=K_FOLD, shuffle=SHUFFLE_FOLDS)

print('Training Model...')
for i, (train_idx, test_idx) in enumerate(kf.split(X, Y_)):
    print('\n[K = ' + str(i+1) + ']')
    ''' SMOTE - Generate Synthetic Data '''
    # sm = SMOTE(kind='regular')
    # X_resampled = []
    # X_res, Y_res = sm.fit_sample(X[train_idx], Y[train_idx])

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

''' Full Model Training...
gnb = GaussianNB()
gnb.fit(X, Y)

def predict(sentence):
    print(sentence)
    if gnb.predict(preprocess(sentence))[0] == 0:
        print('NOT CLICKBAIT!')
    else:
        print('CLICKBAIT')
    print()

print()
predict('10 Things Yuya Screwed Up While Building A Simple Classifier Model')
predict('Five Unbelievable Things Going on in Nittany Data Labs!!!!')
predict('You won\'t believe how these 9 shocking clickbaits work!')
predict('You\'ll Never Look at Barbie Dolls The Same Once You See These Paintings')
predict('Well If It Isn\'t Coupons, I Can\'t Even IMAGINE What It Is!')
predict('Trump vows to back winner of Alabama GOP Senate runoff')
'''
