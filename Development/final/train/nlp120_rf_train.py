'''
NLP 120 Features + Random Forest Model
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import sys
import csv
import string
import pickle
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from nltk import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold

sys.path.append('../..')
from util.load_data import JSONData

import warnings
warnings.filterwarnings("ignore")

''' Constants and Parameters '''
DATA_ROOT = '../../../Data/dataset/'       # Root Folder of where Dataset Resides
MODEL_ROOT = '../../../Models/'            # Root Folder of where Model Resides
np.random.seed(9892)                       # Seed Parameter for PRNG

''' Import Data '''
# Load Dataset
data_load = JSONData(DATA_ROOT+'instances_train.jsonl', DATA_ROOT+'truth_train.jsonl', DATA_ROOT+'instances_test.jsonl')
raw_data = csv.reader(open('../../../Data/final_feat/feat.csv', 'rb'), delimiter=',')
X = map(lambda x: map(lambda y: float(y), x), raw_data)
train_Y = data_load.load_train_Y()

# Filter Out Invalid Results
print('PREFILTER: ' + str(len(X)))
X = filter(lambda x: len(x) == 121, X)
print('POSTFILTER: ' + str(len(X)))

# Filter/Process Adjusting Results
print('Loading Dataset')
X = np.array(map(lambda x: x[1:], X), dtype='float')
Y = np.array(map(lambda x: 0 if x['truthClass'] == 'no-clickbait' else 1, map(lambda x: train_Y[int(x[0])], X)))

''' Train Full Model '''
print('Training Full Model')
# randforest = RandomForestClassifier(criterion='entropy')
# randforest.fit(X, Y)

gnb = GaussianNB()
gnb.fit(X, Y)

''' Test Model - Generate Predictions '''
# Format and Impute Errors from Preprocessing Phase
data = pd.read_csv('../../../Data/validation/valid_X.csv', sep=',',header=None)
data = data.replace([np.inf, -np.inf, np.nan], 0) # Replace missing or infinite values with zero.
X_test = data.values

ALT_DATA_ROOT = '/tmp/clickbait/Data/clickbait17-train-170331/'
data_load = JSONData(ALT_DATA_ROOT+'instances.jsonl', ALT_DATA_ROOT+'truth.jsonl', ALT_DATA_ROOT+'instances.jsonl')
# train_X = data_load.load_train_X()
Y_test = data_load.load_train_Y()

Y_test = np.array(map(lambda x: [0] if x['truthClass'] == 'no-clickbait' else [1], Y_test))

Y_pred = map(lambda x: gnb.predict([x[1:]])[0], X_test)
print(Y_pred[0])

from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred, labels=[0, 1]))
