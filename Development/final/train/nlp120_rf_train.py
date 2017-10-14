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
DATA_ROOT2 = '../../../Data/full_feat/'    # Root Folder of where Dataset Resides
MODEL_ROOT = '../../../Models/'            # Root Folder of where Model Resides
np.random.seed(9892)                       # Seed Parameter for PRNG

''' Import Data '''
# Load Dataset
print('Loading Dataset')
data_load = JSONData(DATA_ROOT+'instances_train.jsonl', DATA_ROOT+'truth_train.jsonl', DATA_ROOT+'instances_test.jsonl')
raw_data = pd.read_csv('../../../Data/train_feat/feat.csv', sep=',', header=None).astype(float)
raw_data = raw_data.replace([np.inf, -np.inf, np.nan], 0)
X = raw_data.drop(raw_data.columns[[0]],axis=1).values

train_Y = data_load.load_train_Y()
Y = np.array(map(lambda x: 0 if x['truthClass'] == 'no-clickbait' else 1, map(lambda x: train_Y[int(x[0])], X)))

''' Train Full Model '''
print('Training Full Model')
randforest = RandomForestClassifier(criterion='entropy', verbose=True)
randforest.fit(X, Y)

''' Test Model - Generate Predictions '''
# Format and Impute Errors from Preprocessing Phase
data = pd.read_csv('../../../Data/train_feat/test_feat.csv', sep=',', header=None)
print(data.shape)
idx_list = [i[0] for i in data.iloc[:, [0]].values.tolist()]
data = data.replace([np.inf, -np.inf, np.nan], 0)
print(data.shape)
X_test = data.drop(data.columns[[0]],axis=1).values.tolist()

pred = randforest.predict(X_test)

''' Output Predictions to File '''
output = open('predictions.csv', 'wb')
for i in zip(idx_list, pred):
    output.write(str(i[0])+','+str(i[1])+'\n')
output.close()
