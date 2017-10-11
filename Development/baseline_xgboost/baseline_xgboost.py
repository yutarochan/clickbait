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
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, StratifiedKFold

sys.path.append('..')
from util.load_data import JSONData
from util.scores import ScoreReport

import warnings
warnings.filterwarnings("ignore")

''' Constants and Parameters '''
DATA_ROOT = '../../Data/'       # Root Folder of where Dataset Resides
MODEL_ROOT = '../../Models/dataset/'    # Root Folder of where Model Resides
K_FOLD = 10
SHUFFLE_FOLDS = True
np.random.seed(9892)                    # Seed Parameter for PRNG

report = ScoreReport('Baseline Model + StratifiedKFold - FIXED ROC')  # Automated Score Reporting Utility

''' Import Data '''
# Load Dataset
raw_data = csv.reader(open(DATA_ROOT+'feat.csv', 'rb'), delimiter=',')
# train_Y = data_load.load_train_Y()
