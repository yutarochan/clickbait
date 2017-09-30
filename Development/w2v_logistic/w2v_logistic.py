'''
Word2Vec + Logistic Function
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import sys
import gensim
import string
import numpy as np
from nltk import word_tokenize
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

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

# Model Hyperparameters
MODEL_DIR = '../../Models/w2v_cb_corpus/w2v_cb_headline_15_15_DIM-100.bin.gz'
COMB = 'MIN'                            # Word Embedding Combination Method

report = ScoreReport('Baseline Model')  # Automated Score Reporting Utility

''' Import Data '''
# Load Dataset
data_load = JSONData(DATA_ROOT+'instances_train.jsonl', DATA_ROOT+'truth_train.jsonl', DATA_ROOT+'instances_test.jsonl')
train_X = data_load.load_train_X()
train_Y = data_load.load_train_Y()

# Load Word Embedding Model
model = gensim.models.Word2Vec.load(MODEL_DIR)

''' Preprocess Data '''
# Build Feature Vector
def preprocess(text):
    text = word_tokenize(text.lower())                                      # Tokenize & Normalize Text
    text = filter(lambda x: x not in string.punctuation, text)              # Remove Punctuation
    return text

# Finalize Feature and Target Vectors
X = np.array(map(lambda x: preprocess(x['targetTitle']), train_X))
Y = np.array(map(lambda x: [0] if x['truthClass'] == 'no-clickbait' else [1], train_Y)
