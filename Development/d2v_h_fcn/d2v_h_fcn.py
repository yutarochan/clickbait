'''
Word Embedding + FC Neural Network Architecture
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import sys
import random
import string
import numpy as np
from nltk import word_tokenize
from operator import itemgetter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold

import numpy
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D

sys.path.append('..')
from util.load_data import JSONData
from util.scores import ScoreReport

import warnings
warnings.filterwarnings("ignore")

''' Constants and Parameters '''
DATA_ROOT = '../../Data/dataset/'           # Root Folder of where Dataset Resides
MODEL_ROOT = '../../Models/d2v_h_fcn/'      # Root Folder of where Model Resides

# Model Hyperparameters
K_FOLD = 10
SHUFFLE_FOLDS = True
MAX_WORDS = 50
EPOCHS = 100
BATCH_SIZE = 256
np.random.seed(9892)                        # Seed Parameter for PRNG

report = ScoreReport('Fully Connected Neural Network - MAX_DIM ' + str(MAX_WORDS) + ' ' + str(EPOCHS) + ' + EPOCHS ')  # Automated Score Reporting Utility

''' Import Data '''
# Load Dataset
data_load = JSONData(DATA_ROOT+'instances_train.jsonl', DATA_ROOT+'truth_train.jsonl', DATA_ROOT+'instances_test.jsonl')
train_X = data_load.load_train_X()
train_Y = data_load.load_train_Y()

''' Preprocess Data '''
# Build Feature Vector
def preprocess(text):
    text = word_tokenize(text.lower())                                      # Tokenize & Normalize Text
    text = filter(lambda x: x not in string.punctuation, text)              # Remove Punctuation
    return text

# Finalize Feature and Target Vectors
X = np.array(map(lambda x: preprocess(x['targetTitle']), train_X))
Y = np.array(map(lambda x: [0] if x['truthClass'] == 'no-clickbait' else [1], train_Y))

''' CV Model Training '''
# K-Fold and Score Tracking
kf = KFold(n_splits=K_FOLD, shuffle=SHUFFLE_FOLDS)

print('Training Model...')
for i, (train_idx, test_idx) in enumerate(kf.split(X)):
    print('\n[K = ' + str(i+1) + ']')

    # Pad Sequence for Embeddings
    X_train = sequence.pad_sequences(X[train_idx], maxlen=MAX_WORDS)

    # Build Model
    model = Sequential()
    model.add(Embedding(len(X_train), 32, input_length=MAX_WORDS))
    model.add(Flatten())
    model.add(Dense(150, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Train Model
    model.fit(X_train, Y[train_idx], validation_data=(X[test_idx], Y[test_idx]), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

    # Generate Predictions
    y_pred = logistic.predict(_X_test)
    y_prob = logistic.predict_proba(_X_test)

    # Append to Report
    y_prob = map(lambda x: x[1][x[0]], zip(y_pred, y_prob))
    report.append_result(Y[test_idx].reshape(y_pred.shape), y_pred, y_prob)

# Generate Prediction Reports
report.generate_report()
