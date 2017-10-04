'''
Word Embedding + FC Neural Network Architecture
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import sys
import random
import string
import gensim
import numpy as np
from nltk import word_tokenize
from operator import itemgetter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D

sys.path.append('..')
import text
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

## Word Embedding Hyperparameter
MIN_COUNT = 2     # Minimum Window Size Count
EMB_EPOCH = 1
EMB_PER_EPOCH = 1

## FC Neural Network Hyperaparameter
EPOCHS = 100
BATCH_SIZE = 256
np.random.seed(9892)                        # Seed Parameter for PRNG

report = ScoreReport('Fully Connected Neural Network - MAX_DIM ' + str(MAX_WORDS) + ' ' + str(EPOCHS) + ' + EPOCHS')  # Automated Score Reporting Utility

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
    return ' '.join(text)

# Finalize Feature and Target Vectors
X = map(lambda x: preprocess(x['targetTitle']), train_X)
Y = np.array(map(lambda x: [0] if x['truthClass'] == 'no-clickbait' else [1], train_Y))

tk = text.Tokenizer(lower=True, split=" ")
tk.fit_on_texts(X)
X = tk.texts_to_sequences(X)

''' CV Model Training '''
# K-Fold and Score Tracking
kf = KFold(n_splits=K_FOLD, shuffle=SHUFFLE_FOLDS)

print('Training Model...')
for i, (train_idx, test_idx) in enumerate(kf.split(X)):
    print('\n[K = ' + str(i+1) + ']')

    ''' Generate Wrapper for Training Set '''
    X_LD = map(lambda x: LabeledDocs(x[0], x[1], Y[x[1]][0]), zip(X[train_idx], range(len(X[train_idx]))))
    X_data = DocList(docs=X_LD)

    ''' Build Word Embedding Feature Vector '''
    print('Training Word Embedding Model')
    headline_model = gensim.models.doc2vec.Doc2Vec(size=DIM, min_count=MIN_COUNT, iter=PER_EPOCH, workers=WORKERS)
    headline_model.build_vocab(X_data.toDocEmbArray())
    tmp = X_data.toDocEmbArray()
    for _ in range(EPOCHS):
        random.shuffle(tmp)
        headline_model.train(tmp, total_examples=headline_model.corpus_count, epochs=headline_model.iter)

    # Convert Text to Word Embeddings
    _X_train = map(lambda x: headline_model.docvecs[x.tag], X_data)
    print(_X_train[0])
    sys.exit()

    # Build Model
    print('Train Fully Connected Model')
    model = Sequential()
    # model.add(Embedding(TOP_WORDS, 32, input_length=MAX_WORDS))
    # model.add(Flatten())
    model.add(Dense(150, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Train Model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

    # Generate Predictions
    y_pred = logistic.predict(X_test)
    y_prob = logistic.predict_proba(X_test)

    # Append to Report
    y_prob = map(lambda x: x[1][x[0]], zip(y_pred, y_prob))
    report.append_result(Y[test_idx].reshape(y_pred.shape), y_pred, y_prob)

# Generate Prediction Reports
report.generate_report()
