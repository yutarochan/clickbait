'''
Doc2Vec (Headline) + Logistic Function
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import sys
import random
import gensim
import string
import logging
import numpy as np
from nltk import word_tokenize
from operator import itemgetter
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from keras.models import Sequential
from keras.layers import Dense

sys.path.append('..')
from util.load_data import JSONData
from util.scores import ScoreReport
from doc2vec import LabeledDocs, DocList

import warnings
warnings.filterwarnings("ignore")

# Logging Information for Gensim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

''' Constants and Parameters '''
DATA_ROOT = '../../Data/dataset/'           # Root Folder of where Dataset Resides
MODEL_ROOT = '../../Models/dataset/'        # Root Folder of where Model Resides
MODEL_ROOT = '../../Models/'                # Root Folder of where Embedding Model Resides

# Model Hyperparameters
K_FOLD = 10
SHUFFLE_FOLDS = True
np.random.seed(9892)                        # Seed Parameter for PRNG

NN_EPOCH = 10

# Document Embedding Hyperparameters
WORKERS = 8       # Total Worker Threads
EPOCHS = 15       # Total Epoch Count
PER_EPOCH = 15    # Per Epoch Count
DIM = 150         # Total Dimension of the Word Embedding
MIN_COUNT = 2     # Minimum Window Size Count

report = ScoreReport('Doc2Vec '+str(DIM)+' Dimension - Headline + Neural Network')  # Automated Score Reporting Utility

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

    ''' Generate Wrapper for Training Set '''
    X_LD = map(lambda x: LabeledDocs(x[0], x[1], Y[x[1]][0]), zip(X[train_idx], range(len(X[train_idx]))))
    X_data = DocList(docs=X_LD)

    ''' Build Word Embedding Feature Vector '''
    print('Training Word Embedding Model')
    emb_model = gensim.models.doc2vec.Doc2Vec(size=DIM, min_count=MIN_COUNT, iter=PER_EPOCH, workers=WORKERS)
    emb_model.build_vocab(X_data.toDocEmbArray())
    tmp = X_data.toDocEmbArray()
    for _ in range(EPOCHS):
        random.shuffle(tmp)
        emb_model.train(tmp, total_examples=emb_model.corpus_count, epochs=emb_model.iter)

    # Persist Weights
    # print('Persisting Weights')
    emb_model.save(MODEL_ROOT+'d2v_h_fcn_'+str(EPOCHS)+'_'+str(PER_EPOCH)+'_DIM-'+str(DIM)+'_K-'+str(i+1))

    # Convert Text to Word Embeddings
    _X_train = np.array(map(lambda x: emb_model.docvecs[x.tag], X_data))
    _X_test = np.array(map(lambda x: emb_model.infer_vector(x), X[test_idx]))

    ''' SMOTE - Generate Synthetic Data '''
    # sm = SMOTE(kind='regular')
    # X_resampled = []
    # X_res, Y_res = sm.fit_sample(np.array(_X_train), np.array(Y[train_idx]))

    ''' Build Neural Network Model '''
    model = Sequential()
    
    model.add(Dense(100, input_shape=(DIM, )))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    model.fit(_X_train, Y[train_idx], epochs=NN_EPOCH, validation_data=(_X_test, Y[test_idx]), verbose=1)

    # Infer Feature Vectors Based on Trained Model
    # _X_test = np.array(map(lambda x: emb_model.infer_vector(x), X[test_idx]))

    # Generate Predictions & Confidence Estimates
    y_pred = model.predict_classes(_X_test)
    y_prob = model.predict_proba(_X_test)
    
    # Append to Report
    y_prob = map(lambda x: x[0], y_prob)
    report.append_result(Y[test_idx].reshape(y_pred.shape), y_pred, y_prob)
    
# Generate Prediction Reports
report.generate_report()
