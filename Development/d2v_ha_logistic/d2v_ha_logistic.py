'''
Doc2Vec (Headline + Article) + Logistic Function
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
MODEL_ROOT = '../../Models/d2v_ha_logistic/' # Root Folder of where Embedding Model Resides

# Model Hyperparameters
K_FOLD = 10
SHUFFLE_FOLDS = True
np.random.seed(9892)                        # Seed Parameter for PRNG

# Document Embedding Hyperparameters
WORKERS = 32      # Total Worker Threads
EPOCHS = 1        # Total Epoch Count
PER_EPOCH = 15    # Per Epoch Count
HEAD_DIM = 200    # Total Dimension of the Headline Word Embedding
ARTI_DIM = 200    # Total Dimension of the Article Word Embedding
MIN_COUNT = 2     # Minimum Window Size Count

report = ScoreReport('Doc2Vec '+str(HEAD_DIM+ARTI_DIM)+' Dimension [1 EPOCH VARIANT] - Headline-Article + Logistic Function')  # Automated Score Reporting Utility

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
X_headline = np.array(map(lambda x: preprocess(x['targetTitle']), train_X))
X_article = np.array(map(lambda x: preprocess(' '.join(x['targetParagraphs'])), train_X))
Y = np.array(map(lambda x: [0] if x['truthClass'] == 'no-clickbait' else [1], train_Y))

''' CV Model Training '''
# K-Fold and Score Tracking
kf = KFold(n_splits=K_FOLD, shuffle=SHUFFLE_FOLDS)

print('Training Model...')
for i, (train_idx, test_idx) in enumerate(kf.split(X_headline)):
    print('\n[K = ' + str(i+1) + ']')

    ''' Generate Wrapper for Training Set '''
    X_DL_headline = DocList(docs=map(lambda x: LabeledDocs(x[0], x[1], Y[x[1]][0]), zip(X_headline[train_idx], range(len(X_headline[train_idx])))))
    X_DL_article = DocList(docs=map(lambda x: LabeledDocs(x[0], x[1], Y[x[1]][0]), zip(X_headline[train_idx], range(len(X_headline[train_idx])))))

    ''' Build Word Embedding Feature Vector '''
    print('Training Headline Word Embedding Model')
    h_model = gensim.models.doc2vec.Doc2Vec(size=HEAD_DIM, min_count=MIN_COUNT, iter=PER_EPOCH, workers=WORKERS)
    h_model.build_vocab(X_DL_headline.toDocEmbArray())
    tmp = X_DL_headline.toDocEmbArray()
    for _ in range(EPOCHS):
        random.shuffle(tmp)
        h_model.train(tmp, total_examples=h_model.corpus_count, epochs=h_model.iter)

    print('Training Article Word Embedding Model')
    a_model = gensim.models.doc2vec.Doc2Vec(size=ARTI_DIM, min_count=MIN_COUNT, iter=PER_EPOCH, workers=WORKERS)
    a_model.build_vocab(X_DL_article.toDocEmbArray())
    tmp = X_DL_article.toDocEmbArray()
    for _ in range(EPOCHS):
        random.shuffle(tmp)
        a_model.train(tmp, total_examples=a_model.corpus_count, epochs=a_model.iter)

    # Persist Weights
    print('Persisting Weights')
    h_model.save(MODEL_ROOT+'h2v_cb_h_'+str(EPOCHS)+'_'+str(PER_EPOCH)+'_HDIM-'+str(HEAD_DIM)+'_ADIM-'+str(ARTI_DIM)+'_K-'+str(i+1))
    a_model.save(MODEL_ROOT+'h2v_cb_a_'+str(EPOCHS)+'_'+str(PER_EPOCH)+'_HDIM-'+str(HEAD_DIM)+'_ADIM-'+str(ARTI_DIM)+'_K-'+str(i+1))

    # Convert Text to Word Embeddings
    _X_train = map(lambda z: np.concatenate(z, axis=0), zip(map(lambda x: h_model.docvecs[x.tag], X_DL_headline), map(lambda y: a_model.docvecs[y.tag], X_DL_article)))

    ''' SMOTE - Generate Synthetic Data '''
    # sm = SMOTE(kind='regular')
    # resampled = []
    # X_res, Y_res = sm.fit_sample(np.array(_X_train), np.array(Y[train_idx]))

    ''' Train Classification Model '''
    # Train Model
    logistic = LogisticRegression(class_weight='balanced')
    logistic.fit(_X_train, Y[train_idx])

    # Infer Feature Vectors Based on Trained Model
    _X_test = map(lambda z: np.concatenate(z, axis=0), zip(map(lambda x: h_model.infer_vector(x), X_headline[test_idx]), map(lambda x: a_model.infer_vector(x), X_article[test_idx])))

    # Generate Predictions & Confidence Estimates
    y_pred = logistic.predict(_X_test)
    y_prob = logistic.predict_proba(_X_test)

    # Append to Report
    y_prob = map(lambda x: x[1][x[0]], zip(y_pred, y_prob))
    report.append_result(Y[test_idx].reshape(y_pred.shape), y_pred, y_prob)

# Generate Prediction Reports
report.generate_report()
