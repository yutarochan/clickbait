'''
Word2Vec - ClickBait Corpus Only Training
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import sys
import string
import gensim
import logging
from nltk import word_tokenize

sys.path.append('..')
from util.load_data import JSONData

import warnings
warnings.filterwarnings("ignore")

# Logging Information
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

''' Constants and Parameters '''
DATA_ROOT = '../../Data/dataset/'             # Root Folder of where Dataset Resides
MODEL_ROOT = '../../Models/w2v_cb_corpus/'    # Root Folder of where Model Resides

WORKERS = 8

# Word Embedding Hyperparameters
EPOCHS = 15       # Total Epoch Count
PER_EPOCH = 15    # Per Epoch Count
DIM = 100         # Total Dimension of the Word Embedding 
MIN_COUNT = 3     # Minimum Window Size Count

''' Import Data '''
# Load Dataset
print('Loading Dataset')
data_load = JSONData(DATA_ROOT+'instances_train.jsonl', DATA_ROOT+'truth_train.jsonl', DATA_ROOT+'instances_test.jsonl')
train_X = data_load.load_train_X()
train_Y = data_load.load_train_Y()

print(train_X[0]['targetParagraphs'])

''' Preprocess Data '''
def preprocess(text):
    text = word_tokenize(text.lower())                                      # Tokenize & Normalize Text
    text = filter(lambda x: x not in string.punctuation, text)              # Remove Punctuation
    return text
                                    
# Preprocess Headline and Article Dataset
print('Performing Preprocessing')
data = map(lambda x: preprocess(x['targetTitle']), train_X)    # Headline Only

''' Train Embedding Model '''
print('Training Word Embedding Model')
model = gensim.models.Word2Vec(data, min_count=MIN_COUNT, workers=WORKERS)
for i in range(EPOCHS):
    model.train(data, total_examples=len(data), epochs=PER_EPOCH)

# Persist Weights
print('Persisting Weights')
model.save(MODEL_ROOT+'w2v_cb_headline_'+str(EPOCHS)+'_'+str(PER_EPOCH)+'_DIM-'+str(DIM)+'.bin.gz')
