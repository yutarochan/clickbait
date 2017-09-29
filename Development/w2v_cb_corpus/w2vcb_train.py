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
DATA_ROOT = '../../Data/dataset/'       # Root Folder of where Dataset Resides
MODEL_ROOT = '../../Models/dataset/'    # Root Folder of where Model Resides

WORKERS = 8

# Word Embedding Hyperparameters
EPOCHS = 20       # Total Epoch Count
PER_EPOCH = 20    # Per Epoch Count
DIM = 300         # Total Dimension of the Word Embedding 
MIN_COUNT = 3     # Minimum Window Size Count

''' Import Data '''
# Load Dataset
print('Loading Dataset')
data_load = JSONData(DATA_ROOT+'instances_train.jsonl', DATA_ROOT+'truth_train.jsonl', DATA_ROOT+'instances_test.jsonl')
train_X = data_load.load_train_X()
train_Y = data_load.load_train_Y()

''' Preprocess Data '''
def preprocess(text):
    text = word_tokenize(text.lower())                                      # Tokenize & Normalize Text
    text = filter(lambda x: x not in string.punctuation, text)              # Remove Punctuation
    return text
                                    
# Preprocess Headline and Article Dataset
print('Performing Preprocessing')
data = map(lambda x: preprocess(x['targetTitle']), train_X)    # Headline Only
# data = map(lambda x: preprocess, train_X)

''' Train Embedding Model '''
print('Training Word Embedding Model')
model = gensim.models.Word2Vec(data, min_count=MIN_COUNT, workers=WORKERS)
for i in range(EPOCHS):
    model.train(data, total_examples=len(data), epochs=PER_EPOCH)

print('Test Similarity')
model.most_similar(positive=[''])
