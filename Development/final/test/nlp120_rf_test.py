'''
NLP 120 Features + Random Forest Model
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import sys
import csv
import pickle
import numpy as np

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
data = csv.reader(open('../../../Data/test_feat.csv', 'rb'), delimiter=',')
X = map(lambda x: map(lambda y: float(y), x), data)

''' Generate Predictions '''
# Load Model from Pickle
model_data = open(MODEL_ROOT+'nlp120_rf.pkl', 'wb')
model = pickle.load(model_data)

# Generate Predictions
output = open('predictions.csv', 'wb')
map(lambda x: output.write(str(int(x[0])) + ',' + model.predict(x[1:])))

output.close()
model_data.close()

print('DONE!')
