'''
NLP 120 Features + Random Forest Model
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function

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

# Filter Out Invalid Results
# TODO: IF THERE IS A FILTER ISSUE WE NEED TO FIX ASAP!
print('PREFILTER: ' + str(len(X)))
X = filter(lambda x: len(x) == 121, X)
print('POSTFILTER: ' + str(len(X)))
