'''
Baseline Model
Simple baseline model based on example features from the instruction sets

Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import sys
import string
import numpy as np
from nltk import word_tokenize
from sklearn.cross_validation import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, f1_score, classification_report

sys.path.append('..')
from text2num import text2num
from util.load_data import JSONData

import warnings
warnings.filterwarnings("ignore")

''' Constants and Parameters '''
DATA_ROOT = '../../Data/dataset/'       # Root Folder of where Dataset Resides
MODEL_ROOT = '../../Models/dataset/'    # Root Folder of where Model Resides
K_FOLD = 10
SHUFFLE_FOLDS = True
np.random.seed(9892)                # Seed Parameter for PRNG

''' Import Data '''
# Load Dataset
data_load = JSONData(DATA_ROOT+'instances_train.jsonl', DATA_ROOT+'truth_train.jsonl', DATA_ROOT+'instances_test.jsonl')
train_X = data_load.load_train_X()
train_Y = data_load.load_train_Y()

''' Preprocess Data '''
# Build Feature Vector
def is_numeric(text):
    try: return type(text2num(text)) == type(0)
    except Exception as e: return False

def preprocess(text):
    text = word_tokenize(text.lower())                                      # Tokenize & Normalize Text
    text = filter(lambda x: x not in string.punctuation, text)              # Remove Punctuation

    # Perform Feature Extraction
    word_count = len(text)                                                  # Total Word Count
    avg_word_len = sum(map(lambda x: len(x), text))/float(len(text))        # Average Word Length
    max_word_len = max(map(lambda x: len(x), text))                         # Longest Word Length
    is_number = text[0].isdigit() or is_numeric(text[0])                    # Check if starts with number (Either Numerically or Linguistically)
    start_ws = text[0] in ['who', 'what', 'why', 'where', 'when', 'how']    # Whether it starts with question word

    return [word_count, avg_word_len, max_word_len, int(is_number), int(start_ws)]

# Finalize Feature and Target Vectors
X = np.array(map(lambda x: preprocess(x['targetTitle']), train_X))
Y = np.array(map(lambda x: [0] if x['truthClass'] == 'no-clickbait' else [1], train_Y))

''' CV Model Training '''
# K-Fold and Score Tracking
kf = KFold(n=len(X), n_folds=K_FOLD, shuffle=SHUFFLE_FOLDS)

for i, (train_idx, test_idx) in enumerate(kf):
    print('\n[K = ' + str(i+1) + ']')
    # Train Model & Generate Predictions
    gnb = GaussianNB()
    y_pred = gnb.fit(X[train_idx], Y[train_idx]).predict(X[test_idx])

    print(confusion_matrix(Y[test_idx], y_pred))

    # confusion += confusion_matrix(Y[test_idx], y_pred)
    score = f1_score(Y[test_idx], y_pred, pos_label=1)
    scores.append(score)

    '''
    # Compute ROC curve and area the curve
    prd = gnb.predict_proba(X[test_idx])[:,1]
    fpr, tpr, thresholds = roc_curve(X[test_idx], prd)
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)

    if roc_auc is not None: plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    '''

    print(classification_report(Y[test_idx], y_pred))
    i += 1

''' Compute Average Scores
mean_tpr /= 10
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

# Export mean ROC curve values for aggregate ROC analysis.
mean_roc = np.vstack((mean_fpr.T, mean_tpr.T)).T
mroc_df = pd.DataFrame(data=mean_roc.astype(float))
mroc_df.to_csv('bow_10fold_roc.csv', sep=',', header=False, float_format='%.2f', index=False)
'''

''' Accuracy Reporting '''
print('Average F1 Score: ' + str(sum(scores)/len(scores)))
# print('Confusion matrix: ')
# print(confusion)

''' Full Model Training... '''
gnb = GaussianNB()
gnb.fit(X, Y)

def predict(sentence):
    print(sentence)
    if gnb.predict(preprocess(sentence))[0] == 0:
        print('NOT CLICKBAIT!')
    else:
        print('CLICKBAIT')
    print()

print()
predict('10 Things Yuya Screwed Up While Building A Simple Classifier Model')
predict('Five Unbelievable Things Going on in Nittany Data Labs!!!!')
predict('You won\'t believe how these 9 shocking clickbaits work!')
predict('You\'ll Never Look at Barbie Dolls The Same Once You See These Paintings')
predict('Well If It Isn\'t Coupons, I Can\'t Even IMAGINE What It Is!')
predict('Trump vows to back winner of Alabama GOP Senate runoff')
