'''
Scoring Auxilary Function
Class to help provide organize method to keep track of scores and report them.

Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import numpy as np
from sklearn import metrics as met

class ScoreReport:
    def __init__(self):
        self.acc_list = []      # Accuracy List
        self.tpr_list = []      # True Positive Rate List
        self.fpr_list = []      # False Positive Rate List
        self.f1_list = []       # F-Score (F1 Metric) List
        self.auc_list = []      # AUC/ROC List
        self.conf_list = []     # Confusion Matrix List

        self.mean_tpr = []      # Mean True Positive Rate List
        self.mean_fpr = []      # Mean False Positive Rate List

        '''
        self.scores = []
        self.confusions = []
        self.all_tpr = []
        self.mean_tpr = 0.0
        self.mean_fpr = np.linspace(0, 1, 100)
        '''

    def append_result(self, y_true, y_pred, y_prob):
        # assert(len(y_true) == len(y_pred) and len(y_pred) == len(y_prob))
        fpr, tpr, thresholds = roc_curve(test_label, y_prob)

        self.acc_list.append(met.accuracy_score(y_true, y_pred))
        self.f1_list.append(f1_score(y_true, y_pred, pos_label=1))
        self.conf_list.append(confusion_matrix(y_true, y_pred))

        # self.acc_list.append(map(lambda x: y_true == y_pred, zip(y_true, y_pred)) / float(len(y_true)))

    '''
    def append_score(self, score):
        self.scores.append(score)

    def average_score(self):
        return sum(self.scores) / float(len(self.scores))
    '''

# Unit Testing
if __name__ == '__main__':
    report = ScoreReport()
    y_pred = [1, 0, 1]
    y_true = [1, 1, 1]
    report.append_result(y_true, y_pred)
