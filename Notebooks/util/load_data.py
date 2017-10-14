'''
Load Dataset Modules
Auxillary class used to load the dataset provided to us.

Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import json

class JSONData:
    def __init__(self, train, label, test):
        self.train_x_path = train
        self.train_y_path = label
        self.test_path = test

    def load_train_X(self):
        instance_raw = open(self.train_x_path, 'rb').read().replace('\"', '"').split('\n')[:-1]
        return map(json.loads, instance_raw)

    def load_train_Y(self):
        truth_raw = open(self.train_y_path, 'rb').read().replace('\"', '"').split('\n')[:-1]
        return map(json.loads, truth_raw)

    def load_test(self):
        test_raw = open(self.test_path, 'rb').read().replace('\"', '"').split('\n')[:-1]
        return map(json.loads, test_raw)

# Unit Testing
if __name__ == '__main__':
    DATA_ROOT = '../../Data/'
    loader = JSONData(DATA_ROOT+'dataset/instances_train.jsonl', DATA_ROOT+'dataset/truth_train.jsonl', DATA_ROOT+'dataset/instances_test.jsonl')

    print('TRAIN X SIZE: ' + str(len(loader.load_train_X())))
    print('TRAIN Y SIZE: ' + str(len(loader.load_train_Y())))
    print('TEST SIZE: ' + str(len(loader.load_test())))
