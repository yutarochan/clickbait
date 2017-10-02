'''
Doc2Vec - Data Utility Module
Auxilary functions to handle dataset loading and preprocessing tasks for Gensim's Document Embeddings.
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
import os
import sys
import csv
import nltk
import string
import random
import numpy as np
from glob import glob
from pandas import DataFrame
from functools import partial
from multiprocessing import Pool
from collections import defaultdict
from gensim.models.doc2vec import TaggedDocument

class LabeledDocs(object):
    def __init__(self, text, tag, label):
        self.text = text
        self.tag = tag
        self.label = label

    def getTokens(self):
        if type(self.text) is not list:
            return [i.lower() for i in nltk.word_tokenize(self.text)]
        else:
            return self.text

    def getTD(self):
        return TaggedDocument(words=self.text, tags=[self.tag])

class LabeledDocEmb(object):
    def __init__(self, tokens, tag, label):
        self.doc = TaggedDocument(words=tokens, tags=[tag])
        self.tag = tag
        self.label = label

class DocList(object):
    def __init__(self, docs=None):
        if docs is None: self.docs = []
        else: self.docs = docs

    def add(self, text, tag, label):
        self.docs.append(LabeledDocs(tokens, doc_tag, int(doc_file['label'])))

    def append(self, doc_obj):
        self.docs.append(doc_obj)

    def toArray(self):
        docList_arr = []
        for d in self.docs: docList_arr.append(d)
        return docList_arr

    def toDocEmbArray(self):
        docList_arr = []
        for d in self.docs: docList_arr.append(TaggedDocument(words=d.text, tags=[d.tag]))
        return docList_arr

    def tokenArray(self):
        docList_arr = []
        for d in self.docs: docList_arr.append(d.getTokens())
        return docList_arr

    def __iter__(self):
        for d in self.docs: yield d

    def split(self, ratio=0.25, shuffle=True):
        if shuffle: random.shuffle(self.docs)
        return DocList(self.docs[int(len(self.docs)*ratio)+1:]), DocList(self.docs[0:int(len(self.docs)*ratio)])

    def kfold(self, k=5, shuffle=True):
        if shuffle: random.shuffle(self.docs)
        chunk = self.size() / k
        for i in range(0, self.size(), (self.size()/k)):
            if len(self.toArray()[i : i+self.size()/k]) != chunk:
                break
            else:
                yield DocList(self.toArray()[0: i] + self.toArray()[i+self.size()/k : self.size()]), DocList(self.toArray()[i : i+self.size()/k])

    def size(self):
        return len(self.docs)

    def count(self):
        pos = sum([x for x in doc if x.label is 1])
        return { 'pos' : pos, 'neg' : len(self.doc) - pos }

    def word_count(self):
        return sum([len(d.text) for d in self.docs])

    def getLabels(self):
        for d in self.docs: yield d.label

    def getUnique(self):
        frequency = defaultdict(int)
        for text in self.toArray():
            for token in text.getTokens(): frequency[token] += 1
        return [[token for token in text.getTokens() if frequency[token] > 1] for text in self.toArray()]
