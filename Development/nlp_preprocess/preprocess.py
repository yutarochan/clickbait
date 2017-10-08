'''
Feature Extraction Script
Script used to perform the preprocessing externally.

Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import sys
import string
import itertools
import numpy as np
from nltk.util import ngrams
from collections import Counter
from multiprocessing import Pool
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from textstat.textstat import textstat as ts

sys.path.append('..')
from text2num import text2num
from util.load_data import JSONData

# Supress Warnings for Package
import warnings
warnings.filterwarnings("ignore")

# UTF-8 Encoding Issue
reload(sys)
sys.setdefaultencoding('utf8')

''' Constants and Parameters '''
DATA_ROOT = '../../Data/dataset/'       # Root Folder of where Dataset Resides
MODEL_ROOT = '../../Models/dataset/'    # Root Folder of where Model Resides

POOL_THREADS = 64

''' Import Data '''
# Load Dataset
print('Loading Dataset...')
data_load = JSONData(DATA_ROOT+'instances_train.jsonl', DATA_ROOT+'truth_train.jsonl', DATA_ROOT+'instances_test.jsonl')
train_X = data_load.load_train_X()
train_Y = data_load.load_train_Y()

''' Define Preprocessing Functions '''
pos_list = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

def is_numeric(text):
    try: return type(text2num(text)) == type(0)
    except Exception as e: return False

def pos_2gram(title, p1, p2):
    pos_list = pos_tag(word_tokenize(title))
    return sum(map(lambda x: x[0][1] == p1 and x[1][1] == p2, zip(pos_list[:-1], pos_list[1:])))

def pos_3gram(title, p1, p2, p3):
    pos_list = pos_tag(word_tokenize(title))
    return sum(map(lambda x: x[0][0][1] == 'NNP' and x[0][1][1] == 'NNP' and x[1][1] == 'VBZ', zip(zip(pos_list[:-1], pos_list[1:]), pos_list[2:])))

def mean_wordlen(text):
    word_lens = map(lambda x: map(lambda y: len(y), word_tokenize(x)), text)
    return np.mean(list(itertools.chain.from_iterable(word_lens)))

def max_wordlen(text):
    word_lens = map(lambda x: map(lambda y: len(y), word_tokenize(x)), text)
    return np.max(list(itertools.chain.from_iterable(word_lens)))

def pos_thnn(title):
    pos_list = pos_tag(word_tokenize(title))
    return sum(map(lambda x: (x[0][0].lower() == 'this' or x[0][0].lower() == 'these') and x[1][1] == 'NN', zip(pos_list[:-1], pos_list[1:])))

def kw_post_match(kw, post):
    return len((set(word_tokenize(kw.lower())) - set(word_tokenize(' '.join(post).lower()))) - set(stopwords.words('english')) - set(list(string.punctuation)))

def pos_text_ratio(text, pos):
    return len(filter(lambda x: x[1] == pos, pos_tag(word_tokenize(' '.join(text))))) / float(len(word_tokenize(' '.join(text))))

def ngram_ptext(text, n):
    return sum(Counter(ngrams(word_tokenize(' '.join(text)), n)).values())

nnp_num = lambda x: sum(map(lambda y: y[1] == 'NNP', pos_tag(word_tokenize(x))))
wlen_title = lambda text: len(filter(lambda x: x.isalpha(), word_tokenize(text)))
num_start = lambda x: x[0].isdigit() or is_numeric(x[0])
in_num = lambda t: sum(map(lambda x: x[1] == 'IN', pos_tag(word_tokenize(t))))
wrb_num = lambda t: sum(map(lambda x: x[1] == 'WRB', pos_tag(word_tokenize(t))))
nnp_num = lambda t: sum(map(lambda x: x[1] == 'NN', pos_tag(word_tokenize(t))))
wh_start = lambda t: word_tokenize(t)[0].lower() in ['who', 'what', 'why', 'where', 'when', 'how']
qm_exist = lambda t: sum(map(lambda x: str(x) == '?', word_tokenize(t))) > 0
prp_count = lambda t: sum(map(lambda x: x[1] == 'PRP', pos_tag(word_tokenize(t))))
vbz_count = lambda t: sum(map(lambda x: x[1] == 'VBZ', pos_tag(word_tokenize(t))))
sw_ratio = lambda t: sum(map(lambda x: x.lower() in stopwords.words('english'), word_tokenize(t))) / float(len(word_tokenize(t)))
wp_count = lambda t: sum(map(lambda x: x[1] == 'WP', pos_tag(word_tokenize(t))))
dt_count = lambda t: sum(map(lambda x: x[1] == 'DT', pos_tag(word_tokenize(t))))
pos_count = lambda t: sum(map(lambda x: x[1] == 'POS', pos_tag(word_tokenize(t))))
comma_count = lambda t: len(filter(lambda x: x == ',', word_tokenize(t)))
wdt_count = lambda t: sum(map(lambda x: x[1] == 'WDT', pos_tag(word_tokenize(t))))
rb_count = lambda t: sum(map(lambda x: x[1] == 'RB', pos_tag(word_tokenize(t))))
rbs_count = lambda t: sum(map(lambda x: x[1] == 'RBS', pos_tag(word_tokenize(t))))
vbn_count = lambda t: sum(map(lambda x: x[1] == 'VBN', pos_tag(word_tokenize(t))))
ex_exist = lambda t: int(sum(map(lambda x: x[1] == 'EX', pos_tag(word_tokenize(t)))) > 0)
ratio = lambda t: map(lambda x: pos_text_ratio(t, x), pos_list)
ngram_feat = lambda x: [ngram_ptext(x, i) for i in range(6)]

def preprocess(x):
    print('PROCESSING ID: ' + str(x['id']))
    try:
        fvec = []
        fvec.append(int(x['id'])) # Append Article ID
        fvec.append(nnp_num(x['targetTitle']))
        if len(x['targetParagraphs']) > 0:
            fvec.append(ts.automated_readability_index(' '.join(x['targetParagraphs'])))
            fvec.append(ts.avg_letter_per_word(' '.join(x['targetParagraphs'])))
            fvec.append(ts.avg_sentence_length(' '.join(x['targetParagraphs'])))
            fvec.append(ts.avg_sentence_per_word(' '.join(x['targetParagraphs'])))
            fvec.append(ts.avg_syllables_per_word(' '.join(x['targetParagraphs'])))
            fvec.append(ts.char_count(' '.join(x['targetParagraphs'])))
            fvec.append(ts.coleman_liau_index(' '.join(x['targetParagraphs'])))
            fvec.append(ts.dale_chall_readability_score(' '.join(x['targetParagraphs'])))
            fvec.append(ts.difficult_words(' '.join(x['targetParagraphs'])))
            fvec.append(ts.flesch_kincaid_grade(' '.join(x['targetParagraphs'])))
            fvec.append(ts.flesch_reading_ease(' '.join(x['targetParagraphs'])))
            fvec.append(ts.gunning_fog(' '.join(x['targetParagraphs'])))
            fvec.append(ts.lexicon_count(' '.join(x['targetParagraphs'])))
            fvec.append(ts.linsear_write_formula(' '.join(x['targetParagraphs'])))
            fvec.append(ts.polysyllabcount(' '.join(x['targetParagraphs'])))
            fvec.append(ts.sentence_count(' '.join(x['targetParagraphs'])))
            fvec.append(ts.smog_index(' '.join(x['targetParagraphs'])))
            fvec.append(ts.syllable_count(' '.join(x['targetParagraphs'])))
            fvec.append(mean_wordlen(x['targetParagraphs']))
            fvec += ratio(x['targetParagraphs'])
            fvec += ngram_feat(x['targetParagraphs'])
        else:
            fvec += [0]*55
        if len(word_tokenize(' '.join(x['postText']))) > 0:
            fvec.append(max_wordlen(x['postText']))
            fvec.append(sw_ratio(' '.join(x['postText'])))
            fvec += ngram_feat(x['postText'])
        else:
            fvec += [0]*8
        fvec.append(len(word_tokenize(x['targetTitle'])))
        fvec.append(wlen_title(x['targetTitle']))
        fvec.append(pos_2gram(x['targetTitle'], 'NNP', 'NNP'))
        fvec.append(int(num_start(x['targetTitle'])))
        fvec.append(in_num(x['targetTitle']))
        fvec.append(pos_2gram(x['targetTitle'], 'NNP', 'VBZ'))
        fvec.append(pos_2gram(x['targetTitle'], 'IN', 'NNP'))
        fvec.append(wrb_num(x['targetTitle']))
        fvec.append(nnp_num(x['targetTitle']))
        fvec.append(int(wh_start(x['targetTitle'])))
        fvec.append(int(qm_exist(x['targetTitle'])))
        fvec.append(pos_thnn(x['targetTitle']))
        fvec.append(prp_count(x['targetTitle']))
        fvec.append(vbz_count(x['targetTitle']))
        fvec.append(pos_3gram(x['targetTitle'], 'NNP', 'NNP', 'VBZ'))
        fvec.append(pos_2gram(x['targetTitle'], 'NN', 'IN'))
        fvec.append(pos_3gram(x['targetTitle'], 'NN', 'IN', 'NNP'))
        fvec.append(pos_2gram(x['targetTitle'], 'NNP', '.'))
        fvec.append(pos_2gram(x['targetTitle'], 'PRP', 'VBP'))
        fvec.append(wp_count(x['targetTitle']))
        fvec.append(dt_count(x['targetTitle']))
        fvec.append(pos_2gram(x['targetTitle'], 'NNP', 'IN'))
        fvec.append(pos_3gram(x['targetTitle'], 'IN', 'NNP', 'NNP'))
        fvec.append(pos_count(x['targetTitle']))
        fvec.append(pos_2gram(x['targetTitle'], 'IN', 'NN'))
        if len(x['targetKeywords']) > 0 and len(x['postText']) > 0:
            fvec.append(kw_post_match(x['targetKeywords'], x['postText']))
        else:
            fvec += [0]*1
        fvec.append(comma_count(x['targetTitle']))
        fvec.append(pos_2gram(x['targetTitle'], 'NNP', 'NNS'))
        fvec.append(pos_2gram(x['targetTitle'], 'IN', 'JJ'))
        fvec.append(pos_2gram(x['targetTitle'], 'NNP', 'POS'))
        fvec.append(wdt_count(x['targetTitle']))
        fvec.append(pos_2gram(x['targetTitle'], 'NN', 'NN'))
        fvec.append(pos_2gram(x['targetTitle'], 'NN', 'NNP'))
        fvec.append(pos_2gram(x['targetTitle'], 'NNP', 'VBD'))
        fvec.append(rb_count(x['targetTitle']))
        fvec.append(pos_3gram(x['targetTitle'], 'NNP', 'NNP', 'NNP'))
        fvec.append(pos_3gram(x['targetTitle'], 'NNP', 'NNP', 'NN'))
        fvec.append(rbs_count(x['targetTitle']))
        fvec.append(vbn_count(x['targetTitle']))
        fvec.append(pos_2gram(x['targetTitle'], 'VBN', 'IN'))
        fvec.append(pos_2gram(x['targetTitle'], 'JJ', 'NNP'))
        fvec.append(pos_3gram(x['targetTitle'], 'NNP', 'NN', 'NN'))
        fvec.append(pos_2gram(x['targetTitle'], 'DT', 'NN'))
        fvec.append(ex_exist(x['targetTitle']))
        fvec += ngram_feat(x['targetTitle'])
    except Exception as e:
        print('EXCEPTION AT ID ' + str(x['id']))
        print(e)
        sys.exit()

    return fvec

''' Perform Preprocessing '''
def build_rec(row):
    data = ''
    for i, x in enumerate(row):
        if i < len(row)-1:
            data += str(x)+','
        else:
            data += str(x)
    data += '\n'
    return data

'''
p = Pool(POOL_THREADS)
X = p.map(preprocess, train_X[101:201])
p.close()
p.join()

print('\nWriting Results to File')
output = open('feat001.csv', 'wb')
map(lambda x: output.write(build_rec(x)), X)
output.close()

print('\nDONE!')
'''

print(train_X[2235])
print(preprocess(train_X[2235]))
