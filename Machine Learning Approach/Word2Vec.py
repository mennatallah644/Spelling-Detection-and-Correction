# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings 
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim

#
#fname = get_tmpfile("vectors.kv") 
#word_vectors.save(fname) word_vectors = KeyedVectors.load(fname, mmap='r')
#model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4) 
#word_vectors = model.wv 
#fname = get_tmpfile("vectors.kv") 
#word_vectors.save(fname) 
#word_vectors = KeyedVectors.load(fname, mmap='r')
#
#
#wv_from_text = KeyedVectors.load_word2vec_format(datapath(''), binary=False) # C text format
#wv_from_bin = KeyedVectors.load_word2vec_format(datapath("euclidean_vectors.bin"), binary=True) # C bin format

from gensim.test.utils import datapath


model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz',binary=True)





words = model.index2word

print " words length "
print len(words)


w_rank = {}
for i,word in enumerate(words):
    w_rank[word] = i

WORDS = w_rank




def words(text): return re.findall(r'\w+', text.lower())

def P(word): 
    "Probability of `word`."
    # use inverse of rank as proxy
    print "Probability of `word`."
    print "p function "
    # returns 0 if the word isn't in the dictionary
    print - WORDS.get(word, 0)
    return - WORDS.get(word, 0)

def correction(word): 
    "Most probable spelling correction for word."
    print "correction function"
    print "Most probable spelling correction for word."
    print max(candidates(word), key=P)
    return max(candidates(word), key=P)

def candidates(word): 
    print "candidates"
    "Generate possible spelling corrections for word."
    print "Generate possible spelling corrections for word."
    print (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])
#contenpted
def known(words): #[contenpted]
    "The subset of `words` that appear in the dictionary of WORDS."
    print "known function"
    print "The subset of `words` that appear in the dictionary of WORDS."
    print set(w for w in words if w in WORDS)
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    print " edit1"
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    print "All edits that are one edit away from `word`."
    print  set(deletes + transposes + replaces + inserts)
    return set(deletes + transposes + replaces + inserts)

def edits2(word):  #contenpted
    "All edits that are two edits away from `word`."
    print "edit 2 function"
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import re
from collections import Counter


import time
predict =[]
label=[]
def spelltest(tests, verbose=False):
    print " spell test function started"
    "Run correction(wrong) on all (right, wrong) pairs; report results."
    start = time.clock()
    good, unknown = 0, 0
    n = len(tests)
    for right, wrong in tests:
        print " wrong word is " + wrong
        w = correction(wrong)
        print " corrected word is " + w
        predict.append(w)
        print " right word is " + right
        label.append(right)
        print  " is the 2 words are the same ? " + w == right
        good += (w == right)
        if w != right:
            unknown += (right not in WORDS)
            if verbose:
                print'correction({}) => {} ({}); expected {} ({})'.format(wrong, w, WORDS[w], right, WORDS[right])
    dt = time.clock() - start
    print '{:.0%} of {} correct ({:.0%} unknown) at {:.0f} words per second '.format(good / n, n, unknown / n, n / dt) 
    print good
    print unknown 
    
#[ (contented,contenpted) (contented, contende) (contented, contended) (contented, contentid)]

def Testset(lines):
    print " test set is ready"
    "Parse 'right: wrong1 wrong2' lines into [('right', 'wrong1'), ('right', 'wrong2')] pairs."
    return [(right, wrong)
            for (right, wrongs) in (line.split(':') for line in lines)
            for wrong in wrongs.split()]



spelltest(Testset(open('spell-testset1.txt')))


print correction('quikly')
print correction('israil')
print correction('neighbour')

print classification_report(label,predict)
print len(predict)
print "Confusion matrix"
print confusion_matrix(label,predict)
# txt= "first is thr firt year of college"
# print([correction(word) for word in txt.split(" ")])






















