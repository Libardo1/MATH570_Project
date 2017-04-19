import numpy as np
import os
import sys

from collections import defaultdict
from pandas import read_csv, DataFrame
from sklearn.svm import SVC

words = defaultdict(DataFrame)

for d in os.listdir(os.curdir):
    if d.startswith('tctodd'):
        for f in os.listdir(d):
            word = f[:f.find('-')]
            f = os.getcwd() + '/' + d + '/' + f
            print 'Reading from file:', f
            with open(f, 'r') as fp:
                temp = read_csv(f, delimiter='\t', header=None)
                words[word] = words[word].append(temp)

print len(words['give'])
print words['give']

print len(words.keys())
#for key in words.keys():
