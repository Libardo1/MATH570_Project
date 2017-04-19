import numpy as np
import os
import sys

from collections import defaultdict
from pandas import read_csv, DataFrame
from sklearn.svm import SVC

words = defaultdict(DataFrame)

# This isn't working correctly. Have to change how files are read
os.chdir('data/')
for d in os.listdir(os.curdir):
    if d.startswith('tctodd'):
        for f in os.listdir(d):
            word = f[:f.find('-')]
            f = os.getcwd() + '/' + d + '/' + f
            print 'Reading from file:', f
            with open(f, 'r') as fp:
                temp = read_csv(f, delimiter='\t', header=None)
                words[word] = words[word].append(temp)
os.chdir('../')

training = defaultdict(DataFrame)
testing = defaultdict(DataFrame)

# Divide data in 75 - 25 training - testing
for key in words.keys():
    print int(len(words[key]) * 0.75)
    training[key] = words[key].head(int(len(words[key]) * 0.75))
    testing[key]  = words[key].tail(int(len(words[key]) * 0.25))

