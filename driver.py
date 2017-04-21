import numpy as np
import os
import sys

from collections import defaultdict
from pandas import read_csv, DataFrame
from sklearn.svm import SVC

class SVM(object):
    def __init__(self, _class_='', _time=-1, _clf=SVC()):
        self.class_ = _class_
        self.time = _time
        self.clf = _clf

class Data(object):
    def __init__(self, _time=-1):
        self.time = _time
        self.data = DataFrame()

    def addData(self, label, _data=np.array([])):
        self.data[label] = _data[0]

training = defaultdict(list)
testing = defaultdict(DataFrame)

# This isn't working correctly. Have to change how files are read
os.chdir('data/')
for d in os.listdir(os.curdir):
    if d.startswith('tctodd'):
        for f in os.listdir(d):
            print 'Reading from file:', f
            loc = f.rfind('-')
            word = f[:loc]
            num  = int(f[loc+1: loc+2])
            f = os.getcwd() + '/' + d + '/' + f
            with open(f, 'r') as fp:
                temp = read_csv(f, delimiter='\t', header=None)
                if num == 3:
                    testing[word] = temp
                else:
                    training[word].append(temp)
os.chdir('../')

# Structuring Data
time_data = defaultdict(Data)
time = 0;
keys = training.keys()
while keys:
    temp_data = Data(_time=time)
    for key in keys:
        if time < len(training[key][0]):
            temp_data.addData(key + '0', training[key][0].loc[[time]].as_matrix())
        if time < len(training[key][1]):
            temp_data.addData(key + '1', training[key][1].loc[[time]].as_matrix())
        else:
            keys.remove(key)
    time_data[time] = (temp_data)
    time += 1


# Classes -> Times -> Classifiers
SVMs = defaultdict(lambda : defaultdict(SVC))

# Training Classifiers
for tim, dat in time_data.iteritems():
    signs = set([key[:-1] for key in dat.data.columns])
    for sign in signs:

        # Get all signs from extended labels
        labels = [lab for lab in dat.data.columns if lab.startswith(sign)]
        temp_cols = list(dat.data.columns)
        for lab in labels:
            temp_cols.remove(lab)
        Y = [0] * len(temp_cols)
        temp_cols = labels + temp_cols
        Y = [1] * len(labels) + Y
        if not all(x == Y[0] for x in Y):
            clf = SVC()
            clf.fit(dat.data[temp_cols].T.as_matrix(), Y)
            SVMs[sign][tim] = clf

# Testing
