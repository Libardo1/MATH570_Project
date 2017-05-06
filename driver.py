import numpy as np
import pickle
import os
import sys
import time

from collections import defaultdict, namedtuple
from pandas import read_csv, DataFrame, Series, concat
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC, LinearSVC

import matplotlib.pyplot as plt

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

t0 = time.time()
training = defaultdict(list)
testing = defaultdict(DataFrame)

# This isn't working correctly. Have to change how files are read
print "\nReading in Data ....\n"
os.chdir('data/')
total_data = DataFrame([])
total_labs = []
for d in os.listdir(os.curdir):
    if d.startswith('tctodd'):
        for f in os.listdir(d):
            loc = f.rfind('-')
            word = f[:loc]
            num  = int(f[loc+1: loc+2])
            f = os.getcwd() + '/' + d + '/' + f
            with open(f, 'r') as fp:
                temp = read_csv(f, delimiter='\t', header=None)
                temp.replace([None, np.nan, np.inf], 0) 

                if len(temp) > 100:
                    temp = Series(temp.head(100).as_matrix().flatten())
                else:
                    zeroes = DataFrame(np.zeros((100 - len(temp), 22)))
                    temp = Series(concat([temp, zeroes]).as_matrix().flatten())
                    #temp = temp.append(zeroes, ignore_index=True).as_matrix().flatten()
                    # Fill up to 100 values

                total_data = total_data.append(temp, ignore_index=True)
                #total_data = total_data.append(temp, ignore_index=True)
                total_labs.append(word)
                if num == 3:
                    testing[word] = temp
                else:
                    training[word].append(temp)
os.chdir('../')

filename = 'RF_onevone_model.sav'
#pc_train = PCA(n_components=5)
#pc_train.fit(total_data.as_matrix())

whole_train = total_data.head(int(len(total_data) * 0.8))
whole_test  = total_data.tail(int(len(total_data) * 0.2))

train_labels = total_labs[:int(len(total_data) * 0.8)]
test_labels  = total_labs[int(len(total_data) * 0.2):]

print '\nTraining ....\n'

#'''
model = OneVsOneClassifier(RandomForestClassifier())

#model.fit(pc_train.transform(whole_train.as_matrix()), train_labels)
model.fit(whole_train.as_matrix(), train_labels)

pickle.dump(model, open(filename, 'wb'))
sys.exit()
#'''

model = pickle.load(open(filename, 'rb'))

#preds = model.predict(pc_train.transform(whole_test))
preds = model.predict(whole_test)

results = [x == y for x, y in zip(preds, test_labels)]

t1 = time.time()

print '\nAccuracy = ', sum(results) / float(len(results))

print '\nProgram took', t1 - t0, ' seconds to complete\n'

sys.exit()

# Structuring Data
print "\nStructuring Data ....\n"
time_data = defaultdict(Data)
test_data = defaultdict(Data)
time = 0;
keys = training.keys()
while keys:
    temp_data = Data(_time=time)
    ttest_data = Data(_time=time)
    for key in keys:
        if time < len(testing[key]):
            ttest_data.addData(key, testing[key].loc[[time]].as_matrix())
        if time < len(training[key][0]):
            temp_data.addData(key + '0', training[key][0].loc[[time]].as_matrix())
        if time < len(training[key][1]):
            temp_data.addData(key + '1', training[key][1].loc[[time]].as_matrix())
        if time >= len(training[key][0]) and time >= len(training[key][1]):
            keys.remove(key)
    time_data[time] = (temp_data)
    test_data[time] = (ttest_data)
    time += 1

# Training accuracy

# Times -> Classes -> Classifiers
SVMs = defaultdict(lambda : defaultdict(OneVsRestClassifier))

print "\nTraining Classifiers ....\n"
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
            clf = OneVsRestClassifier(SVC(kernel='rbf'))
            if word == 'later':
                plt.plot(pc_train.transform(dat.data[temp_cols].T.as_matrix()), color='red')
            clf.fit(pc_train.transform(dat.data[temp_cols].T.as_matrix()), Y)
            SVMs[tim][sign] = clf
#    if tim > 50: break

# Testing
num_correct = 0
num_total = 0

print "\nTesting ....\n"
# Iterating over test data
for word, data in testing.iteritems():
    sums = defaultdict(float, [(i, 0) for i in testing.keys()])
    for i, row in data.iterrows():

        # Iterating over all svms over the specified time
        for class_, svc in SVMs[i].iteritems():

            # Calculate distance from hyperplane

            pc_pred = pc_train.transform(row.reshape(1,-1))
            if word == 'later':
                plt.plot(pc_pred, color='blue')
            pred = svc.predict(pc_pred)
            if pred[0] == 1:
                y = svc.decision_function(pc_pred)
                sums[class_] += np.linalg.norm(y)

    max_dist = max(sums.values())
    clf = ''

    for key, value in sums.iteritems():
        if max_dist == value:
            print '\n\nreal_dist', sums[word]
            print 'max_dist', max_dist
            clf = key
            break
    print word, clf
    if word == clf:
        num_correct += 1
    num_total += 1

'''
# Iterating over test data
for time_, data_ in test_data.iteritems():
    sums = dict(zip(test_data.keys(), [0] * len(test_data)))

    # Iterating over all svms over the specified time
    for class_, svc in SVMs[time_]:

        # Calculate distance from hyperplane
        y = svc.decision_function(data_.data.as_matrix())
        w_norm = np.linalg.norm(svc.coef_)
        dist = y / w_norm

        sums[class_] += dist
'''


print 'num correct', num_correct
print 'num total', num_total
print 'Total Accuracy', float(num_correct) / num_total
plt.show()
