# coding=UTF-8
import os
import random
import numpy
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
# from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import sys

# cellline = sys.argv[1]
cellline = 'HeLa-S3'
fin = open('./data/' + cellline + 'train.csv', 'r')
lines = fin.readlines()
print(len(lines), len(lines[0]))
row = len(lines) - 1  # 行数
col = len(lines[1].split(',')) - 18  # 列数（文件前17列是数据，后面是细胞系中增强子和启动子的结果）
print('row,col:', row, col)

arrays = numpy.zeros((row, col))
print('arrays:', len(arrays))
labels = numpy.zeros(row)

for i in range(row):
    data = lines[i + 1].strip().split(',')  # 第一行为表头，不考虑
    arrays[i] = data[18:]
    labels[i] = data[7]

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
f1 = []
auc = []
aupr = []

print('start for')
for train, test in cv.split(arrays, labels):
    estimator = GradientBoostingClassifier(n_estimators=4000, learning_rate=0.001, max_depth=25, max_features='log2',
                                           random_state=0)
    estimator.fit(arrays[train, :], labels[train])  # 速度慢
    sort_index = [i[0] for i in sorted(enumerate(estimator.feature_importances_), key=lambda x: x[1])]
    array = arrays[:, sort_index[-16:]]
    estimator.fit(array[train, :], labels[train])  # 速度慢
    y_pred = estimator.predict(array[test, :])
    y_prob = estimator.predict_proba(array[test, :])
    f1.append(metrics.f1_score(labels[test], y_pred))
    fpr, tpr, th = metrics.roc_curve(labels[test], y_prob[:, 1], pos_label=1)
    auc.append(metrics.auc(fpr, tpr))
    aupr.append(metrics.average_precision_score(labels[test], y_prob[:, 1]))
print("end for")

print('TargetFinder,feature selection:')
print(numpy.mean(f1), numpy.std(f1))
print(numpy.mean(auc), numpy.std(auc))
print(numpy.mean(aupr), numpy.std(aupr))
