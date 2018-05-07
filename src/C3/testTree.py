# -*- coding: utf-8 -*- 
import trees

dataSet, labels = trees.createDataSet()

print dataSet
print labels


#计算熵
print trees.calcShannonEnt(dataSet)#0.970950594455

dataSet[0][-1] = 'maybe'

print dataSet
print trees.calcShannonEnt(dataSet)#1.37095059445
#熵越大，则混合的数据越多

#还原
dataSet[0][-1] = 'yes'
print dataSet
#[[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
#划分数据集

#当第0列，值为0 的实例
print trees.splitDataSet(dataSet, 0, 0)
#[[1, 'no'], [1, 'no']]


#当第0列，值为1 的实例
print trees.splitDataSet(dataSet, 0, 1)
#[[1, 'yes'], [1, 'yes'], [0, 'no']]

print trees.chooseBestFeatureToSplit(dataSet)
#0

print "---createTree---"

print trees.createTree(dataSet, labels)

"""
---createTree---
classList: 
['yes', 'yes', 'no', 'no', 'no']
baseEntropy: 0.970950594455
value: 0
value: 1
newEntropy: 0.550977500433
value: 0
value: 1
newEntropy: 0.8
---
classList: 
['no', 'no']
---
classList: 
['yes', 'yes', 'no']
baseEntropy: 0.918295834054
value: 0
value: 1
newEntropy: 0.0
---
classList: 
['no']
---
classList: 
['yes', 'yes']
{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
"""

