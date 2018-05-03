import C2.kNN as kNN

datingDataMat, datingLabels = kNN.file2matrix('..\\datingTestSet2.txt')

print datingDataMat

print datingLabels

print '---'

import matplotlib
import matplotlib.pyplot as plt
from  numpy import *

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
ax.scatter(datingDataMat[:,1],datingDataMat[:,2] \
           , 15.0 * array(datingLabels), 15.0 * array(datingLabels))
plt.show() 