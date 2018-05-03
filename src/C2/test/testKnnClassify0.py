import C2.kNN as kNN

group,labels = kNN.createDataSet()

print 'group:'
print group

print 'labels:'
print labels

print '---'

print 'test classify0 result:'

result = kNN.classify0([0,0], group, labels, 3)
print result #B

print '---'
