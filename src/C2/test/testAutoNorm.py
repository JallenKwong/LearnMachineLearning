import C2.kNN as kNN


datingDataMat, datingLabels = kNN.file2matrix('..\\datingTestSet2.txt')
normMat, ranges, minVals = kNN.autoNorm(datingDataMat)

print normMat

print ranges

print minVals