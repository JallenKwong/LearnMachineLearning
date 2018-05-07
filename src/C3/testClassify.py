import treePlotter
import trees

dataSet, labels = trees.createDataSet()
myTree = treePlotter.retrieveTree(0)

print myTree


print trees.classify(myTree, labels, [1, 0])
#no

print trees.classify(myTree, labels, [1, 1])
#yes


