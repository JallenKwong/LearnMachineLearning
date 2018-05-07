import trees
import treePlotter

myTree = treePlotter.retrieveTree(0)

trees.storeTree(myTree, 'classifierStorage.txt')
print trees.grabTree('classifierStorage.txt')