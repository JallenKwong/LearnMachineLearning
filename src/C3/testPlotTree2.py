# -*- coding: utf-8 -*- 

import treePlotter

myTree = treePlotter.retrieveTree(0)
print myTree

#开始绘制决策树
treePlotter.createPlot(myTree)