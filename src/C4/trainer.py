from numpy  import *
from loadData import *

listOfPosts, listClasses = loadDataSet()

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
#     print numTrainDocs 6
    
    numWords = len(trainMatrix[0])
#    print numWords 32 non-repeat word
    
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    
    p0Num = zeros(numWords); p1Num = zeros(numWords)
#     print p0Num
#     print p1Num
    
    #p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones() 
    #p0Denom = 2.0; p1Denom = 2.0
    p0Denom = 0.0; p1Denom = 0.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
#     p1Vect = log(p1Num/p1Denom)
#     p0Vect = log(p0Num/p0Denom)

#   print p0Num
#   print p1Num

    """
    ['cute', 'love', 'help', 'garbage', 'quit', 'I', 'problems',
    'is', 'park', 'stop', 'flea', 'dalmation', 'licks', 'food',
     'not', 'him', 'buying', 'posting', 'has', 'worthless', 
     'ate', 'to', 'maybe', 'please', 'dog', 'how', 'stupid', 
     'so', 'take', 'mr', 'steak', 'my']

    [1. 1. 1. 0. 0. 1. 1. 1. 0. 1. 1. 1. 1. 0. 0. 2. 0. 0. 1. 0. 1. 1. 0. 1.
     1. 1. 0. 1. 0. 1. 1. 3.]
    [0. 0. 0. 1. 1. 0. 0. 0. 1. 1. 0. 0. 0. 1. 1. 1. 1. 1. 0. 2. 0. 1. 1. 0.
     2. 0. 3. 0. 1. 0. 0. 0.]
    """

    p1Vect = p1Num/p1Denom          #change to log()
    p0Vect = p0Num/p0Denom          #change to log()
    return p0Vect, p1Vect, pAbusive

myVocabList = createVocabList(listOfPosts)
print myVocabList

trainMat = []
for post in listOfPosts:
    trainMat.append(setOfWords2Vec(myVocabList, post))
    
# print trainMat
"""[[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1], 
[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0], 
[1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], 
[0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1], 
[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]]"""

p0V, p1V, pAb = trainNB0(trainMat, listClasses)

print p0V
print p1V
print pAb

"""
[0.04166667 0.04166667 0.04166667 0.         0.         0.04166667
 0.04166667 0.04166667 0.         0.04166667 0.04166667 0.04166667
 0.04166667 0.         0.         0.08333333 0.         0.
 0.04166667 0.         0.04166667 0.04166667 0.         0.04166667
 0.04166667 0.04166667 0.         0.04166667 0.         0.04166667
 0.04166667 0.125     ]
[0.         0.         0.         0.05263158 0.05263158 0.
 0.         0.         0.05263158 0.05263158 0.         0.
 0.         0.05263158 0.05263158 0.05263158 0.05263158 0.05263158
 0.         0.10526316 0.         0.05263158 0.05263158 0.
 0.10526316 0.         0.15789474 0.         0.05263158 0.
 0.         0.        ]
0.5
"""

