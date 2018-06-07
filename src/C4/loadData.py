from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec
                 
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my Vocabulary!" % word
    return returnVec

listOfPosts, listClasses = loadDataSet()
#print listOfPosts
#print listClasses

"""[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'], 
 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'], 
 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]"""

#[0, 1, 0, 1, 0, 1]


myVocabList = createVocabList(listOfPosts)
#print myVocabList
"""['cute', 'love', 'help', 'garbage', 
'quit', 'I', 'problems', 'is', 'park', 
'stop', 'flea', 'dalmation', 'licks', 
'food', 'not', 'him', 'buying', 'posting',
'has', 'worthless', 'ate', 'to', 'maybe', 
'please', 'dog', 'how', 'stupid', 'so', 
'take', 'mr', 'steak', 'my']"""

#print listOfPosts[0]
#['my', 'dog', 'has', 'flea', 'problems', 'help', 'please']

#print setOfWords2Vec(myVocabList, listOfPosts[0])
"""[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
1, 0, 0, 0, 0, 0, 0, 1]"""

