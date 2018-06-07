from numpy import *
from loadData import *

def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords); p1Num = ones(numWords)      #change to ones() 
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)          #change to log()
    p0Vect = log(p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #print vec2Classify
    # [0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
    
    #print p0Vec
    """[-2.56494936 -2.56494936 -2.56494936 -3.25809654 -3.25809654 -2.56494936
     -2.56494936 -2.56494936 -3.25809654 -2.56494936 -2.56494936 -2.56494936
     -2.56494936 -3.25809654 -3.25809654 -2.15948425 -3.25809654 -3.25809654
     -2.56494936 -3.25809654 -2.56494936 -2.56494936 -3.25809654 -2.56494936
     -2.56494936 -2.56494936 -3.25809654 -2.56494936 -3.25809654 -2.56494936
     -2.56494936 -1.87180218]"""
    
    #print p1Vec
    """[-3.04452244 -3.04452244 -3.04452244 -2.35137526 -2.35137526 -3.04452244
     -3.04452244 -3.04452244 -2.35137526 -2.35137526 -3.04452244 -3.04452244
     -3.04452244 -2.35137526 -2.35137526 -2.35137526 -2.35137526 -2.35137526
     -3.04452244 -1.94591015 -3.04452244 -2.35137526 -2.35137526 -3.04452244
     -1.94591015 -3.04452244 -1.65822808 -3.04452244 -2.35137526 -3.04452244
     -3.04452244 -3.04452244]""" 
    
    #print vec2Classify * p1Vec
    """
    [-0.         -3.04452244 -0.         -0.         -0.         -0.
     -0.         -0.         -0.         -0.         -0.         -3.04452244
     -0.         -0.         -0.         -0.         -0.         -0.
     -0.         -0.         -0.         -0.         -0.         -0.
     -0.         -0.         -0.         -0.         -0.         -0.
     -0.         -3.04452244]
    """
    
    #print sum(vec2Classify * p1Vec)
    # -9.13356731317
    
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    
    if p1 > p0:
        return 1
    else: 
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    
    """
    print array(trainMat)
    [[0 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 1 1 0 0 0 0 0 0 1]
     [0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 1 0 0 0 0 0 1 1 0 1 0 1 0 1 0 0 0]
     [1 1 0 0 0 1 0 1 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1]
     [0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 1 1 0 0 0 1 0 0 0 1 1 1]
     [0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 1 0 0 1 0 0 0 0 1 0 1 0 0 0 0 0]]
    """
    
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    
    """
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb)
    """
    
#testingNB()    
#['love', 'my', 'dalmation'] classified as:  0
#['stupid', 'garbage'] classified as:  1