from sklearn.feature_extraction.text import CountVectorizer
import cPickle
import numpy as np
from sklearn.linear_model import LogisticRegression,SGDClassifier
import os
def pickleLoader(pklFile):
    try:
        while True:
            yield cPickle.load(pklFile)
    except EOFError:
        pass

def make_unigram_matrix(pklFile,pklFile2):#,out_X,out_y):
    corpus=[]
    y=[] #list of labels
    with open(pklFile) as f:
        for line in pickleLoader(f):
            # y.append(line[0]) #label
            context = " ".join(line[1])
            print (line[0],context)
            # corpus.append(context) #context
    # x = len(y)
    # print "positive:", x
    with open(pklFile2) as f:
        for line in pickleLoader(f):
            context = " ".join(line[1])
            print (line[0], context)
            # y.append(line[0])  # label
            # context = " ".join(line[1])
            # corpus.append(context)  # context
    # print "negative:",len(y)-x
    # print "in total:",len(y)#corpus, y
    return y

def read(pklFile1):
    f=open(pklFile1, 'rb')
    # X = cPickle.load(f)
    Y = cPickle.load(f)
    f.close()
    return Y

def reader(pklFile):
    with open(pklFile) as f:
        for line in pickleLoader(f):
            # y.append(line[0]) #label
            context1 = " ".join(line[1])
            context2 = " ".join(line[2])
            print (line[0],context1,context2)
            raw_input("---")


if __name__ == '__main__':
    # print "building....."

    # x=make_unigram_matrix('presup_ptb/test/positive_data.pkl','presup_ptb/test/negative_data.pkl')
    # y=reader("presup_ptb/test/positive_data.pkl")
    x=reader("presup_ptb/test/positive_data_split.pkl")
    # print x
    # print len(x)


