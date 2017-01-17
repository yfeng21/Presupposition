from sklearn.feature_extraction.text import CountVectorizer
import cPickle
import os
import numpy as np
from sklearn.linear_model import LogisticRegression,SGDClassifier
from scipy import sparse
from sklearn.svm import LinearSVC

def pickleLoader(pklFile):
    try:
        while True:
            yield cPickle.load(pklFile)
    except EOFError:
        pass

def make_unigram_matrix(pklFile,pklFile2):#,out_X,out_y):
    corpus=[]
    y=[] #list of labels
    pos_feature=[]
    duplicates=[]
    with open(pklFile) as f:
        for line in pickleLoader(f):
            if line[0]=='too':
                print line
                y.append(1)
                context = " ".join(line[1])
                tags = " ".join(line[2])
                corpus.append(context) #context
                pos_feature.append(tags)
                if len(line[1]) != len(set(line[1])):
                    duplicates.append([1])
                else:
                    duplicates.append([0])

    # with open(pklFile2) as f:
    #     for line in pickleLoader(f):
    #         print line
    #         y.append(0)  # label
    #         context = " ".join(line[1])
    #         tags = " ".join(line[2])
    #         corpus.append(context)  # context
    #         pos_feature.append(tags)
    #         if len(line[1]) != len(set(line[1])):
    #             duplicates.append([1])
    #         else:
    #             duplicates.append([0])
    # duplicates=np.asarray(duplicates)
    return corpus, y,pos_feature,duplicates

def read(pklFile1):
    f=open(pklFile1, 'rb')
    # X = cPickle.load(f)
    Y = cPickle.load(f)
    f.close()
    return Y




if __name__ == '__main__':
    print "building....."
    # subdir = "/scratch/data/Yulan/Adverbial/presup-src/src/presup_giga_also"
    subdir='E:\Summer\presup-src\src\presup_ptb'
    corpus, y,pos_feature,duplicates = make_unigram_matrix(os.path.join(subdir, "train/positive_data.pkl"),
                                    os.path.join(subdir, "train/negative_data.pkl"))
    # corpus_test, y_test,pos_feature_test,duplicates_test = make_unigram_matrix(os.path.join(subdir, "test/positive_data.pkl"),
    #                                           os.path.join(subdir, "test/negative_data.pkl"))
    count_vect = CountVectorizer()
    count_bigram= CountVectorizer(ngram_range=(1,2))
    X_corpus = count_vect.fit_transform(corpus)
    print count_vect.get_feature_names()
    X_tag = count_bigram.fit_transform(pos_feature)
    print count_bigram.get_feature_names()
    X=sparse.hstack((X_corpus,X_tag),format="csr").A #np.concatenate((X_corpus, X_tag), axis=1)
    print X
    raw_input("-----")
    # print duplicates.shape
    # X=sparse.hstack([X,duplicates]) #np.append(X,duplicates,axis=1)


    X_corpus_test = count_vect.transform(corpus_test)
    X_tag_test=count_bigram.transform(pos_feature_test)
    X_test=sparse.hstack([X_corpus_test,X_tag_test],format="csr")
    # X_test=sparse.hstack((duplicates_test,X_test))


    clf=LogisticRegression()
    clf=clf.fit(X, y)
    # predicted= clf.predict(X_test)
    # predicted = map(int, predicted)
    result=clf.score(X_test,y_test)
    print result


