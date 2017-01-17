'''
Created on Feb 18, 2016

@author: jcheung
'''

import numpy as np
from resources.ptb2 import PTB2Corpus
from utils.paths import Paths
from models.presup_triggers import PresupExtractor
import time
import pickle
from numpy.lib.function_base import corrcoef
from nltk import bigrams
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import HashingVectorizer
import time

def extract_ptb2(paths, step):
    corpus = PTB2Corpus(paths.ptb2)
    subdir = './presup'
    
    # extract set
    extractor = PresupExtractor({'debug': False, 'records_out': True})
    extractor.run(corpus,"train")
    
    # extract comparables
    
    # extract features


def pickleLoader(pklFile):
    try:
        while True:
            yield pickle.load(pklFile)
    except EOFError:
        pass

def create_bigram_matrix(pklFile):
    '''turn into a matrix of feature vectors'''
    '''FeatureHasher Method'''
    # creating matrix X
    bigram_feature_list = []  # a list of lists of bigram features
    label_list = []
    with open(pklFile) as f:
        # print tokens
        for line in pickleLoader(f):
            label = line[0]
            context = line[1]  # list of tokens
            bigram_feature_list.append(list(bigrams(context)))
            label_list.append(label)
            print list(bigrams(context))

    # print label_list
    fh = FeatureHasher(input_type='string')
    X = fh.transform(((' '.join(x) for x in sample) for sample in bigram_feature_list))

    # vector y
    nrows = len(label_list)
    y = np.zeros(nrows)
    for i in range(nrows):
        if label_list[i] is not 'none':
            y[i] = 1

    # print X.shape
    print y.shape
    print y

if __name__ == '__main__':
    print 'Presupposition extractor'
    t0 = time.time();
    paths = Paths()

    #create_bigram_matrix("extract_data.pkl")

    # step 1: extract initial set
    extract_ptb2(paths, 'init')
    
    # step 2: extract comparison set
    print('Execution time is %f s' % (time.time() - t0))

    
