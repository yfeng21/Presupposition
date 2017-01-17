'''
Created on Jan 19, 2016

@author: jcheung
'''

from resources.ptb2 import PTB2Corpus
from utils.paths import Paths
from models.classification import DetExtractor, ClassifierRecords, \
    DetClassifier, DetFeatureMatrices
import time
import csv, os
import numpy as np
import cPickle

def extract_ptb2(paths):
    corpus = PTB2Corpus(paths.ptb2)
    subdir = './features'
    
    # extract features
    extractor = DetExtractor({'debug': False})
    extractor.run(corpus)
    extractor.save(subdir)
    
def baseline():
    subdir = './features'
    dataset = ClassifierRecords(subdir)
    
    # count distribution of each tag in evaluation set
    counts = {'the':0, 'a':0, 'none':0}
    for r in dataset.train:
        counts[r.label] += 1
    print counts
    
    # majority baseline:
    denom = counts['the'] + counts['a'] + counts['none']
    numer = max(counts['the'], counts['a'], counts['none'])
    print 'Majority baseline: %d/%d = %.4f%%' % (numer, denom, (float(numer) /denom * 100))

def build_matrices():
    subdir = './features'
    output_dir = './features/h5f'
    dataset = ClassifierRecords(subdir)
    dataset.create_feature_matrices(output_dir)
    
def run_classifier():
    subdir_h5f = './features/h5f'
    matrices = DetFeatureMatrices(subdir_h5f)
    
    print matrices.train_X.shape, matrices.train_y.shape
    print matrices.dev_X.shape, matrices.dev_y.shape
    print matrices.test_X.shape, matrices.test_y.shape
    model = DetClassifier({'classifier': 'logistic_regression'})
    #model = DetClassifier({'classifier': 'svm'})
    st = time.time()
    print 'Training'
    model.train(matrices.train_X, matrices.train_y)
    et = time.time()
    print 'Training took %.2fs' % (et - st)
    
    # evaluate accuracy on training set, dev set, test set
    
    answers = model.predict(matrices.train_X)
    eval_display(model, answers, matrices.train_y, 'Training set')
    
    answers = model.predict(matrices.dev_X)
    eval_display(model, answers, matrices.dev_y, 'Dev set')
    
    answers = model.predict(matrices.test_X)
    eval_display(model, answers, matrices.test_y, 'Test set')
    
    model.prf1(answers, matrices.test_y)
    # baseline prf1
    model.prf1(np.ones(len(matrices.test_y)), matrices.test_y)
    
    ne_eval(model, matrices.test_X, answers, matrices.test_y)
    
def ne_eval(model, X, answers, y):
    # find feature for ne only
    feats = {}
    subdir_h5f = './features/h5f'
    with open(os.path.join(subdir_h5f, 'features.csv'), 'rb') as csvf:
        r = csv.reader(csvf)
        for row in r:
            feats[row[0]] = int(row[1])
    ne_feat = feats['ne'] # this is the feature for ne=no
    
    print ne_feat
    # masks
    col = X[:, ne_feat].todense()
    col = np.array(col)
    col = col.reshape(len(col))

    print col.shape
    non_ne_mask = (col == 0)
    ne_mask = (col == 1)
    
    # accuracy for named entities only
    eval_display(model, answers[ne_mask], y[ne_mask], 'Test set NE')
    # accuracy for non-named entities only
    eval_display(model, answers[non_ne_mask], y[non_ne_mask], 'Test set non-NE')
    
    # baseline
    eval_display(model, np.ones(len(y[ne_mask])), y[ne_mask], 'Baseline Test set NE')
    # accuracy for non-named entities only
    eval_display(model, np.ones(len(y[non_ne_mask])), y[non_ne_mask], 'Baseline Test set non-NE')

    
    # save outputs
    with open('classifier_output.pkl', 'wb') as fh:
        triple = (answers, y, col)
        cPickle.dump(triple, fh) 

def eval_display(model, answers, y, msg):
    corr, tot, acc = model.eval(answers, y)
    print 
    print '%s accuracy' % msg
    print 'Correct:  %d' % corr
    print 'Total:    %d' % tot
    print 'Accuracy: %.2f%%' % (acc * 100)
    print
    
def extract_corpus_for_lstm():
    corpus = PTB2Corpus(paths.ptb2)
    subdir = './nodet'
    
    # extract features
    extractor = DetExtractor({'debug': False})
    extractor.extract_nodet(corpus, subdir)
     
if __name__ == '__main__':
    print 'Determiner prediction'
    paths = Paths()
    
    # feature extraction
    #extract_ptb2(paths)
    
    # majority baseline method -- mostly to test loading of the features
    #baseline()
    
    # classifier-based method
    #build_matrices()
    run_classifier()
    # extract_corpus_for_lstm()
    
    