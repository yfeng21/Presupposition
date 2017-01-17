'''
Created on Mar 2, 2016

@author: jcheung
'''

from resources.bnc import BNCCorpus

from utils.paths import Paths
from models.classification import DetExtractor, ClassifierRecords, \
    DetClassifier, DetFeatureMatrices
import time
import sys

def extract_determiners(paths):
    corpus = BNCCorpus(paths.bnc)
    subdir = './features_bnc/bnc'
    
    my_iter = corpus.doc_iter()
    st = time.time()
    doc = my_iter.next()
    et = time.time()
    print '%s %d sentences in %.1f secs' % (doc.num, len(doc.sents), (et - st))
    # extract features
    extractor = DetExtractor({'debug': False})
    extractor.run_incremental(corpus, subdir)
    
def extract_corpus_for_lstm(paths):
    corpus = BNCCorpus(paths.bnc)
    subdir = './features_bnc/nodet'
    
    # extract features
    extractor = DetExtractor({'debug': False})
    extractor.extract_nodet(corpus, subdir)
    
if __name__ == '__main__':
    print 'Determiner prediction: BNC'
    paths = Paths()
    
    task = sys.argv[1]
    
    # feature extraction
    if task == 'extract_determiners':
        extract_determiners(paths)
    elif task == 'extract_corpus_for_lstm':
        extract_corpus_for_lstm(paths)