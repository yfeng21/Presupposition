'''
Created on Mar 18, 2016

@author: jcheung
'''
import cPickle
import numpy as np


def eval_all(label, test, gold, ne):
    print label
    print np.sum(test == gold) / float(len(gold))
    print 'NE'
    print np.sum(test[ne == 1] == gold[ne == 1]) / float(len(gold[ne == 1]))
    print 'non-NE'
    print np.sum(test[ne == 0] == gold[ne == 0]) / float(len(gold[ne == 0]))
    # prf1(test, gold)
    confusion_matrix(test, gold)
    
def prf1(test, gold):
        '''
        test: a numpy array of labels
        gold: a numpy array of labels with the same length as test
        '''
        print '---'
        prf1_dict = {}
        # return P, R, F1 for each of the three classes
        for val in xrange(3):
            # val is 0 (a), 1 (none), or 2 (the)
            print val
            # precision: restrict gold to those cases where system predicts val
            pmask = (test == val)
            arr = gold[pmask]
            pnumer = np.sum(arr == val)
            pdenom = len(arr)
            p = pnumer / float(pdenom) 
            print 'P: %d/%d = %.4f' % (pnumer, pdenom, p)
            # recall: restrict test to those cases where gold label is val
            rmask = (gold == val)
            arr = test[rmask]
            rnumer = np.sum(arr == val)
            rdenom = len(arr)
            r = rnumer / float(rdenom)
            print 'R: %d/%d = %.4f' % (rnumer, rdenom, r)
            f1 = 2 * p * r / (p + r)
            print 'F1: %.4f' % f1
            print
            prf1_dict[val] = (p, r, f1)
        print '---'
        return prf1_dict
    
def confusion_matrix(test, gold):
    conf = np.zeros((3, 3))
    # dimension 1: actual
    # dimension 2: predicted
    
    for i in xrange(len(gold)):
        conf[gold[i], test[i]] += 1
        
    print conf
    
    
def find_cases(vec1, vec2, correct, ne):
    # TODO: find cases where one system is right and the other one is wrong;
    # tie it back to the original data set
    right1 = (vec1 == correct)
    right2 = (vec2 == correct)
    '''
    # want a case where right2 is correct, right1 is not, and ne is false
    for i in xrange(len(correct)):
        if right2[i] and not right1[i] and not ne[i]:
            print i, vec1[i], vec2[i], correct[i], ne[i]
        if i > 1000: break
    ''' 
    # want a case where right2 does incorrectly
    for i in xrange(len(correct)):
        if not right2[i]:
            print i, vec1[i], vec2[i], correct[i], ne[i]
        if i > 1000: break
            
            
def load_pickle(f):
    with open(f, 'rb') as fh:
        obj = cPickle.load(fh)
    predictions, correct, ne = obj
    
    print predictions.shape
    print correct.shape
    print ne.shape
    return predictions, correct, ne

def load_lstm_predictions():
    path = './LSTM_rand_POS.pkl'
    f = open(path, 'rb')
    LSTM_rand_POS = cPickle.load(f)
    f.close()
    print LSTM_rand_POS
    
    path = './LSTM_w2v_POS.pkl'
    f = open(path, 'rb')
    LSTM_w2v_POS = cPickle.load(f)
    f.close()
    print LSTM_w2v_POS
    return swap(LSTM_rand_POS), swap(LSTM_w2v_POS) 

def load_all_lstm_predictions(d):
    vecs = []
    for i in xrange(1, 5):
        path = './%s/exp%d.pkl' % (d, i)
        f = open(path, 'rb')
        v = cPickle.load(f)
        f.close()
        vecs.append(swap(v))
    return vecs

def swap(v):
    v[v == 1] = 4
    v[v == 0] = 5
    v[v == 5] = 1
    v[v == 4] = 0
    # print v
    return v
    
if __name__ == '__main__':
    f = 'classifier_output.pkl'
    logreg, correct, ne = load_pickle(f)
    #rand_pos, w2v_pos = load_lstm_predictions()
    # find_cases(logreg, rand_pos, w2v_pos, correct, ne)
    #eval_all('logreg', logreg, correct, ne)
    #vecs = load_all_lstm_predictions()
    #eval_all('rand-pos', vecs[0], correct, ne)
    #eval_all('rand+pos', vecs[1], correct, ne)
    #eval_all('w2v-pos', vecs[2], correct, ne)
    #eval_all('glove-pos', vecs[3], correct, ne)
    #eval_all('w2v+pos', vecs[4], correct, ne)
    #eval_all('glove+pos', vecs[5], correct, ne)
    
    
    # find cases where one is right, the other is wrong
    # compare rand+pos to glove+pos
    #find_cases(vecs[1], vecs[5], correct, ne)
    
    
    vecs = load_all_lstm_predictions('alternative_models')
    eval_all('1', vecs[0], correct, ne)
    eval_all('2', vecs[1], correct, ne)
    eval_all('3', vecs[2], correct, ne)
    eval_all('4', vecs[3], correct, ne)