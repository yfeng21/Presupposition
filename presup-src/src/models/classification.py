'''
Created on Jan 19, 2016

@author: jcheung
'''

import numpy as np
import csv, os, glob
import tables as tb
from scipy.sparse import lil_matrix, csr_matrix
from utils.utils import tbOpen, load_h5f_csr, write_h5f_csr, write_h5f_array, load_h5f_array
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import codecs
from nltk.corpus import wordnet as wn
from resources.count_or_mass import countability

pos_stoplist = set(['DT', ',', '.', '$', "''", '#', '-LRB-', '-RRB-', ':', '``'])

class DetExtractor(object):
    '''
    Extract features
    '''

    def __init__(self, params):
        '''
        Constructor
        '''
        self.debug = params.get('debug', False)
        self.records_dict = None
        
    def run(self, corpus):
        ndocs = 0
        self.records_dict = {} # doc -> list of records
        # corpus might be something of type PTB2Corpus, for example 
        for doc in corpus.doc_iter('all'):
            self.records_dict[doc.num] = self.extract_records(doc)
            ndocs += 1
            if self.debug and ndocs >= 5:
                break
    
    def run_incremental(self, corpus, outdir):
        ndocs = 0
        # self.records_dict = {} # doc -> list of records
        # corpus might be something of type PTB2Corpus, for example 
        for doc in corpus.doc_iter('all'):
            records = self.extract_records(doc)
            self.save_incremental(outdir, doc.num, records)
            ndocs += 1
            if self.debug and ndocs >= 5:
                break
        
    def save(self, subdir):
        '''
        Aggregate features and values, write to disk in the folder subdir.
        '''
        # first row, header
        first_r = self.records_dict.itervalues().next()[0]
        features = first_r.fdict.keys()
        features.sort()
        first_row = ['loc', 'label'] + features
        
        # possible values
        poss = {} # feature name -> set of values
        poss['label'] = set()
        for f in features:
            poss[f] = set()
            
        for docn, records in self.records_dict.iteritems():
            with codecs.open(os.path.join(subdir, '%s.csv' % docn), 'wb', encoding='utf-8', errors='ignore') as csvf:
                w = csv.writer(csvf)
                w.writerow(first_row)
                for r in records:
                    poss['label'].add(r.label)
                    row = [r.loc, r.label]
                    for f in features:
                        val = r.fdict[f]
                        row.append(val)
                        poss[f].add(val)
                    w.writerow(row)
        
        # write possible values
        with open(os.path.join(subdir, 'feature-vals.csv'), 'wb') as csvf:
            w = csv.writer(csvf)
            row = ['label'] + sorted(poss['label'])
            w.writerow(row)
            for f in features:
                row = [f] + sorted(poss[f])
                w.writerow(row)
    
    def save_incremental(self, subdir, docn, records):
        '''
        Aggregate features and values, write to disk in the folder subdir.
        '''
        # first row, header
        if len(records) < 1: return
        features = records[0].fdict.keys()
        features.sort()
        first_row = ['loc', 'label'] + features
        first_row_s = ','.join(map(str,first_row))
        
        # possible values
        poss = {} # feature name -> set of values
        poss['label'] = set()
        for f in features:
            poss[f] = set()
            
        
        with codecs.open(os.path.join(subdir, '%s.csv' % docn), 'w', encoding='utf-8', errors='ignore') as csvf:
            #w = csv.writer(csvf)
            #w.writerow(first_row)
            csvf.write(first_row_s + '\n')
            
            for r in records:
                poss['label'].add(r.label)
                row = ['"%s"' % str(r.loc), r.label]
                for f in features:
                    val = r.fdict[f]
                    row.append(val)
                    poss[f].add(val)
                #w.writerow(row)
                csvf.write(','.join(row) + '\n')
    
        # write possible values
        with codecs.open(os.path.join(subdir, 'feature-vals.csv'), 'w', encoding='utf-8', errors='ignore') as csvf:
            #w = csv.writer(csvf)
            row = ['label'] + sorted(poss['label'])
            #w.writerow(row)
            csvf.write(','.join(row) + '\n')
            for f in features:
                row = [f] + sorted(poss[f])
                #w.writerow(row)
                csvf.write(','.join(row) + '\n')
                
    def extract_nodet(self, corpus, subdir):
        ndocs = 0
        for doc in corpus.doc_iter('all'):
            outf = os.path.join(subdir, '%s.txt' % doc.num)
            self.extract_nodet_doc(doc, outf)
            ndocs += 1
            if self.debug and ndocs >= 5:
                break
    def extract_nodet_doc(self, doc, outf):
        with codecs.open(outf, 'w', encoding='utf-8', errors='ignore') as out:
            for sentn, sent in doc.sents.iteritems():
                self.extract_nodet_sent(sent, out)
    
    def extract_nodet_sent(self, sent, out):
        explored = set()
        for depid, node in sent.parse.iternodes():
            par = node.governor
            rel = node.reltype
            chi = node.dependent
            if chi in explored: continue
            if rel.startswith('conj'): continue # in conjoined NPs, use the other incoming edge
            explored.add(chi) # prevent duplicates
            
            chi_token = sent.at(chi)
            out.write('%s/%s' % (chi_token.word, chi_token.pos))
            if chi_token.pos == 'DT' and chi_token.word.lower() in ['a', 'an', 'the']:
                out.write('-')
            elif chi_token.pos.startswith('N') and rel not in ['compound']:
                out.write('*')
                # extract label
                children = sent.parse.dependent_nodes_of(node.dependent)
                label = 'none'
                for child in children:
                    #print '  ', child
                    reltype = child.reltype
                    if reltype == 'det':
                        # answer
                        if sent.at(child.dependent).word.lower() == 'the':
                            label = 'the'
                        elif sent.at(child.dependent).word.lower() in ['a', 'an']:
                            label = 'a'
                out.write(label)
            out.write(' ')
        out.write('\n')
        
    def extract_records(self, doc):
        '''
        Extract records from a Document.
        Returns a list of Records
        '''
        records = []
        print doc.num
        for sentn, sent in doc.sents.iteritems():
            #print sentn, sent.text()
            records.extend(self.extract_from_sent(doc.num, sentn, sent))
        return records

        
    def extract_from_sent(self, docn, sentn, sent):
        '''
        Extract records from a Sentence.
        '''
        records = []
        explored = set()
        for depid, node in sent.parse.iternodes():
            par = node.governor
            rel = node.reltype
            chi = node.dependent
            if chi in explored: continue
            if rel.startswith('conj'): continue # in conjoined NPs, use the other incoming edge
            explored.add(chi) # prevent duplicates
            
            chi_token = sent.at(chi)
            #print chi_token.word, chi_token.pos, rel,
            if chi_token.pos.startswith('N') and rel not in ['compound']:
                #print '*'
                r = self.extract_record(docn, sentn, chi, sent, chi_token, node)
                # this is a record!
                records.append(r) 
            else:
                pass
                #print
        return records
                
    def extract_record(self, docn, sentn, wordn, sent, hnoun, node):
        '''
        Extract from the sentence at the specified token hnoun, using the node (i.e., edge) 
        whose child is the location in the sentence 
        '''
        if self.debug:
            print '----------------------------------------'
            print 'CASE: ',
            print sent.text()
            print hnoun.word,
            print node
        
        r = Record()
        
        r.loc = (docn, sentn, wordn)
        # feature extraction, label extraction
        
        # head noun
        lemma = hnoun.lemma.lower()
        r.fdict['hn'] = lemma
        # number and NE-hood
        r.fdict['hnpos'] = hnoun.pos
        r.fdict['nehood'] = hnoun.ner 
        
        # lexname
        r.fdict['lexname'] = 'na'
        synsets = wn.synsets(lemma)
        if len(synsets) > 0:
            r.fdict['lexname'] = synsets[0].lexname
        
        # countability
        r.fdict['count'] = countability(lemma)
        
        # named entity
        r.fdict['ne'] = 'no'
        if hnoun.pos in ['NNP', 'NNPS']:
            r.fdict['ne'] = 'yes'
        
        # obj of prep
        r.fdict['pobj'] = 'no'
        r.fdict['pobjval'] = 'na'
        if node.reltype.startswith('nmod:'):
            r.fdict['pobj'] = 'yes'
            r.fdict['pobjval'] = node.reltype[5:]
        
        # check for expletive there
        r.fdict['exist'] = 'no'
        par_token = sent.at(node.governor)
        if par_token is not None:
            # get siblings
            sibs = sent.parse.dependent_nodes_of(node.governor)
            # see if grandparent is there
            for sib in sibs:
                sib_token = sent.at(sib.dependent)
                if sib_token is not None and sib_token.pos == 'EX':
                    r.fdict['exist'] = 'yes'
        
        # get children
        i = node.dependent
        children = sent.parse.dependent_nodes_of(i)
        
        
        r.label = 'none' # no determiners
        r.fdict['amod'] = 'no'
        r.fdict['pmod'] = 'no'
        r.fdict['pmodval'] = 'na'
        r.fdict['agrade'] = 'na'
        r.fdict['rmod'] = 'no'
        r.fdict['qmod'] = 'no'
        r.fdict['possmod'] = 'no'
        r.fdict['nummod'] = 'no'
        
        for child in children:
            #print '  ', child
            reltype = child.reltype
            if reltype == 'det':
                # answer
                if sent.at(child.dependent).word.lower() == 'the':
                    r.label = 'the'
                elif sent.at(child.dependent).word.lower() in ['a', 'an']:
                    r.label = 'a'
                else:
                    r.fdict['qmod'] = 'yes'
            
            # prep mod
            elif reltype.startswith('nmod:') and reltype != 'nmod:poss':
                r.fdict['pmod'] = 'yes'
                r.fdict['pmodval'] = reltype[5:]
            
            # poss mod
            elif reltype == 'nmod:poss':
                r.fdict['possmod'] = 'yes'
            
            # adj mod
            elif reltype == 'amod':
                # adj mod
                # adj grade
                r.fdict['amod'] = 'yes'
                r.fdict['agrade'] = sent.at(child.dependent).pos # JJ, JJR, or JJS
            
            elif reltype == 'neg':
                if sent.at(child.dependent).word == 'no':
                    r.fdict['qmod'] = 'yes'
                
            # rel clause mod
            elif reltype == 'acl:relcl':
                r.fdict['rmod'] = 'yes'
            
            # nummod
            elif reltype == 'nummod':
                r.fdict['nummod'] = 'yes'
        r.fdict['pos-3'] = 'none'
        r.fdict['pos-2'] = 'none'
        r.fdict['pos-1'] = 'none'
        r.fdict['pos+1'] = 'none'
        r.fdict['pos+2'] = 'none'
        r.fdict['pos+3'] = 'none'
        
        # POS +1/2/3 from head noun
        j, k = 1, 1 # j keeps track of feature position, k keeps track of token position
        while j < 4 and i + k <= len(sent.tokens):
            if sent.at(i + k).pos == 'DT' and sent.at(i + k).word.lower() in ['a', 'an', 'the']:
                token = '-'
            else:
                token = sent.at(i + k)
            r.fdict['pos+%d' % j] = token.pos
            j += 1
            k += 1
        
        # POS -1/2/3 from head noun
        j, k = 1, 1 # j keeps track of feature position, k keeps track of token position
        while j < 4 and i - k > 0:
            if sent.at(i + k).pos == 'DT' and sent.at(i + k).word.lower() in ['a', 'an', 'the']:
                token = '-'
            else:
                token = sent.at(i - k)
            r.fdict['pos-%d' % j] = token.pos
            j += 1
            k += 1
            
        # word +/- 1 from blank
        
        
        if self.debug:
            print r.fdict
            print r.label
            print r.loc
            print '----------------------------------------'
        return r
        
        
class Record:
    def __init__(self):
        self.fdict = {} # fname -> fval
        self.label = None
        self.loc = None # (docnum, sentnum, tokennum)
    def __str__(self):
        return '%s\t%s\t%s' % (self.loc, self.label, str(self.fdict))

class ClassifierRecords:
    def __init__(self, subdir):
        # load possible features
        self.feature_set = {} # feat_name -> vals
        with open(os.path.join(subdir, 'feature-vals.csv'), 'rb') as csvf:
            r = csv.reader(csvf)
            for row in r:
                self.feature_set[row[0]] = row[1:]
            
        # load each subset of the data
        self.dev = self.load_records(os.path.join(subdir, 'dev'))
        self.train = self.load_records(os.path.join(subdir, 'train'))
        self.test = self.load_records(os.path.join(subdir, 'test'))
        print 'Done loading'
        
    def load_records(self, subdir):
        records = []
        for fname in glob.glob(os.path.join(subdir, '*.csv')):
            with open(fname, 'rb') as csvf:
                r = csv.reader(csvf)
                header = r.next()
                assert header[0] == 'loc' and header[1] == 'label'
                
                for row in r:
                    record = Record()
                    record.loc = row[0]
                    record.label = row[1]
                    for f in xrange(2, len(row)):
                        record.fdict[header[f]] = row[f]
                    records.append(record)
        return records
    
    def create_feature_matrices(self, output_dir):
        min_threshold = 5
        
        # label: simply 0, 1, or 2
        binary_features = [f for f in self.feature_set if len(self.feature_set[f]) == 2]
        multival_features = [f for f in self.feature_set if len(self.feature_set[f]) != 2]
        multival_features.remove('label')
        #print binary_features
        #print multival_features
        
        # counts of value frequencies
        counts = {} # f -> val -> count
        
        # simple binary features: 0 or 1 --- don't threshold
        for r in self.train:
            # threshold for multival_features
            for f in multival_features:
                val = r.fdict[f]
                d = counts.setdefault(f, {})
                d.setdefault(val, 0)
                d[val] += 1
        
        #print counts['pos+1']
        #print len([x for x in counts['hn'] if counts['hn'][x] > min_threshold])
        
        # for each feature, map value name to index
        feat_map = {} # f -> val -> index
        for f in self.feature_set:
            if f == 'label': continue
            feat_map[f] = {}
            if f in binary_features:
                for val in sorted(self.feature_set[f]):
                    feat_map[f][val] = len(feat_map[f]) 
            else:
                # multival_features: 1 binary feature for each possible value 
                vals = [x for x in counts[f] if counts[f][x] > min_threshold]
                vals.sort()
                for val in vals:
                    feat_map[f][val] = len(feat_map[f])
                feat_map[f]['UNK'] = len(feat_map[f])
                
        # determine an ordering to features; write the feature offsets
        features = {}
        offset = 0
        for val in sorted(feat_map.keys()):
            features[val] = offset
            diff = len(feat_map[val])
            if diff == 2:
                diff = 1 # Mar 17 change: binary features only take up one feature
            offset += diff
        
        nfeatures = 0
        # write to output
        csv_write_rows(os.path.join(output_dir, 'features.csv'), sorted(features.items()))
        csv_write_rows(os.path.join(output_dir, 'labels.csv'), enumerate(['a', 'none', 'the']))
        for f in feat_map:
            diff = len(feat_map[f])
            if diff == 2:
                diff = 1 # Mar 17 change: binary features only take up one feature
            nfeatures += diff
            csv_write_rows(os.path.join(output_dir, '%s.csv' % (f)), sorted(feat_map[f].items()))
        
        
        label_map = {'a':0, 'none':1, 'the':2}
        
        # other features: go through training set; threshold infrequent cases into UNK
        print nfeatures, 'features'
    
        # convert to matrix form
        for records, subset in [(self.train, 'train'), 
                                (self.dev, 'dev'),
                                (self.test, 'test')]:
            print subset
            nrows = len(records)
            X = lil_matrix((nrows, nfeatures))
            y = np.zeros(nrows)
            
            # turn each record into a feature vector
            for i, record in enumerate(records):
                for f in feat_map:
                    offset = features[f]
                    val = record.fdict[f]
                    #print f, val, feat_map[f].get(val)
                    #print feat_map[f] 
                    j = feat_map[f].get(val)
                    
                    # Mar 17 change: binary features processed differently
                    if f in binary_features:
                        if j == 1:
                            X[i, offset] = 1
                    else: 
                        # multival features: find the correct feature to set to 1
                        if j is None:
                            j =  feat_map[f]['UNK']
                        j += offset
                        X[i, j] = 1
                y[i] = label_map[record.label]
                #print X[i, :]
                #print y[i]
                #print 
                #if i > 10: exit()
            h5f = tbOpen(os.path.join(output_dir, '%s.h5f' % subset), 'w')
            print X.shape
            print y.shape
            write_h5f_csr(h5f, '/', 'X', tb.Float64Atom(), X.tocsr())
            write_h5f_array(h5f, '/', 'y', tb.Float64Atom(), y)
        
def csv_write_rows(fname, rows):
    with open(fname, 'wb') as csvf:
        w = csv.writer(csvf)
        for row in rows:
            w.writerow(row)
            
            
class DetFeatureMatrices():
    def __init__(self, subdir):
        self.train_X, self.train_y = self.load_matrices(os.path.join(subdir, 'train.h5f'))
        self.dev_X, self.dev_y = self.load_matrices(os.path.join(subdir, 'dev.h5f'))
        self.test_X, self.test_y = self.load_matrices(os.path.join(subdir, 'test.h5f'))
        
    def load_matrices(self, fname):
        h5f = tbOpen(fname)
        X = load_h5f_csr(h5f, '/X')
        y = load_h5f_array(h5f, '/y')
        h5f.close()
        return X, y


    
class DetClassifier(object):
    '''
    Classification of determiner results.
    '''

    def __init__(self, params):
        '''
        Constructor; should be able to toggle between different classifiers.
        '''
        if params.get('classifier') == 'logistic_regression':
            print 'Logistic regression'
            self.clf = LogisticRegression()
        else:
            print 'Support vector machine'
            self.clf = SVC()
        
    def train(self, X, y):
        self.clf.fit(X, y)
    
    def predict(self, X):
        return self.clf.predict(X)
    
    def eval(self, test, gold):
        '''
        test: a numpy array of labels
        gold: a numpy array of labels with the same length as test
        
        Return the triple (#correct, #total, accuracy)
        '''
        correct = np.sum(test == gold)
        return correct, len(gold), float(correct) / len(gold)
    
    def prf1(self, test, gold):
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
    
    def save(self, fname):
        pass
    
    def load(self, fname):
        pass