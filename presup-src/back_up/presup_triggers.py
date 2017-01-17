'''
Created on Feb 18, 2016

@author: jcheung
'''

import sys
sys.path.insert(0, r'../')

import cPickle
import random
import numpy as np
import csv, os, glob
import tables as tb
from scipy.sparse import lil_matrix, csr_matrix
from utils.utils import tbOpen, load_h5f_csr, write_h5f_csr, write_h5f_array, load_h5f_array
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

pos_stoplist = set(['DT', ',', '.', '$', "''", '#', '-LRB-', '-RRB-', ':', '``'])
target_words = ['too', 'again', 'also', 'still', 'yet'] #, 'anymore', 'just', 'only', 'even']
# anymore # rare, can be predicted by presence of DEO
#'just' # very ambiguous
#'only' 
#'even' # also ambiguous; many uses are not presuppositional or related to context

class PresupExtractor(object):
    '''
    Extract features
    '''

    def __init__(self, params):
        '''
        Constructor
        '''
        self.params = params
        self.debug = params.get('debug', True)
        self.records_dict = None
        
    def run(self, corpus,subset='all'):
        ndocs = 0
        self.records_dict = {} # doc -> list of records


        for doc in corpus.doc_iter(subset,0):
            self.extract_num(doc)
            self.records_dict[doc.num] = self.extract_records(doc)
            ndocs += 1
            if self.debug and ndocs >= 5:
                break
            # print doc.num


        print 'Done with iteration ', str(ndocs)
###########################Edited##########################
        # a dict to store the verbs modified by adverb
        verb_dict={}
        counts = dict.fromkeys(target_words, 0)
        for _, records in self.records_dict.iteritems():
            for r in records:
                counts[r.label] += 1
                if self.params.get('records_out'):
                    if not r.governor in verb_dict:
                        verb_dict[r.governor] = 1
                    else:
                        verb_dict[r.governor] += 1
                    print '%s\t%s\t%s\t%s\t%s' % (str(r.loc), r.governor, r.gov_pos, r.label, r.gov_word)
        # # print sum(verb_dict.itervalues())
        # # print verb_dict
        # print counts


        # print "extracting negative cases"
        # self.extract_negative(corpus,subset,'results/extract_negative_test_data.pkl')
        # print "done extracting negative cases"
        # print "extracting positive cases"
        # self.extract_positive(corpus,subset, 'presup/positive_data.pkl')
        # print "done extracting positive cases"


    def extract_negative(self, corpus,subset,verb_dict,out):
        # extract the negative cases
        with open(out, 'wb') as output:
            for doc in corpus.doc_iter(subset,1):
                name_list = self.extract_num(doc)
                # print name_list
                for one in name_list:
                    if one in verb_dict:
                        indices, new_name_list = self.search(name_list, one)
                        if len(indices)>1:
                            new_indices=random.sample(indices,len(indices))  #shuffle indices for unbiased sample
                            indices = new_indices
                        #print indices
                        for number in indices:
                            if set(target_words).isdisjoint(name_list[number - 3:number + 3]):
                                # print one,verb_dict[one]
                                if verb_dict[one] >= 1:
                                    verb_dict[one] -= 1
                                    if number <= 100:
                                        tuple = (0, new_name_list[:number])
                                    else:
                                        tuple = (0, new_name_list[number - 100:number])
                                    cPickle.dump(tuple, output)
        # print "getting rare verblist"
        # rare_verblist=[(i, verb_dict[i]) for i in verb_dict if verb_dict[i] != 0]
        # with open('results/verblist_all.pkl','wb') as f:
        #     cPickle.dump(verb_dict,f)
        #     cPickle.dump(rare_verblist,f)


    def extract_positive(self,corpus,subset,out):
        with open(out, 'wb') as output:
            for doc in corpus.doc_iter(subset, 0):
                name_list = self.extract_num(doc)
                word_list = [word[0] for word in name_list]
                pos_list = [word[1] for word in name_list]
                for one in target_words:
                    if one in word_list:
                        print " ".join(x for x in word_list)
                        indices_of_adverb, indices_of_end = self.search(word_list, one)
                        for i in range(len(indices_of_adverb)):
                            j= indices_of_adverb[i] #position of adverb
                            k= indices_of_end[i] #position of end of sentence
                            if j<=100:
                                tuple=(one,word_list[:j]+word_list[j+1:k],pos_list[:j]+pos_list[j+1:k])
                            else:
                                tuple = (one, word_list[j-100:j]+word_list[j+1:k],pos_list[j-100:j]+pos_list[j+1:k])
                            print tuple
                            raw_input("---")
                            # cPickle.dump(tuple,output)


    def search(self, name_list, key_word):
        # print name_list
        indices_of_adverb = [i for i, x in enumerate(name_list) if x==key_word]
        indices_of_end = [number+name_list[number:].index(".")for number in indices_of_adverb] #indices of the end of the sentence
        # print indices_of_adverb,indices_of_end
        # raw_input("---")
        return indices_of_adverb,indices_of_end



    def extract_num(self, doc):
        name_list = [] #(word,pos)
        for sentn, sent in doc.sents.iteritems():
            for k in sent.tokens.keys():
                word = sent.tokens[k].word
                pos = sent.tokens[k].pos
                # if word.startswith("@@@@"):
                #     special_tuple=("@@@@","@@@@")
                #     name_list.append(tuple)
                #     word=word[4:]
                tuple = (word,pos)
                name_list.append(tuple)
        # print name_list
        return name_list


    def extract_records(self, doc):
        '''
        Extract records from a Document.
        Returns a list of Records
        '''
        records = []
        #print doc.num
        for sentn, sent in doc.sents.iteritems():
            # print sentn, sent.text()
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
            w = chi_token.word.lower()
            if w in target_words:
                #print '*'
                r = self.extract_record(docn, sentn, chi, sent, chi_token, node)
                # this is a record!
                if r is not None:
                    records.append(r)
                    # print node
                    # print r
                    # print w
                    # print docn
                    # print chi_token.word
                    # raw_input('record')
            else:
                pass
        # print records
        # raw_input("---")
        return records
                
    def extract_record(self, docn, sentn, wordn, sent, hnoun, node):
        '''
        Extract from the sentence at the specified token hnoun, using the node (i.e., edge) 
        whose child is the location in the sentence 
        '''
        r = PresupRecord()
        r.label = hnoun.word.lower()
        r.loc = (docn, sentn, wordn)
        
        i = node.governor
        if i >= 1:
            r.governor = sent.at(i).lemma
            r.gov_pos = sent.at(i).pos
            r.gov_word = "@@@@"+ sent.at(i).word
            # too + ADJ is another "too"
            if r.label == 'too' and r.gov_pos.startswith(('JJ', 'RB')):
                return None


        return r


class PresupRecord:
    def __init__(self):
        self.fdict = {} # fname -> fval
        self.label = None
        self.governor = None
        self.gov_pos = None
        self.gov_word = None
        self.loc = None # (docnum, sentnum, tokennum)
    def __str__(self):
        return '%s\t%s\t%s' % (self.loc, self.label, str(self.fdict))
