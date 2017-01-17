'''
Created on 2014-06-11

@author: Jackie
'''
import os


class Paths:
    def __init__(self):
        loc = self.determine_loc().strip()
        self.load_paths(loc)


    def determine_loc(self):
        '''
        "if os.path.exists('./config'):
            return open('./config').read()
        # default
        return 'hapax'
        '''
        return 'yulan'
        
    def load_paths(self, loc):
        # many paths are actually undefined
        if loc == 'hapax':
            self.root = 'C:\\cygwin64\\home\\jcheung\\projects\\2016\\definiteness\\'
            self.word2vec = 'C:\\cygwin64\\home\\jcheung\\projects\\data\\word2vec\\GoogleNews-vectors-negative300.h5f'
            self.ptb2 ="C:\\cygwin64\\home\\jcheung\\projects\\data\\ptb2\\preprocessed\\wsj_mrg\\"
            self.external = os.path.join(self.root, 'src', 'external')
            
        elif loc == 'home':
            self.root = 'C:\\cygwin64\\home\\jcheung\\src\\definiteness\\'
            
        elif loc == 'mcgill':
            self.bnc = '/home/rldata/BNC/corenlp_xml'
            self.root = '/home/2014/jcheung/projects/2016/definiteness'
            self.external = os.path.join(self.root, 'src', 'external')
            
        elif loc == 'yulan':
            self.root = 'E:\\Summer\\presup-src'
            self.ptb2 ='E:\\Summer\\wsj_mrg'
            self.external = os.path.join(self.root, 'src', 'external')
