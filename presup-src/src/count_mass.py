'''
Created on Mar 11, 2016

@author: jcheung

Clean the files to generate lists of count nouns and mass nouns.
'''

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

punct = '.,\'`":;()[]{}<>?!@%&'

def clean_candidates(f, out_f):
    lemmatizer = WordNetLemmatizer()
    cands = set()
    with open(f) as fh:
        for line in fh:
            n = line.strip().split()[-1]
            #print n,
            n = n.strip(punct)
            
            # only allow words found as a noun in wordnet hierarchy
            if len(wn.synsets(n, 'n')) > 0:
                n = lemmatizer.lemmatize(n, 'n')
                #print n
                if len(n) > 0 and n[0].isalpha() and n[0].islower():
                    cands.add(n)
    print len(cands), 'candidates'
    out = list(cands)
    out.sort()
    with open(out_f, 'w') as out_fh:
        for n in out:
            out_fh.write(n + '\n')
    
if __name__ == '__main__':
    cn = './external/count_noun_candidates.txt'
    mn = './external/mass_noun_candidates.txt'
    
    clean_candidates(cn, './external/count_nouns.txt')
    clean_candidates(mn, './external/mass_nouns.txt')
    