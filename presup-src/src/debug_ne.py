import cPickle
import numpy as np
from models.classification import DetFeatureMatrices
import os, csv

def get_ne_vec():
	# jad's version
	path = './named_entities.pkl'
	f = open(path, 'rb')
	ne_list = cPickle.load(f)
	non_ne_list = cPickle.load(f)
	f.close()

	#print ne_list[:20]
	#print non_ne_list[:20]
	tot_cases = len(ne_list) + len(non_ne_list)
	ne_vec = np.zeros(tot_cases)
	ne_vec[ne_list] = 1
	return ne_vec


def get_my_vec():
	# my version
	subdir_h5f = './features/h5f'
	matrices = DetFeatureMatrices(subdir_h5f)
	
	X = matrices.test_X
	feats = {}
	subdir_h5f = './features/h5f'
	with open(os.path.join(subdir_h5f, 'features.csv'), 'rb') as csvf:
	    r = csv.reader(csvf)
	    for row in r:
	        feats[row[0]] = int(row[1])
	ne_feat = feats['ne'] # this is the feature for ne=no
	
	# masks
	col = X[:, ne_feat].todense()
	col = np.array(col)
	col = col.reshape(len(col))
	
	return col
	

def main():
	ne_vec = get_ne_vec()
	ne_vec2 = get_my_vec()
	
	print np.where(ne_vec[:100])
	print np.where(ne_vec2[:100])

	print "length of list of indices of NE:    ", np.sum(ne_vec == 1)
	print "length of list of indices of non-NE:", np.sum(ne_vec == 0)

	print "length of list of indices of NE:    ", np.sum(ne_vec2 == 1)
	print "length of list of indices of non-NE:", np.sum(ne_vec2 == 0)

if __name__ == '__main__':
	main()