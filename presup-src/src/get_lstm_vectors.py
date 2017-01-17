import cPickle as pkl

def get_vectors():
	path = './LSTM_rand_POS.pkl'
	f = open(path, 'rb')
	LSTM_rand_POS = cPickle.load(f)
	f.close()

	path = './LSTM_w2v_POS.pkl'
	f = open(path, 'rb')
	LSTM_w2v_POS = cPickle.load(f)
	f.close()

	return LSTM_rand_POS, LSTM_w2v_POS

if __name__ == '__main__':

	LSTM_rand_POS, LSTM_w2v_POS = get_vectors()