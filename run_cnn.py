'''
Created on Jul 13, 2015

@author: kashefy
'''
import numpy as np
from scipy import signal
import mnist_loader

if __name__ == '__main__':
    
    x = np.array([[1, 1, 1, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 1, 1, 1],
                  [0, 0, 1, 1, 0],
                  [0, 1, 1, 0, 0]],
                 dtype='float')
    
    w_k = np.array([[1, 0, 1],
                    [0, 1, 0],
                    [1, 0, 1],],
                   dtype='float')
    
    w_k = np.rot90(w_k, 2)
    
    print x.shape, w_k.shape
    f = signal.convolve2d(x, w_k, 'valid')
    
    print f
    
    path_mnist = '/media/win/Users/woodstock/dev/data/MNIST/mnist.pkl.gz'
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper(path_mnist)
    
    filter_dim = (8, 8)
    nb_filters = 10
    weights = [np.random.standard_normal(filter_dim) for k in xrange(nb_filters)]
    biases = [np.random.standard_normal((1,)) for k in xrange(nb_filters)]
    
    print "done"
    
    pass