'''
Created on Jul 3, 2015

@author: kashefy
'''
import mnist_loader
import cost
from network import Network
from network2 import Network2

def run_network(training_data, validation_data, test_data):
    
    #~ net = Network([784, 30, 10])
    net = Network([784, 10])
    
    epochs = 30
    mini_batch_size = 10
    eta = 3.0   # learning rate
    net.SGD(training_data, epochs, mini_batch_size, eta, test_data=test_data)
    
    return 0
    
def run_network2(training_data, validation_data, test_data):

    net = Network2([784, 30, 10], cost=cost.CrossEntropyCost())
    net.SGD(training_data, 30, 10, 0.5,
            lmbda=5.0,
            evaluation_data=validation_data,
            monitor_evaluation_accuracy=True,
            monitor_evaluation_cost=True,
            monitor_training_accuracy=True,
            monitor_training_cost=True)
    
    return 0

if __name__ == '__main__':
    
    path_mnist = '/media/win/Users/woodstock/dev/data/MNIST/mnist.pkl.gz'
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper(path_mnist)
    
    run_network(training_data, validation_data, test_data)
    #~ run_network2(training_data, validation_data, test_data)
    
    pass
