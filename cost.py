'''
Created on Jul 7, 2015

@author: kashefy
'''
import numpy as np
from network import sigmoid_prime_vec

class CrossEntropyCost(object):
    '''
    classdocs
    '''
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.nan_to_num(np.sum(-y*np.log(a)-(1-y)*np.log(1-a)))
    
    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return a-y
    
class QuadraticCost:
    '''
    classdocs
    '''
    @staticmethod
    def fn(a, y):
        """
        Return the cost associated with an output ``a`` and desired output
        ``y``.
        """
        return np.linalg.norm(a-y)**2/2.
    
    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return a-y * sigmoid_prime_vec