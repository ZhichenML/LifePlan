# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 22:25:14 2018

@author: Zhichen
2018/3/16 linear regression layer parameters are too large, solved by dividing the number of time points.
RNN don't need to devide sample number, but linear regression must.
"""

import numpy as np
import matplotlib.pyplot as plt

from mylstm import LSTMParam, LSTMnetwork

class Euclidean_loss_layer:
    # class method, otherwise, parameter missing
    @classmethod
    def loss(self, pred, label):
        return (pred[0] - label) **2
    
    #def loss(self, predict, label):
    #    return list(map(lambda x: (x[o] - x[1])**2, zip(predict, label)))
#    @classmethod
#    def diff(self, predict, label): #* wrong derivative
#        derivative = np.zeros_like(predict)
#        derivative[0] = 2 * (predict[0] -label)
#        return derivative
    @classmethod
    def diff(self, pred, label):
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff


def example():
    
    np.random.seed(0)
    
    x_dim = 50
    hidden_size = 100

    Lstmparam = LSTMParam(hidden_size, x_dim)
    Lstmnet = LSTMnetwork(Lstmparam)
    
    y_list = [-0.5, 0.2, 0.1, -0.5]
    x_list = [np.random.random(x_dim) for _ in y_list]
    # the data is the same, check the derivative
    for iter_epoch in range(100):
        print("iter", "%2s" % str(iter_epoch), end=": ")
        for ind in range(len(y_list)):
            Lstmnet.predict(x_list[ind])
            
        print("y_pred = [" +
              ", ".join(["% 2.5f" % Lstmnet.state_list[ind].state.h[0] for ind in range(len(y_list))]) +
              "]", end=", ")
            
        loss = Lstmnet.get_loss(y_list, Euclidean_loss_layer)
        print("loss:", "%.3e" % loss)
    
        Lstmparam.apply_diff(lr = 0.1)
        Lstmnet.x_list_clear()


class linear_regression_layer:
    """the functions are all @classmethod"""
    """ self is a must """
    @classmethod
    def random_init(self, a, b, *args): 
        np.random.seed(0)
        return np.random.rand(*args) * (b - a) + a
    
    """ self is needed in calling the function, otherwise the dim is wrong
    classmethod don't need self parameter"""
    @classmethod
    def __init__(self, out_dim, in_dim):
        self.W = self.random_init( -0.1, 0.1, in_dim) 
        self.b = self.random_init( -0.1, 0.1, out_dim) 
        
        self.diff_W = np.zeros_like(self.W)
        self.diff_b = np.zeros_like(self.b)
        
        #from mylstm import random_init
#        b1 = self.random_init(self, -0.1, 0.1, 5, 3)
#        b2 = self.random_init(-0.1, 0.1, 3)
#        b3 = np.dot(a1,a2)
        
    @classmethod
    def loss(self, pred, label):
        self.output = np.dot(self.W, pred) + self.b 
        loss = (self.output - label) ** 2 #+ 0.5 * np.sum(list(map(lambda x: x **2, self.W)))
        #loss = np.sum(list(map(lambda x: (x[0] - x[1]) ** 2, zip(list(self.output), label))))
        return loss
    @classmethod
    def diff(self, pred, label):
        diff_input = np.zeros_like(pred) 
        d_input = 2 * (self.output - label)
        #d_input = 2 * list(map(lambda x: x[0] - x[1], zip(self.output, label)))
        diff_input = self.W * d_input
        
        self.diff_W += d_input * pred  #+ self.W
        self.diff_b += np.sum(d_input)
        
        
        
        return diff_input
    
    @classmethod
    def output_layer_diff(self, sample_cnt, lr = 1):
        self.W -= lr * self.diff_W / sample_cnt
        self.b -= lr * self.diff_b / sample_cnt
       
        self.diff_W = np.zeros_like(self.W)
        self.diff_b = np.zeros_like(self.b)
      

def example_LR():
    
    np.random.seed(0)
    
    x_dim = 50
    y_dim = 1
    hidden_size = 100

    Lstmparam = LSTMParam(hidden_size, x_dim)
    Lstmnet = LSTMnetwork(Lstmparam)
    linear_regression_layer(y_dim, hidden_size)
    
    #generate a dataset (a sequence)
    y_list = [-0.5, 0.2, 0.1, -0.5]
    x_list = [np.random.random(x_dim) for _ in y_list]
    # the data is the same, check the derivative
    for iter_epoch in range(100):
        print("iter", "%2s" % str(iter_epoch), end=": ")
        for ind in range(len(y_list)):
            Lstmnet.predict(x_list[ind])
            
        print("y_pred = [" +
              ", ".join(["% 2.5f" % Lstmnet.state_list[ind].state.h[0] for ind in range(len(y_list))]) +
              "]", end=", ")
            
        loss = Lstmnet.get_loss(y_list, linear_regression_layer)
        print("loss:", "%.3e" % loss)
    
        Lstmparam.apply_diff(lr = 0.1)
        linear_regression_layer.output_layer_diff(lr = 0.1)
        Lstmnet.x_list_clear()

def example_Beef():
    np.random.seed(0)
    
    import scipy as sp
    dataset = sp.io.matlab.mio.loadmat("PreBeef")
    s = list(dataset['training'][0])
    s = s[0][0]
    
    x_list = s[0:-1]
    y_list = s[1:]
    sample_cnt = len(y_list)
    
    x_dim = 1
    y_dim = 1
    hidden_size = 10
        
    Lstmparam = LSTMParam(hidden_size, x_dim)
    Lstmnet = LSTMnetwork(Lstmparam)
    linear_regression_layer(y_dim, hidden_size)
    
    num_epoch = 100
    Epoch_loss = np.zeros(num_epoch)
    # the data is the same, check the derivative
    for iter_epoch in range(num_epoch):
        print("iter", "%2s" % str(iter_epoch), end=": ")
        for ind in range(len(y_list)):
            Lstmnet.predict(x_list[ind])
            
#        print("y_pred = [" +
#              ", ".join(["% 2.5f" % Lstmnet.state_list[ind].state.h[0] for ind in range(len(y_list))]) +
#              "]", end=", ")
            
        loss = Lstmnet.get_loss(y_list, linear_regression_layer) #Euclidean_loss_layer
        Epoch_loss[iter_epoch] = loss
        print("\n loss:", "%.3e" % loss)
#        print("W = ",linear_regression_layer.W)
        Lstmparam.apply_diff(1, lr = 0.5)
        linear_regression_layer.output_layer_diff(sample_cnt, lr = 0.01)
        Lstmnet.x_list_clear()
        
    plt.figure()
    plt.plot(range(num_epoch), Epoch_loss)
    plt.xlabel('Training Epoches')
    plt.ylabel('Loss')
    plt.title('LSTM Training Loss')
    
    plt.show()
   
    
    
    
    
if __name__ == "__main__":
    example_Beef()