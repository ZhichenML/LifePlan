# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 10:38:25 2018

@author: lenovo
"""
import numpy as np

def sigmoid(x):
    return 1. /(1 + np.exp(-x))
    
def stable_sigmoid1(x):
    import math
    return math.exp(-np.logaddexp(0,-x))

def stable_sigmoid2(x):
    if x >= 0:
        return 1. /(1 + np.exp(-x))
    else:
        z = np.exp(x)
        return z /(1+z)

def sigmoid_derivative (value):
    return value * (1-value)

def tanh_derivative(value):
    return 1. - value ** 2

def random_init(a, b, *args):
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a

class LSTMParam:
    
    def __init__(self, hidden_size, x_dim):
        self.hidden_size = hidden_size
        self.x_dim = x_dim
        concate_dim = hidden_size + x_dim
        # init network parameters
        self.Wg = random_init(-0.1, 0.1, hidden_size, concate_dim)
        self.Wi = random_init(-0.1, 0.1, hidden_size, concate_dim)
        self.Wf = random_init(-0.1, 0.1, hidden_size, concate_dim)
        self.Wo = random_init(-0.1, 0.1, hidden_size, concate_dim)
        self.bg = random_init(-0.1, 0.1, hidden_size)
        self.bi = random_init(-0.1, 0.1, hidden_size)
        self.bf = random_init(-0.1, 0.1, hidden_size)
        self.bo = random_init(-0.1, 0.1, hidden_size)
        # init network derivatives
        self.Wg_diff = np.zeros((hidden_size, concate_dim)) 
        self.Wi_diff = np.zeros((hidden_size, concate_dim)) 
        self.Wf_diff = np.zeros((hidden_size, concate_dim)) 
        self.Wo_diff = np.zeros((hidden_size, concate_dim)) 
        self.bg_diff = np.zeros(hidden_size) 
        self.bi_diff = np.zeros(hidden_size) 
        self.bf_diff = np.zeros(hidden_size) 
        self.bo_diff = np.zeros(hidden_size)
        
      
#        self.Wg_diff = np.zeros_like(self.Wg)
#        self.Wi_diff = np.zeros_like(self.Wi)
#        self.Wf_diff = np.zeros_like(self.Wf)
#        self.Wo_diff = np.zeros_like(self.Wo)
#        self.bg_diff = np.zeros_like(self.bg)
#        self.bi_diff = np.zeros_like(self.bi)
#        self.bf_diff = np.zeros_like(self.bf)
#        self.bo_diff = np.zeros_like(self.bo)
        
        
    def apply_diff(self, sample_cnt, lr = 1,):
        self.Wg -= self.Wg_diff * lr / sample_cnt
        self.Wi -= self.Wi_diff * lr / sample_cnt
        self.Wf -= self.Wf_diff * lr / sample_cnt
        self.Wo -= self.Wo_diff * lr / sample_cnt
        self.bg -= self.bg_diff * lr / sample_cnt
        self.bi -= self.bi_diff * lr / sample_cnt
        self.bf -= self.bf_diff * lr / sample_cnt
        self.bo -= self.bo_diff * lr / sample_cnt
        # re-set the derivatives as zeros
        self.Wg_diff = np.zeros_like(self.Wg)
        self.Wi_diff = np.zeros_like(self.Wi)
        self.Wf_diff = np.zeros_like(self.Wf)
        self.Wo_diff = np.zeros_like(self.Wo)
        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi)
        self.bf_diff = np.zeros_like(self.bf)
        self.bo_diff = np.zeros_like(self.bo)
        

        
        
class LSTMstate:
    def __init__(self, hidden_size):
        self.g = np.zeros(hidden_size)
        self.i = np.zeros(hidden_size)
        self.f = np.zeros(hidden_size)
        self.o = np.zeros(hidden_size)
        self.s = np.zeros(hidden_size)
        self.h = np.zeros(hidden_size)
        
        self.diff_h = np.zeros_like(self.h)
        self.diff_s = np.zeros_like(self.s)
        
class LSTMnode:
    
    def __init__(self, LSTMParam, LSTMstate):
        self.Param = LSTMParam
        self.state = LSTMstate
        self.concate_x = None
    
    
    def forward(self, x, h_prev = None, s_prev = None):
        if s_prev is None: s_prev = np.zeros_like(self.state.s)
        if h_prev is None: h_prev = np.zeros_like(self.state.h)
        
        self.s_prev = s_prev
        self.h_prev = h_prev
        
        """ it should be the last hidden state """
        concate_x = np.hstack((x, h_prev))
        """ it should be the sigmoid function for the gates """
        self.state.g = np.tanh(np.dot(self.Param.Wg, concate_x) + self.Param.bg)
        self.state.i = sigmoid(np.dot(self.Param.Wi, concate_x) + self.Param.bi)
        self.state.f = sigmoid(np.dot(self.Param.Wf, concate_x) + self.Param.bf)
        self.state.o = sigmoid(np.dot(self.Param.Wo, concate_x) + self.Param.bo)
        self.state.s = self.state.f * self.s_prev + self.state.i * self.state.g 
        self.state.h = self.state.s * self.state.o
        self.concate_x = concate_x
           
    
    def BPTT(self, diff_h, diff_s):
        # there intermediate results not strored in LSTMnode
        ds = self.state.o * diff_h + diff_s
        do = self.state.s * diff_h
        di = self.state.g * ds
        dg = self.state.i * ds 
        df = self.s_prev * ds
        
        # derivative go through the activation function
        di_input = sigmoid_derivative(self.state.i) * di
        df_input = sigmoid_derivative(self.state.f) * df
        do_input = sigmoid_derivative(self.state.o) * do
        dg_input = tanh_derivative(self.state.g) * dg
        
        # derivative of parameters, *property of Param.
        self.Param.Wg_diff += np.outer(dg_input, self.concate_x)
        self.Param.Wi_diff += np.outer(di_input, self.concate_x)
        self.Param.Wf_diff += np.outer(df_input, self.concate_x)
        self.Param.Wo_diff += np.outer(do_input, self.concate_x)
        self.Param.bg_diff += dg_input;
        self.Param.bi_diff += di_input;
        self.Param.bf_diff += df_input;
        self.Param.bo_diff += do_input;
        
        # derivative of last hidden state, * used before defined
        diff_concate_x = np.zeros_like(self.concate_x)
        diff_concate_x += np.dot(self.Param.Wg.T, dg_input)
        diff_concate_x += np.dot(self.Param.Wi.T, di_input)
        diff_concate_x += np.dot(self.Param.Wf.T, df_input)
        diff_concate_x += np.dot(self.Param.Wo.T, do_input)
        
        self.state.diff_h = diff_concate_x[self.Param.x_dim:]
        self.state.diff_s = ds * self.state.f
         
        
class LSTMnetwork:
    
    def __init__(self, Param):
        self.Param = Param
        self.x_list = []
        self.state_list = []
    
    def x_list_clear(self):
        self.x_list = []
        
    def predict(self, x):
        self.x_list.append(x)
        if len(self.x_list) > len(self.state_list):
            Lstmstate = LSTMstate(self.Param.hidden_size)
            self.state_list.append(LSTMnode(self.Param, Lstmstate))
            
        idx = len(self.x_list)-1
        if idx == 0:
            self.state_list[idx].forward(self.x_list[idx])
        else:
            s_prev = self.state_list[idx-1].state.s
            h_prev = self.state_list[idx-1].state.h
            self.state_list[idx].forward(self.x_list[idx],s_prev,h_prev)
     
         
    def get_loss(self, y_list, loss_layer):
        
        assert len(self.x_list) == len(y_list)
        idx = len(self.x_list)-1
        loss = loss_layer.loss(self.state_list[idx].state.h, y_list[idx])
        diff_h = loss_layer.diff(self.state_list[idx].state.h, y_list[idx])
        
        """ it should be zeros not zeros_like """
        diff_s = np.zeros(self.Param.hidden_size)
        self.state_list[idx].BPTT(diff_h, diff_s)
        
        idx -= 1
        
        while idx >= 0:
            loss += loss_layer.loss(self.state_list[idx].state.h, y_list[idx])
            diff_h = loss_layer.diff(self.state_list[idx].state.h, y_list[idx])
            diff_h += self.state_list[idx+1].state.diff_h
            diff_s = self.state_list[idx+1].state.diff_s
            self.state_list[idx].BPTT(diff_h, diff_s)
            idx -= 1
            
            
        return loss
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        