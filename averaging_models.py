#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 02:44:26 2023

@author: krishna
"""
import numpy as np
import pandas as pd
from collections import Counter

"""
This averaging algorithm is a simple averaging algorithm that averages all
the weights of client by giving equal weights.
"""
def model_average_q0(client_weights):
    average_weight_list=[]
    for index1 in range(len(client_weights[0])):
        layer_weights=[]
        for index2 in range(len(client_weights)):
            weights=client_weights[index2][index1]
            layer_weights.append(weights)
        average_weight=np.mean(np.array([x for x in layer_weights]), axis=0)
        average_weight_list.append(average_weight)
    return average_weight_list



"""
Algorithm for finding q. Here p indicates the ratio of malicious samples to benign sample.
"""
import load_data

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 

def find_q():
    X_train_list, y_train_list, X_valid, y_valid = load_data.load_cicids_2017_q()
    p_all_node_list = []
    
    # here malicious nodes are 3,5,7
    malicious_nodes = [3,5,7]
    for index in range(0, len(y_train_list)):
        counter_num = Counter(y_train_list[index])
        total_sample = len(y_train_list[index])
        try:
            benign_sample = counter_num[0]
        except:
            p_all_node_list.append(0)
        malicious_sample = total_sample - benign_sample
        p_node = malicious_sample/benign_sample
        p_all_node_list.append(p_node)
        
    q_all_list_node = softmax(p_all_node_list)
    return q_all_list_node
    


"""
This averaging algorithm is a q-based averaging algorithm that gives weight
to each clients and their weight get updated accordingly.
q = 0 means only normal traffic
q > 1 means malicious traffic.
"""

def model_average_q1(client_weights, q_all_list_node):
    print('Initiating averagiing with q1')
    average_weight_list=[]
    
    for index1 in range(len(client_weights[0])):
        layer_weights=[]
        for index2 in range(len(client_weights)):
            weights=client_weights[index2][index1]
            layer_weights.append(weights)
            
            
        layer_weights_parameterized = []
        for index3 in range(len(q_all_list_node)):
            layer_weights_parameterized.append(layer_weights[index3]*q_all_list_node[index3])
        
        average_weight=np.sum(np.array([x for x in layer_weights_parameterized]), axis=0)
        average_weight_list.append(average_weight)
        
                
    return average_weight_list
