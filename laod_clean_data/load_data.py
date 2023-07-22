#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 03:25:41 2023

@author: krishna
"""


"""
This function is to prevent loading the data from local computer multiple times.
"""
# def retreive_data(): 
import pandas as pd
import numpy as np
   
def retreive_data():
    X_train = pd.read_csv('/Users/krishna/Documents/research_projects/zero-days/Data/cicids-2017/cicids_x_train.csv')
    X_valid = pd.read_csv('/Users/krishna/Documents/research_projects/zero-days/Data/cicids-2017/cicids_x_test.csv')
    y_train = np.load('/Users/krishna/Documents/research_projects/zero-days/Data/cicids-2017/cicids-y-train.npy')
    y_valid = np.load('/Users/krishna/Documents/research_projects/zero-days/Data/cicids-2017/cicids-y-test.npy')
    
    return X_train, X_valid, y_train, y_valid

"""
This function loads data where normal attack and malicious attack
are mixed together and distributed in all the node
"""
def load_cicids_2017():
    X_train, X_valid, y_train, y_valid = retreive_data()
    X_valid = X_valid[:100000]
    y_valid = y_valid[:100000]
    
    return X_train, X_valid, y_train, y_valid

"""
This function loads data where malicious attack are distributed in only certain nodes
where some amount of benign traffic is also present. The ratio of malicious to benign
traffic in these nodes is 9:1. and other normal traffic are distributed in
non-malicious nodes. If K is total nodes then K_b are benign nodes and K_m are 
malicious nodes.
"""
def load_cicids_2017_q():
    X_train, X_valid, y_train, y_valid = retreive_data()    
    X_train['Label'] = y_train
    X_train_full = X_train
    X_train = X_train.drop(['Label'], axis = 1)
    
    #In this dataset 0 represents benign traffic and other value represents other.
    X_malicious = X_train_full[X_train_full['Label']!=0]
    X_benign = X_train_full[X_train_full['Label']==0]
    
    attack_dictionary = {}
    
    #selecting only subset of data as there are too much data in valiation set.
    X_valid = X_valid[1000:5000]
    y_valid = y_valid[1000:5000]                
        
    """
    Nodes 4,6,8 are malicious
    """
    
    malicious_nodes = [4,6,8]
    X_train_list = []
    y_train_list = []
    for index in range(1,11):
        if index in malicious_nodes:
            dataset_malicious = X_malicious[index*10000:(index+1)*10000]
            dataset_benign = X_benign[index*2000:(index+1)*2000]
            dataset_joined = pd.concat([dataset_benign, dataset_malicious])
            dataset_joined = dataset_joined.sample(frac=1)
            y_train_list.append(dataset_joined['Label'])
            X_train_list.append(dataset_joined.drop(['Label'], axis = 1))
        
        else:
            dataset_joined = X_benign[index*2000:(index+1)*2000]
            dataset_joined = dataset_joined.sample(frac=1)
            y_train_list.append(dataset_joined['Label'])
            X_train_list.append(dataset_joined.drop(['Label'], axis = 1))
            

    
    return X_train_list, y_train_list, X_valid, y_valid


# def load_nsl_kdd():
    