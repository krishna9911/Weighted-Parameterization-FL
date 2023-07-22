#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 19:08:06 2021

@author: krishna
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from tensorflow import keras
import time
import pickle
from imblearn.over_sampling import SMOTE
from tensorflow.keras.regularizers import  l1,l2
import tensorflow as tf


#loading the dataset ##should be in the form of X_train, y_train, X_valid,y_valid
# import clean_cicids_2017
import load_data
x_data, y_data, X_valid, y_valid = load_data.load_cicids_2017_q()


#dividing the data for cicids2017
# import dataset_divider
# x_data, y_data=dataset_divider.divide_without_label(10,X_train[:100000], y_train[:100000])



#loading the model here cicids 2017 or nsl kdd.
import models
model_cicids_2017 = models.get_cicids_model()
model_nsl_kdd = models.get_nslkdd_model()
    

#Change here for the required averginf algo you need
import averaging_models


def create_model():
    model=model_cicids_2017
    weight=model.get_weights()
    return weight
    

def evaluate_model(accuracy_list,weight,learning_rate):
    model=models.get_cicids_model()
    model.set_weights(weight)
    model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=learning_rate),metrics=['accuracy'])
    result=model.evaluate(X_valid, y_valid)
    
    if len(accuracy_list)==0:
        accuracy_list.append(0)
        if result[1] > accuracy_list[len(accuracy_list)-1]:
            return True,result[1]
        
    elif result[1] > accuracy_list[len(accuracy_list)-1]:
            return True,result[1]
    else:
        return False,result[1]
    

#initializing the client automatically
from client import Client
def train_server(training_rounds,epoch,batch,learning_rate):
    
    accuracy_list=[]
    loss_list = []
    client_weight_for_sending=[]
    
    print('Total number of training rounds=', training_rounds)
    q_all_list_node = averaging_models.find_q()
    print(q_all_list_node)      #remove it 
    for index1 in range(1,training_rounds):
        print('Training for round ', index1, 'started')
        client_weights_tobe_averaged=[]
        for index in range(len(y_data)):
            print(index)
            print('-------Client-------', index)
            if index1==1:
                print('Sharing Initial Global Model with Random Weight Initialization')
                initial_weight=create_model()
                client=Client(x_data[index],y_data[index],epoch,learning_rate,initial_weight,batch)
                weight=client.train()
                client_weights_tobe_averaged.append(weight)
            else:
                client=Client(x_data[index],y_data[index],epoch,learning_rate,client_weight_for_sending[index1-2],batch)
                weight=client.train()
                client_weights_tobe_averaged.append(weight)
    
        
        #calculating the avearge weight from all the clients
        client_average_weight=averaging_models.model_average_q1(client_weights_tobe_averaged, q_all_list_node)
        client_weight_for_sending.append(client_average_weight)
        

        #validating the model with avearge weight
        model=model_cicids_2017

        model.set_weights(client_average_weight)
        model.compile(loss='sparse_categorical_crossentropy',optimizer=keras.optimizers.Adam(learning_rate=learning_rate),metrics=['accuracy'])
        result=model.evaluate(X_valid, y_valid)
        loss = result[0]
        accuracy = result[1]
        print('#######-----Acccuracy for round ', index1, 'is ', accuracy, ' and loss is', loss , ' ------########')
        accuracy_list.append(accuracy)
        loss_list.append(loss)
        
    return accuracy_list,client_weight_for_sending, loss_list


def train_server_weight_discard(training_rounds,epoch,batch,learning_rate):
    accuracy_list=[]
    client_weight_for_sending=[]
    
    for index1 in range(1,training_rounds):
        print('Training for round ', index1, 'started')
        client_weights_tobe_averaged=[]
        for index in range(len(y_data)):
            print(len(y_data))
            print('-------Client-------', index)
            if index1==1:
                print('Sharing Initial Global Model with Random Weight Initialization')
                initial_weight=create_model()
                client=Client(x_data[index],y_data[index],epoch,learning_rate,initial_weight,batch)
                weight=client.train()
                client_weights_tobe_averaged.append(weight)
            else:
                client=Client(x_data[index],y_data[index],epoch,learning_rate,client_weight_for_sending[index1-2],batch)
                weight=client.train()
                client_weights_tobe_averaged.append(weight)
        
        #calculating the avearge weight from all the clients
        client_average_weight=averaging_models.model_average_q0(client_weights_tobe_averaged)
        boolean, accuracy=evaluate_model(accuracy_list,client_average_weight,learning_rate)
        if boolean==True:
            client_weight_for_sending.append(client_average_weight)
            print('#######-----Acccuracy for round ', index1, 'is ', accuracy, ' ------########')
            accuracy_list.append(accuracy)
            
        else:
            print('Weight discarded due to low accuarcy')
            client_weight_for_sending.append(client_weight_for_sending[len(client_weight_for_sending)-1])
            accuracy_list.append(accuracy_list[len(accuracy_list)-1])
            
    return accuracy_list,client_weight_for_sending
        

#initializng the traiing work
def train_main():
    start=time.time()
    training_accuracy, weights, loss_list = train_server(100,2,64,0.01)
    end=time.time()
    print('TOTAL TIME ELPASED = ', end-start)
    return training_accuracy, loss_list
    
if __name__== "__main__":
    training_accuracy, loss_list = train_main()
    

# saving the accuracy and loss
training_accuracy1 = np.array(training_accuracy)
loss1 = np.array(loss_list)
np.save('accuracy_cicids_fedq', training_accuracy1)
np.save('loss_cicids_fedq', loss1)

loss_fed_avg = np.load('/Users/krishna/Documents/research_projects/zero-days/loss_cicids_fedavg.npy')
loss1 = np.load('/Users/krishna/Documents/research_projects/zero-days/loss_cicids_fedq.npy')

import matplotlib.pyplot as plt
# loss_train = training_accuracy
# loss_val = history.history['val_loss']
epochs = range(1,100)
plt.plot(epochs, loss1, 'g', label='Federated Q loss')
plt.plot(epochs, loss_fed_avg, 'b', label='Federated avg loss')
plt.title('Loss')
plt.xlabel('Communication rounds')
plt.ylabel('Federated Q vs Federated Average loss')
plt.legend()
plt.show()



