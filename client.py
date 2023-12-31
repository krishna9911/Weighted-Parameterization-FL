#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 15:43:56 2021

@author: krishna
"""
import models

class Client:
    
    def __init__(self, dataset_x, dataset_y, epoch_number, learning_rate,weights,batch):
        self.dataset_x=dataset_x
        self.dataset_y=dataset_y
        # self.mini_batch=mini_batch
        self.epoch_number=epoch_number
        self.learning_rate=learning_rate
        # self.decay_rate=decay_rate
        self.weights=weights
        self.batch=batch
        
        
    def train(self): 
        import numpy as np
        import pandas as pd
        import matplotlib as plt
        from tensorflow import keras
        import server
        
        # model=server.get_model()
        model = models.get_cicids_model()
        
        #setting weight of the model
        model.set_weights(self.weights)
        
        model.compile(loss='sparse_categorical_crossentropy',optimizer='RMSprop',metrics=['accuracy'])
        history = model.fit(self.dataset_x, self.dataset_y,epochs=self.epoch_number,batch_size=self.batch) 
        
        #getting the final_weight
        output_weight = model.get_weights()
        
        # wait.stop()
        
        return output_weight
        
        
        

        



    