# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:49:00 2023

@author: user
"""
import pandas as pd
import numpy as np
import pingouin as pg
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  , confusion_matrix , ConfusionMatrixDisplay , balanced_accuracy_score , f1_score,precision_score , recall_score, roc_auc_score, roc_curve

from config import *

########################################### encoders #########################################################################
class te1_Encoder(keras.Model):
    
    def __init__(self, data,latent_dim_te1 , layers_size_te1):
        super().__init__()
        self.data=data
        self.latent_dim_te1=latent_dim_te1
        self.layers_size_te1=layers_size_te1
        
        self.encoder=keras.Sequential()
        self.encoder.add(layers.Dense(layers_size_te1[0], activation='relu' , bias_constraint= b_constrain , kernel_constraint= w_constrain)) ##,kernel_regularizer=regulizer  #, kernel_initializer= initializer, bias_constraint= b_constrain , kernel_constraint= w_constrain ))#, kernel_constraint= w_constrain))
        
        #self.encoder.add(layers.Dropout(0.2))
        self.encoder.add(layers.Dense(layers_size_te1[1], activation='relu' , bias_constraint= b_constrain, kernel_constraint= w_constrain)) ##kernel_regularizer=regulizer#, kernel_initializer= initializer))# , kernel_constraint= w_constrain))
        self.encoder.add(layers.Dense(layers_size_te1[1], activation='relu' , bias_constraint= b_constrain, kernel_constraint= w_constrain)) ##kernel_regularizer=regulizer, kernel_initializer= initializer))# , kernel_constraint= w_constrain))
        #self.encoder.add(layers.Dropout(0.2))
        self.encoder.add(layers.Dense(layers_size_te1[2], activation='relu' , bias_constraint= b_constrain, kernel_constraint= w_constrain)) ##kernel_regularizer=regulizer#, kernel_initializer= initializer))# , kernel_constraint= w_constrain))
        
        self.z_mean = layers.Dense(latent_dim_te1, name="z_mean" , activation = "relu"  )
        self.z_log_var = layers.Dense(latent_dim_te1, name="z_log_var" , activation = "relu"  )
        
    def call(self,x):
        
        x=self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var=self.z_log_var(x)
        
        return z_mean , z_log_var 


class te2_Encoder(keras.Model):
    
    def __init__(self, data,latent_dim_te2 , layers_size_te2):
        super().__init__()
        self.data=data
        self.latent_dim_te2=latent_dim_te2
        self.layers_size_te2=layers_size_te2
        
        self.encoder=keras.Sequential()
        self.encoder.add(layers.Dense(layers_size_te2[0], activation='relu' , bias_constraint= b_constrain , kernel_constraint= w_constrain)) ##kernel_regularizer=regulizer  #, kernel_initializer= initializer ))#, kernel_constraint= w_constrain))
        self.encoder.add(layers.Dense(layers_size_te2[0], activation='relu' , bias_constraint= b_constrain , kernel_constraint= w_constrain)) ##kernel_regularizer=regulizer  #, kernel_initializer= initializer ))#, kernel_constraint= w_constrain))
        
        #self.encoder.add(layers.Dropout(0.2))
        self.encoder.add(layers.Dense(layers_size_te2[1], activation='relu' , bias_constraint= b_constrain, kernel_constraint= w_constrain)) ##kernel_regularizer=regulizer#, kernel_initializer= initializer))# , kernel_constraint= w_constrain))
        self.encoder.add(layers.Dense(layers_size_te2[1], activation='relu' , bias_constraint= b_constrain, kernel_constraint= w_constrain)) ##kernel_regularizer=regulizer, kernel_initializer= initializer))# , kernel_constraint= w_constrain))
        #self.encoder.add(layers.Dropout(0.2))
        self.encoder.add(layers.Dense(layers_size_te2[2], activation='relu' , bias_constraint= b_constrain, kernel_constraint= w_constrain)) ##kernel_regularizer=regulizer#, kernel_initializer= initializer))# , kernel_constraint= w_constrain))
        
        self.z_mean = layers.Dense(latent_dim_te2, name="z_mean" , activation = "relu"  )
        self.z_log_var = layers.Dense(latent_dim_te2, name="z_log_var" , activation = "relu"  )
        
    def call(self,x):
        
        x=self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var=self.z_log_var(x)
        
        return z_mean , z_log_var 



class stu_Encoder(keras.Model):
    
    def __init__(self, data,stu_latent_dim , stu_layers_size):
        super().__init__()
        self.data=data
        self.stu_latent_dim=stu_latent_dim
        self.stu_layers_size=stu_layers_size
        
        self.encoder=keras.Sequential()
        self.encoder.add(layers.Dense(stu_layers_size[0], activation='relu',kernel_regularizer=regulizer  , kernel_initializer= initializer, bias_constraint= b_constrain , kernel_constraint= w_constrain))
        self.encoder.add(layers.Dense(stu_layers_size[1], activation='relu',kernel_regularizer=regulizer  , kernel_initializer= initializer, bias_constraint= b_constrain , kernel_constraint= w_constrain))
        self.encoder.add(layers.Dense(stu_layers_size[2], activation='relu',kernel_regularizer=regulizer  , kernel_initializer= initializer, bias_constraint= b_constrain , kernel_constraint= w_constrain))
        
        self.z_mean = layers.Dense(stu_latent_dim, name="z_mean" ,kernel_regularizer=regulizer  , kernel_initializer= initializer, bias_constraint= b_constrain , kernel_constraint= w_constrain)
        self.z_log_var = layers.Dense(stu_latent_dim, name="z_log_var",kernel_regularizer=regulizer , kernel_initializer= initializer, bias_constraint= b_constrain , kernel_constraint= w_constrain)
        
    def call(self,x):
        
        x=self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var=self.z_log_var(x)
        
        return z_mean , z_log_var 
########################################### decoders ######################################################################


class te1_Decoder(keras.Model):
    
    def __init__(self, data , latent_dim_te1 , layers_size_te1):
        super().__init__()
        self.data=data
        self.latent_dim_te1=latent_dim_te1
        self.layers_size_te1=layers_size_te1
        
        self.decoder=keras.Sequential()
        
        self.decoder.add(layers.Dense(layers_size_te1[2], activation='relu'  , bias_constraint= b_constrain)) ##kernel_regularizer=regulizer  #, kernel_initializer= initializer ))#, kernel_constraint= w_constrain))
        #self.decoder.add(layers.Dropout(0.2))
        self.decoder.add(layers.Dense(layers_size_te1[1], activation='relu' , bias_constraint= b_constrain)) ##kernel_regularizer=regulizer  #, kernel_initializer= initializer ))#, kernel_constraint= w_constrain))
        self.decoder.add(layers.Dense(layers_size_te1[1], activation='relu' , bias_constraint= b_constrain)) ##kernel_regularizer=regulizer  #, kernel_initializer= initializer ))#, kernel_constraint= w_constrain))
        #self.decoder.add(layers.Dropout(0.2))
        self.decoder.add(layers.Dense(layers_size_te1[0], activation='relu' , bias_constraint= b_constrain)) ##kernel_regularizer=regulizer  #, kernel_initializer= initializer ))#, kernel_constraint= w_constrain))
        #self.decoder.add(layers.Dropout(0.2))
        self.decoder.add(layers.Dense(data.shape[1], activation='relu'))
        
        
    def call(self,z):
        
        x=self.decoder(z)
        
        return x 


class te2_Decoder(keras.Model):
    
    def __init__(self, data , latent_dim_te2 , layers_size_te2):
        super().__init__()
        self.data=data
        self.latent_dim_te2=latent_dim_te2
        self.layers_size_te2=layers_size_te2
        
        self.decoder=keras.Sequential()
        
        self.decoder.add(layers.Dense(layers_size_te2[2], activation='relu'  , bias_constraint= b_constrain)) ##kernel_regularizer=regulizer  #, kernel_initializer= initializer ))#, kernel_constraint= w_constrain))
        #self.decoder.add(layers.Dropout(0.2))
        self.decoder.add(layers.Dense(layers_size_te2[1], activation='relu' , bias_constraint= b_constrain)) ##kernel_regularizer=regulizer  #, kernel_initializer= initializer ))#, kernel_constraint= w_constrain))
        self.decoder.add(layers.Dense(layers_size_te2[1], activation='relu' , bias_constraint= b_constrain)) ##kernel_regularizer=regulizer  #, kernel_initializer= initializer ))#, kernel_constraint= w_constrain))
        #self.decoder.add(layers.Dropout(0.2))
        self.decoder.add(layers.Dense(layers_size_te2[0], activation='relu' , bias_constraint= b_constrain)) ##kernel_regularizer=regulizer  #, kernel_initializer= initializer ))#, kernel_constraint= w_constrain))
        #self.decoder.add(layers.Dropout(0.2))
        self.decoder.add(layers.Dense(data.shape[1], activation='relu'))
        
        
    def call(self,z):
        
        x=self.decoder(z)
        
        return x 
    
    
class stu_Decoder(keras.Model):
    
    def __init__(self, data , stu_latent_dim , stu_layers_size):
        super().__init__()
        self.data=data
        self.stu_latent_dim=stu_latent_dim
        self.stu_layers_size=stu_layers_size
        
        self.decoder=keras.Sequential()
        
        self.decoder.add(layers.Dense(stu_layers_size[2], activation='relu',kernel_regularizer=regulizer  , kernel_initializer= initializer, bias_constraint= b_constrain , kernel_constraint= w_constrain))
        self.decoder.add(layers.Dense(stu_layers_size[1], activation='relu',kernel_regularizer=regulizer  , kernel_initializer= initializer, bias_constraint= b_constrain , kernel_constraint= w_constrain))
        self.decoder.add(layers.Dense(stu_layers_size[0], activation='relu',kernel_regularizer=regulizer  , kernel_initializer= initializer, bias_constraint= b_constrain , kernel_constraint= w_constrain))
        self.decoder.add(layers.Dense(data.shape[1], activation='relu',kernel_regularizer=regulizer  , kernel_initializer= initializer, bias_constraint= b_constrain , kernel_constraint= w_constrain))
        
        
    def call(self,z):
        
        x=self.decoder(z)
        
        return x 

    

