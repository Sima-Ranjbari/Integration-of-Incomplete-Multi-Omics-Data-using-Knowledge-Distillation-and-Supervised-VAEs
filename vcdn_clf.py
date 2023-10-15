# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:49:20 2023

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

from VAEs import *
############################################# clssifiers ######################################################

class vcdn_clf(keras.Model):
    
    def __init__(self, num_view ,num_class, stu_latent_dim ,  stu_layers_size):
        super().__init__()
        self.num_view = num_view
        self.num_class=num_class
        self.stu_latent_dim = stu_latent_dim
        self.stu_layers_size=stu_layers_size
    
        self.clf=keras.Sequential()             
        self.clf.add(layers.Dense(stu_layers_size[1], activation='relu',kernel_regularizer=regulizer  , kernel_initializer= initializer, bias_constraint= b_constrain , kernel_constraint= w_constrain, input_shape=(2*(pow(stu_latent_dim,num_view)),)))
        self.clf.add(layers.Dense(stu_layers_size[2], activation='relu',kernel_regularizer=regulizer  , kernel_initializer= initializer, bias_constraint= b_constrain , kernel_constraint= w_constrain))
        self.clf.add(layers.Dense(num_class, activation='sigmoid',kernel_regularizer=regulizer  , kernel_initializer= initializer, bias_constraint= b_constrain , kernel_constraint= w_constrain))
        
    def call(self, in_list_means , in_list_vars):
        num_view = len(in_list_vars)
        x = tf.reshape( tf.matmul(tf.expand_dims(in_list_means[0],-1), tf.expand_dims(in_list_means[1],1)) , (-1,pow(stu_latent_dim,2),1) )
        for i in range(2,num_view):
            x = tf.reshape(tf.matmul(x, tf.expand_dims(in_list_means[i],1)),(-1,pow(stu_latent_dim,i+1),1))
        vcdn_feat_means = tf.reshape(x, (-1,pow(stu_latent_dim,num_view)))
        
        

        x = tf.reshape( tf.matmul(tf.expand_dims(in_list_vars[0],-1), tf.expand_dims(in_list_vars[1],1)) , (-1,pow(stu_latent_dim,2),1) )
        for i in range(2,num_view):
            x = tf.reshape(tf.matmul(x, tf.expand_dims(in_list_vars[i],1)),(-1,pow(stu_latent_dim,i+1),1))
        vcdn_feat_vars = tf.reshape(x, (-1,pow(stu_latent_dim,num_view)))
        
        vcdn_feat = tf.concat([vcdn_feat_means , vcdn_feat_vars], axis=-1)
        
        output= self.clf(vcdn_feat)
        
        return output
            
      
def soft_T(x , tempreture):
    return np.exp(x/tempreture)/sum(np.exp(x/tempreture))



class Clf(keras.Model):
    
    def __init__(self,num_class, layers_size ,tempreture):
        super().__init__()

        self.num_class=num_class
        self.layers_size=layers_size
        self.tempreture = tempreture
         
        #self.clf=keras.Sequential()
        self.dense1 = layers.Dense(layers_size[0], activation='relu'  , bias_constraint= b_constrain)
        self.dense2 = layers.Dense(layers_size[1], activation='relu' , bias_constraint= b_constrain)
        self.dense3 = layers.Dense(layers_size[2], activation='relu' , bias_constraint= b_constrain)
        self.dense4 = layers.Dense(num_class, bias_constraint= b_constrain)
  
                  
    def call(self,x):
        x= self.dense1(x)
        x= self.dense2(x)
        x= self.dense3(x)
        x= self.dense4(x)
        softs = tf.nn.softmax(x/self.tempreture)
        pred= tf.nn.softmax(x)
        

        return pred , softs

