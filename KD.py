# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:49:34 2023

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
##########################################teachers##################################
class Te(keras.Model):
    
    def __init__(self,data,data_tensor , num_class , layers_size_te1, layers_size_te2 , latent_dim_te1 , latent_dim_te2 , tempreture, step=1):  
        
        super().__init__()
        
        self.data=data
        self.data_tensor=data_tensor
        self.step = step      
        self.latent_dim_te1 = latent_dim_te1
        self.latent_dim_te2 = latent_dim_te2
        self.num_class = num_class
        self.layers_size_te1 = layers_size_te1
        self.layers_size_te2 = layers_size_te2
        self.tempreture = tempreture
        
        if step == 1:
            self.encoder= te1_Encoder( data, latent_dim_te1 , layers_size_te1)
            self.decoder= te1_Decoder( data, latent_dim_te1 , layers_size_te1)
            self.classifier = Clf(num_class, layers_size_te1 , tempreture)
            
        else:
            self.encoder= te2_Encoder( data, latent_dim_te2 , layers_size_te2)
            self.decoder= te2_Decoder( data, latent_dim_te2 ,  layers_size_te2)
            self.classifier = Clf(num_class, layers_size_te2 , tempreture)
            

    def call(self, x):
      
        means , log_var = self.encoder(x)
        
        if self.step==1:
            eps = tf.keras.backend.random_normal(shape=(x.shape[0], self.latent_dim_te1))
        else:
            eps = tf.keras.backend.random_normal(shape=(x.shape[0], self.latent_dim_te2))
        std = tf.exp(0.5 * log_var)
        z = eps * std+ means
        
        
        classifier_inputs = tf.concat([means , log_var] , axis = 1)
      
        pred_labels , softs = self.classifier(classifier_inputs)
        #soft_labels = sig_T(softs, tempreture= self.tempreture )

        recon = self.decoder(z)
        
        return recon , means, log_var, pred_labels , softs
            
def loss_te_level1(data_tensor , recon , means, log_var, y_cat , pred_labels , sample_weight , alpha):
    

    clf_loss = tf.math.multiply(keras.losses.binary_crossentropy (y_cat, pred_labels) , sample_weight)
    
    BCE = keras.losses.binary_crossentropy(data_tensor , recon)

    kl_loss = -0.5 * (1 + log_var - tf.square(means) - tf.exp(log_var))
    KLD = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    
    return (BCE + KLD + alpha * clf_loss) / len(data_tensor)


def loss_te_level2(data_tensor , recon , means, log_var, y_cat , pred_labels , soft_labels1, soft_labels2, sample_weight , alpha , a,b):
    

    clf_loss = tf.math.multiply(keras.losses.binary_crossentropy (y_cat, pred_labels) , sample_weight)
    
    BCE = keras.losses.binary_crossentropy(data_tensor , recon)

    kl_loss = -0.5 * (1 + log_var - tf.square(means) - tf.exp(log_var))
    KLD = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    
    dl1 = tf.keras.losses.kl_divergence(soft_labels1, pred_labels)
    dl2 = tf.keras.losses.kl_divergence(soft_labels2, pred_labels)
    
    return (BCE + KLD + alpha * clf_loss + a*dl1 + b*dl2) / len(data_tensor)


##########################################student#################################

class STU(keras.Model):
    
    def __init__(self,data1, data2 , data3 , num_class , stu_layers_size , stu_latent_dim , batch_size):  
        
        super().__init__()
        
        self.data1=data1
        self.data2=data2
        self.data3=data3
              
        self.stu_latent_dim = stu_latent_dim
        self.num_class = num_class
        self.stu_layers_size = stu_layers_size
        self.batch_size = batch_size
        
        self.encoder1=   stu_Encoder( data1 ,stu_latent_dim , stu_layers_size)
        self.decoder1=  stu_Decoder(data1 , stu_latent_dim , stu_layers_size)
        
        self.encoder2=  stu_Encoder( data2 ,stu_latent_dim , stu_layers_size)
        self.decoder2=  stu_Decoder(data2 , stu_latent_dim , stu_layers_size)
        
        self.encoder3=  stu_Encoder( data3 ,stu_latent_dim , stu_layers_size)
        self.decoder3=  stu_Decoder( data3 , stu_latent_dim , stu_layers_size)
        
        #self.classifer_stu = Clf(num_class, stu_layers_size)
        self.vcdn = vcdn_clf(num_view, num_class, stu_latent_dim, stu_layers_size)
        
    def call (self , data1,data2,data3):
        
        batch_size = data1.shape[0]
        means_1 , log_var_1 = self.encoder1(data1)
        means_2 , log_var_2 = self.encoder2(data2)
        means_3 , log_var_3 = self.encoder3(data3)
        
        eps = tf.keras.backend.random_normal(shape=(batch_size, stu_latent_dim))

        std_1 = tf.exp(0.5 * log_var_1)
        z_1 = eps * std_1 + means_1
        
        std_2 = tf.exp(0.5 * log_var_2)
        z_2 = eps * std_2 + means_2
        
        std_3 = tf.exp(0.5 * log_var_3)
        z_3 = eps * std_3 + means_3
        
        vcdn_inputs1 = list([means_1 , means_2 , means_3])
        vcdn_inputs2 = list([log_var_1 ,log_var_2 , log_var_3])
  
        #classifier_input = tf.concat([means_1, log_var_1, means_2, log_var_2, means_3, log_var_3], axis=1)
        #pred_labels = self.classifer_stu(classifier_input)
        
        pred_labels = self.vcdn(vcdn_inputs1 , vcdn_inputs2)
        
        recon_1 = self.decoder1(z_1)
        recon_2 = self.decoder2(z_2)
        recon_3 = self.decoder3(z_3)
        
        return recon_1, recon_2, recon_3, means_1, means_2, means_3, log_var_1, log_var_2, log_var_3, pred_labels
    

    
def loss_stu(data1 ,data2,data3 , recon_1,recon_2,recon_3 , means_1,means_2,means_3 , log_var_1,log_var_2,log_var_3 , pred_labels , y_cat_train, softs1 , softs2 ,softs3, a, b, c ,sample_weight ,alpha):
    
    #clf_loss = tf.math.multiply(keras.losses.binary_crossentropy (y_cat_train, pred_labels) , sample_weight)
    clf_loss =keras.losses.binary_crossentropy(y_cat_train, pred_labels)    
    rec_1_loss= keras.losses.binary_crossentropy(data1, recon_1)
    rec_2_loss= keras.losses.binary_crossentropy(data2, recon_2)
    rec_3_loss= keras.losses.binary_crossentropy(data3, recon_3)
    
    BCE = rec_1_loss + rec_2_loss + rec_3_loss
    
    kl_1_loss = -0.5 * (1 + log_var_1 - tf.square(means_1) - tf.exp(log_var_1))
    kl_1_loss = tf.reduce_mean(tf.reduce_sum(kl_1_loss, axis=1))
    kl_2_loss = -0.5 * (1 + log_var_2 - tf.square(means_2) - tf.exp(log_var_2))
    kl_2_loss = tf.reduce_mean(tf.reduce_sum(kl_2_loss, axis=1))
    kl_3_loss = -0.5 * (1 + log_var_3 - tf.square(means_3) - tf.exp(log_var_3))
    kl_3_loss = tf.reduce_mean(tf.reduce_sum(kl_3_loss, axis=1))
    KLD = kl_1_loss + kl_2_loss + kl_3_loss
    
    dl1 = tf.keras.losses.kl_divergence(softs1, pred_labels)
    dl2 = tf.keras.losses.kl_divergence(softs2, pred_labels)
    dl3 = tf.keras.losses.kl_divergence(softs3, pred_labels)
    
    
    return (BCE + KLD + alpha * clf_loss + ( a*dl1 + b*dl2 + c*dl3) )/ data1.shape[0]



