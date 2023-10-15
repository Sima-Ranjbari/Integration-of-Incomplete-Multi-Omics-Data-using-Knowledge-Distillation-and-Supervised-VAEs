# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:49:46 2023

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
from KD import *
from preprocess import *
from vcdn_clf import *

###########################################################################################################################################3
def cal_sample_weight(labels, num_class, use_sample_weight=True):

    if not use_sample_weight:
        return np.ones(len(labels)) 
    count = np.zeros(num_class)
    for i in range(num_class):
        count[i] = np.sum(labels==i)
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        if i==1:
            sample_weight[np.where(labels==i)[0]] =   9*count[i]/np.sum(count) #10 for teacher
        else:
            sample_weight[np.where(labels==i)[0]] =  count[i]/np.sum(count) #zero for teacher
           
    return sample_weight


    

def training_te_level1(data,data_tensor , num_class , layers_size_te1 , layers_size_te2 ,  lr ,latent_dim_te1, latent_dim_te2 , y , y_cat, num_epoch , batch_size, tempreture, use_sample_weight=True , steps=1 ,num_te= 1):
    num_of_batch = int((len(data))/batch_size)+1
    alpha = 0.01 * len(data)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr )
    total_loss =[]
    total_acc=[]
    
    model = Te(data, data_tensor, num_class, layers_size_te1 , layers_size_te2, latent_dim_te1, latent_dim_te2 , tempreture, step=steps)
    
    print(f"start Teacher trainiing in step ({steps}):")
    
    for epoch in range(num_epoch):
        print("Teacher trainiing")
        print("\nStart of epoch %d" % (epoch,)) 
        epoch_pred = []
        epoch_loss = []
        preds= np.zeros(shape=(y.shape[0] , 2 ))
        softs = np.zeros(shape = (y.shape[0] , 2))
        for step in range(num_of_batch):
            
            data_batch_train = data_tensor[step*batch_size: batch_size*(step+1)]
            y_batch_train = y_cat[step*batch_size: batch_size*(step+1)]  
            y_batch_sample_weight = y[step*batch_size: batch_size*(step+1)]  
            
            sample_weight = cal_sample_weight(y_batch_sample_weight, num_class , use_sample_weight)
            with tf.GradientTape() as tape:
                
                recon , means, log_var, pred_labels , soft_labels = model(data_batch_train)
                loss_value = loss_te_level1(data_batch_train, recon, means, log_var, y_batch_train, pred_labels,sample_weight , alpha)

            grads = tape.gradient(loss_value,model.trainable_weights) 
            optimizer.apply_gradients(zip(grads , model.trainable_weights))
            
            loss_batch = np.mean(loss_value)
            epoch_loss.append(loss_batch)
            y_pred = np.argmax(pred_labels , axis=-1)
            epoch_pred.append(y_pred)
            prediction = list(np.concatenate(epoch_pred))
            
            preds[step*batch_size: batch_size*(step+1) , : ] = pred_labels
            softs[step*batch_size: batch_size*(step+1) , : ] = soft_labels
            
        preds = tf.convert_to_tensor(preds)
        softs =tf.convert_to_tensor(softs)
        mean_loss_train = sum(epoch_loss)/len(epoch_loss)
        acc = accuracy_score(y , prediction)
          
        
        total_loss.append(mean_loss_train)
        total_acc.append(acc)    
        print(f"epoch: {epoch} train_loss: {mean_loss_train}, acc: {acc}") 
        
        
    model.save_weights(':/KD_TE_' + str(steps)+str(num_te) , save_format='tf')
    return softs 





def training_te_level2(data,data_tensor , num_class , layers_size_te1 , layers_size_te2 ,  lr ,latent_dim_te1, latent_dim_te2 , y , y_cat, softs1 , softs2, a, b, num_epoch , batch_size, tempreture, use_sample_weight=True ,steps=2 , num_te= 1):
    num_of_batch = int((len(data))/batch_size)+1
    alpha = 0.01 * len(data)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr )
    total_loss =[]
    total_acc=[]
    
    model = Te(data, data_tensor, num_class, layers_size_te1 , layers_size_te2, latent_dim_te1, latent_dim_te2 , tempreture, step=steps)
    
    print(f"start Teacher trainiing in step ({steps}):")
    
    for epoch in range(num_epoch):
        print("Teacher trainiing")
        print("\nStart of epoch %d" % (epoch,)) 
        epoch_pred = []
        epoch_loss = []
        preds= np.zeros(shape=(y.shape[0] , 2 ))
        softs = np.zeros(shape = (y.shape[0] , 2))
        for step in range(num_of_batch):
            
            data_batch_train = data_tensor[step*batch_size: batch_size*(step+1)]
            y_batch_train = y_cat[step*batch_size: batch_size*(step+1)]  
            y_batch_sample_weight = y[step*batch_size: batch_size*(step+1)] 
            
            softs1_batch = softs1[step*batch_size: batch_size*(step+1)]
            softs2_batch = softs2[step*batch_size: batch_size*(step+1)]
            
            sample_weight = cal_sample_weight(y_batch_sample_weight, num_class , use_sample_weight)
            with tf.GradientTape() as tape:
                
                recon , means, log_var, pred_labels , soft_labels = model(data_batch_train)
                loss_value = loss_te_level2(data_batch_train, recon, means, log_var, y_batch_train, pred_labels, softs1_batch , softs2_batch , sample_weight , alpha ,a , b)

            grads = tape.gradient(loss_value,model.trainable_weights) 
            optimizer.apply_gradients(zip(grads , model.trainable_weights))
            
            loss_batch = np.mean(loss_value)
            epoch_loss.append(loss_batch)
            y_pred = np.argmax(pred_labels , axis=-1)
            epoch_pred.append(y_pred)
            prediction = list(np.concatenate(epoch_pred))
            
            preds[step*batch_size: batch_size*(step+1) , : ] = pred_labels
            softs[step*batch_size: batch_size*(step+1) , : ] = soft_labels
            
        preds = tf.convert_to_tensor(preds)
        softs =tf.convert_to_tensor(softs)
        mean_loss_train = sum(epoch_loss)/len(epoch_loss)
        acc = accuracy_score(y , prediction)
        
        
        total_loss.append(mean_loss_train)
        total_acc.append(acc)    
        print(f"epoch: {epoch} train_loss: {mean_loss_train}, acc: {acc}") 
        
        
    model.save_weights(':/KD_TE_' + str(steps)+str(num_te) , save_format='tf')
    return softs 







def get_predictions(x, model_name , step=1):
    x_tensor = tf.convert_to_tensor(x)
    teacher = Te (x, x_tensor, num_class, layers_size_te1 , layers_size_te2, latent_dim_te1, latent_dim_te2 , tempreture = 1.5, step=step)
    teacher.load_weights( model_name )
    
    _ , _, _, _, soft_labels = teacher(x_tensor)
    
    return soft_labels
    
    
def training_stu(data1,data2,data3 , num_class , layers_size ,  lr ,stu_latent_dim , y , y_cat, softs1, softs2, softs3, a, b, c, num_epoch , batch_size , tempreture, use_sample_weight=True):
    
    num_of_batch = int((len(data1))/batch_size)+1
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr )
    alpha = 0.05 * len(data1) 
    total_loss = []
    total_acc = []
    
    stu = STU(data1, data2, data3, num_class, stu_layers_size, stu_latent_dim, batch_size)
    
    print("start STU trainiing")
    for epoch in range(num_epoch):
        print("STU trainiing")
        print("\nStart of epoch %d" % (epoch,)) 
        epoch_pred = []
        epoch_loss = []
        for step in range(num_of_batch):
            
            data1_batch_train = data1[step*batch_size: batch_size*(step+1)]

            
            data2_batch_train = data2[step*batch_size: batch_size*(step+1)]
            
            
            data3_batch_train = data3[step*batch_size: batch_size*(step+1)]
            
            y_batch_train = y_cat[step*batch_size: batch_size*(step+1)]
           
            y_batch_sample_weight = y[step*batch_size: batch_size*(step+1)]  
            
            soft1_batch_train = softs1[step*batch_size: batch_size*(step+1)]
            soft2_batch_train = softs2[step*batch_size: batch_size*(step+1)]
            soft3_batch_train = softs3[step*batch_size: batch_size*(step+1)]
                                       
                                       
            
            sample_weight = cal_sample_weight(y_batch_sample_weight, num_class , use_sample_weight)
                
            with tf.GradientTape() as tape:
                recon_1, recon_2, recon_3, means_1, means_2, means_3, log_var_1, log_var_2, log_var_3, pred_labels= stu(data1_batch_train , data2_batch_train, data3_batch_train)
                loss_value = loss_stu(data1_batch_train ,data2_batch_train,data3_batch_train , recon_1,recon_2,recon_3 , means_1,means_2,means_3 , log_var_1,log_var_2,log_var_3 , pred_labels , y_batch_train, soft1_batch_train, soft2_batch_train, soft3_batch_train, a,b,c , sample_weight, alpha)

            grads = tape.gradient(loss_value,stu.trainable_weights) 
            optimizer.apply_gradients(zip(grads , stu.trainable_weights))
            
            loss_batch = np.mean(loss_value)
            epoch_loss.append(loss_batch)
            y_pred = np.argmax(pred_labels , axis=-1)
            epoch_pred.append(y_pred)
            prediction = list(np.concatenate(epoch_pred))
        mean_loss_train = sum(epoch_loss)/len(epoch_loss)
        acc = accuracy_score(y , prediction)
          
        
        total_loss.append(mean_loss_train)
        total_acc.append(acc)    
        print(f"epoch: {epoch} train_loss: {mean_loss_train}, acc: {acc}") 
        
    stu.save_weights(':/brc_stu' , save_format='tf')
    return total_loss , total_acc, conf 





################### step1 ###################

softs = training_te_level1(x1 , x1_tensor , num_class=2 , layers_size_te1=layers_size_te1 , layers_size_te2= layers_size_te2 , lr=0.005 , latent_dim_te1=latent_dim_te1 , latent_dim_te2= latent_dim_te2, y=y1 , y_cat=y1_cat , num_epoch=15 , batch_size=te_batch_size  ,tempreture=1.5,use_sample_weight=True ,steps=1, num_te= 1)
softs1_2_s2 = get_predictions(x_c_1_2, model_name='KD_TE_11' , step=1)
softs1_3_s2 = get_predictions(x_c_1_3, model_name='KD_TE_11' , step=1)


softs2 = training_te_level1(x2 , x2_tensor , num_class=2 , layers_size_te1=layers_size_te1 , layers_size_te2= layers_size_te2 , lr=0.001 , latent_dim_te1=latent_dim_te1 , latent_dim_te2= latent_dim_te2, y=y2 , y_cat=y2_cat , num_epoch=15 , batch_size=te_batch_size ,  tempreture=1, use_sample_weight=True ,  steps=1 ,num_te= 2)
softs2_1_s2 = get_predictions(x_c_2_1, model_name='KD_TE_12' , step=1)
softs2_3_s2 = get_predictions(x_c_2_3, model_name='KD_TE_12' , step=1)


softs3 = training_te_level1(x3 , x3_tensor , num_class=2 , layers_size_te1=layers_size_te1 , layers_size_te2= layers_size_te2 , lr=0.001 , latent_dim_te1=latent_dim_te1 , latent_dim_te2= latent_dim_te2, y=y3 , y_cat=y3_cat , num_epoch=15 , batch_size=te_batch_size , tempreture=1 , use_sample_weight=True , steps=1 ,num_te= 3)
softs3_1_s2 = get_predictions(x_c_3_1, model_name='KD_TE_13' , step=1)
softs3_2_s2 = get_predictions(x_c_3_2, model_name='KD_TE_13' , step=1)



#################### step2 ##################

print("start step2")
inputs1_2_s2 = tf.concat([x_c_1_2, x_c_2_1] , axis=1)
inputs1_3_s2 = tf.concat([x_c_1_3, x_c_3_1] , axis=1)
inputs2_3_s2 = tf.concat([x_c_2_3, x_c_3_2] , axis=1)




softs2_1 = training_te_level2 (inputs1_2_s2 , inputs1_2_s2 , num_class=2 , layers_size_te1=layers_size_te1 , layers_size_te2= layers_size_te2 , lr=0.001 , latent_dim_te1=latent_dim_te1 , latent_dim_te2= latent_dim_te2, y=y_c_1_2 , y_cat=y_c_1_2_cat , softs1= softs1_2_s2, softs2 = softs2_1_s2, a=0.7 , b=0.3, num_epoch=15 , batch_size=te_batch_size ,  tempreture = 1.5, use_sample_weight=True , steps=2 , num_te= 1)
softs1_2_s3 = get_predictions(tf.concat([x1_c,x2_c],axis=1), model_name='KD_TE_21' , step=2)


softs2_2 = training_te_level2(inputs1_3_s2 , inputs1_3_s2 , num_class=2 , layers_size_te1=layers_size_te1 , layers_size_te2= layers_size_te2 , lr=0.001 , latent_dim_te1=latent_dim_te1 , latent_dim_te2= latent_dim_te2, y=y_c_1_3 , y_cat=y_c_1_3_cat , softs1 = softs1_3_s2, softs2=softs3_1_s2 , a=0.3,b=0.7 ,num_epoch=15 , batch_size=te_batch_size , tempreture=1.5, use_sample_weight=True ,  steps=2 , num_te= 2)
softs1_3_s3 = get_predictions(tf.concat([x1_c, x3_c] ,axis=1), model_name='KD_TE_22' , step=2)


softs2_3 = training_te_level2(inputs2_3_s2 , inputs2_3_s2 , num_class=2 , layers_size_te1=layers_size_te1 , layers_size_te2= layers_size_te2 , lr=0.001 , latent_dim_te1=latent_dim_te1 , latent_dim_te2= latent_dim_te2, y=y_c_2_3 , y_cat=y_c_2_3_cat , softs1 = softs2_3_s2, softs2= softs3_2_s2, a=0.7, b=0.3 ,num_epoch=15 , batch_size=te_batch_size  , tempreture=1.5, use_sample_weight=True , steps=2, num_te= 3)
softs2_3_s3 = get_predictions(tf.concat([x2_c, x3_c] , axis=1), model_name='KD_TE_23' , step=2)



######################### step3 ###################



hist_loss, hist_acc, last_conf = training_stu(data1= x1_c_tensor , data2= x2_c_tensor, data3=x3_c_tensor , num_class=2, layers_size=stu_layers_size,  lr=0.001 , stu_latent_dim=stu_latent_dim, y=y1_c, y_cat=y1_c_cat, softs1=softs1_2_s3 , softs2 = softs1_3_s3 , softs3 = softs2_3_s3 , a=0.2 , b=6.2, c=1.8 ,num_epoch=25, batch_size=stu_batch_size, tempreture=1.5 ,use_sample_weight=True)

############################# testing ##########################    
    
print(f"size of common data is {x1_test.shape}, {x2_test.shape} , {x3_test.shape}")

x1_test_tensor = tf.convert_to_tensor(x1_test)
x1_test_tensor= tf.cast(x1_test_tensor , tf.float32)

x2_test_tensor = tf.convert_to_tensor(x2_test)
x2_test_tensor= tf.cast(x2_test_tensor , tf.float32)

x3_test_tensor = tf.convert_to_tensor(x3_test)
x3_test_tensor= tf.cast(x3_test_tensor , tf.float32)

y_test_cat = keras.utils.to_categorical(y1_test)

# #############################################################################################################

def testing_stu(x1_test_tensor, x2_test_tensor, x3_test_tensor, num_class , layers_size ,latent_dim , y_test , y_test_cat , batch_size):
    stu = STU(x1_test_tensor, x2_test_tensor, x3_test_tensor, num_class, layers_size, latent_dim, batch_size)
    stu.load_weights(':/brc_stu' )
    _, _, _, _, _, _,_ , _, _,pred_labels_score = stu(x1_test_tensor, x2_test_tensor, x3_test_tensor)
    
    
    test_pred = np.argmax(pred_labels_score , axis=1)
    
    balanced_accuracy= balanced_accuracy_score(y_test, test_pred)
    
     
    f1 = f1_score(y_test , test_pred )
    
    precison = precision_score(y_test, test_pred)

    
    recall = recall_score(y_test, test_pred)
    
    
    auc_kd_svae_vcdn = roc_auc_score(y_test, pred_labels_score[:,1])
    
    
    acc = accuracy_score(y_test , test_pred)
    
    
    
    return f1 , auc_kd_svae_vcdn , pred_labels_score[:,1] , balanced_accuracy, precison, recall , 
    
     
    # f1_macro = f1_score(y_test , test_pred , average='macro')
    # print(f" f1_macro score is : {f1_macro}")
# # =============================================================================


f1, auc_kd_svae_vcdn, kd_svae_vcdn_preds , balanced_accuracy, precision, recall , acc = testing_stu(x1_test_tensor, x2_test_tensor, x3_test_tensor, num_class, stu_layers_size, stu_latent_dim, y1_test, y_test_cat, stu_batch_size)






