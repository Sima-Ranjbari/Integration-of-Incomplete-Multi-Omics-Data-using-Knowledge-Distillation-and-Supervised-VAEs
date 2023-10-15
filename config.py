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



#### hyperparameters #####

layers_size_te1=[512 ,256 ,128] 
layers_size_te2=[1024,512,128] 
stu_layers_size=[512,256,64]



latent_dim_te1 =20
latent_dim_te2 =30
stu_latent_dim=10


num_class=2
num_view=3
te_batch_size=20
stu_batch_size = int(preprocess.x1_c_len/8)




regulizer= 'l1_l2'
initializer = tf.keras.initializers.GlorotNormal()
w_constrain = tf.keras.constraints.MinMaxNorm(min_value=0, max_value=5)
b_constrain = tf.keras.constraints.MinMaxNorm(min_value=0, max_value=0.99)




