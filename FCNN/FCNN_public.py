import pandas as pd
import os,sys,time
import numpy as np
import shutil
import deepchem as dc
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.utils import shuffle
import math
#from data_features import user_specified_features

# How to run the script
# python FCNN_public.py train.csv test.csv

train_file = sys.argv[1]
test_file = sys.argv[2]

# Training set
train_df = pd.read_csv(train_file)
Y_train= train_df["activity"]
X_train = train_df.copy()
X_train.drop(["activity","ID"],axis=1,inplace=True)

# Test Set
test_df = pd.read_csv(test_file)
Y_test= test_df["activity"]
X_test = test_df.copy()
X_test.drop(["activity","ID"],axis=1,inplace=True)

# Robust_Scaler
scaler = RobustScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# get the user_sepcified_features 
user_specified_features = list(train_df.columns)
user_specified_features.remove("ID")
user_specified_features.remove("activity")

# save scaled features into new files
idx_train = list(train_df.index)
X_train = pd.DataFrame(X_train,index=idx_train, columns=user_specified_features)
Y_train = pd.DataFrame(Y_train,index=idx_train, columns=["activity"])
train_set = pd.concat([X_train,Y_train],axis=1)
train_set['ID'] = train_df['ID']

idx_test = list(test_df.index)
X_test = pd.DataFrame(X_test,index=idx_test, columns=user_specified_features)
Y_test = pd.DataFrame(Y_test,index=idx_test, columns=["activity"])
test_set = pd.concat([X_test,Y_test],axis=1)
train_set['ID'] = test_df['ID']

train_set.to_csv('%s_new.csv' %sys.argv[1][:-4],header=True,index=False)
test_set.to_csv('%s_new.csv' %sys.argv[2][:-4],header=True,index=False)
train_file_new = '%s_new.csv' %sys.argv[1][:-4]
test_file_new = '%s_new.csv' %sys.argv[2][:-4]


def load_train(train_file_new,test_file_new):
    tasks = ["activity"]
    featurizer_func = dc.feat.UserDefinedFeaturizer(user_specified_features)
    loader = dc.data.UserCSVLoader(tasks=tasks, smiles_field=None, id_field="ID",featurizer=featurizer_func)
    train_set = loader.featurize(train_file_new)
    test_set = loader.featurize(test_file_new)

    return tasks, train_set,test_set

np.random.seed(123)
tasks, train_set,test_set= load_train(train_file_new,test_file_new)

# Fit models

metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)

layer_list = [[512,256,64],[200,100,50],[2000,1000,500],[4000,2000,1000,1000],[1000,500],[2000,1000]]
dropouts_list = [[0.25,0.25,0.10],[0.25,0.25,0.10],[0.25,0.25,0.10],[0.25,0.25,0.25,0.10],[0.25,0.10],[0.25,0.10]]
#nb_epoch_list = [25, 50, 75, 100]
nb_epoch_list = [50]

models = []
r_train_list = []
r_test_list = []
train_time = []

for nb_epoch in nb_epoch_list:
    for i in range(len(layer_list)):
        t0 = time.time()      
        layer_sizes = layer_list[i]
        dropouts = dropouts_list[i]
        n_layers = len(layer_list[i])
        #f.write("The layer size are %s \n" % layer_sizes)
        #f.write("The dropout layers are %s \n" % dropouts)
        model = dc.models.MultitaskRegressor(
            len(tasks),
            len(user_specified_features),
            layer_sizes=layer_sizes,dropouts=dropouts,
            weight_init_stddevs = [0.02]*n_layers,
            bias_init_consts=[1.0]*n_layers,
        weight_decay_penalty = 0.0004,          # or penalty
        weight_decay_penalty_type = "l2",       # or penalty_type
        init="glorot_uniform",
        learning_rate=0.001,
        batch_size=128,
        batchnorm=True,
        #verbosity="high",
        activation='relu',                 
        optimizer='adam',       
        momentum=0.9,
        seed = 5758)
        
        model.fit(train_set,nb_epoch=nb_epoch)
        train_scores = model.evaluate(train_set, [metric])
        test_scores = model.evaluate(test_set, [metric])
        r_train = math.sqrt(train_scores.get('pearson_r2_score'))
        r_test = math.sqrt(test_scores.get('pearson_r2_score'))
        
        t1 = time.time()
        t = t1-t0  
        
        models.append(layer_sizes)
        r_train_list.append(r_train)
        r_test_list.append(r_test)
        train_time.append(t)           
                            
df = pd.DataFrame({'model':models, 'r_train': r_train_list, 'r_test':r_test_list, 'time':train_time})
df = df[['model','r_train','r_test','time']]
df.to_csv('%s_result.csv' % sys.argv[1][9:-4], sep=",",index=False)









