import os,sys,time
from scipy.stats import pearsonr
import pandas as pd

# python MPNN_public.py ADME_HLM_train.csv ADME_HLM_test.csv default
# mode == default, hyperopt

trainset = sys.argv[1]
testset = sys.argv[2]
mode = sys.argv[3]

models = []
r_cv_list = []
r_test_list = []


# 1MPNN1

if mode == 'default':
    
    os.system('chemprop_train --data_path %s --dataset_type regression --metric r2 --save_dir %s_default_checkpoints' %(trainset,sys.argv[1][5:-10]))
    os.system('chemprop_predict --test_path %s --checkpoint_dir %s_default_checkpoints --preds_path %s_default_preds.csv' %(testset,sys.argv[1][5:-10],sys.argv[1][5:-10]))
    
    test_pred = pd.read_csv('%s_default_preds.csv' %sys.argv[1][5:-10])
    test_true = pd.read_csv(testset)
    r_test = pearsonr(test_pred['activity'],test_true['activity'])[0]
    
    models.append('MPNN1_default')
    r_test_list.append(r_test)


# 2. MPNN2

    os.system('chemprop_train --data_path %s --dataset_type regression --features_generator rdkit_2d_normalized --no_features_scaling --metric r2 --save_dir %s_default_rdkit_checkpointsto' %(trainset,sys.argv[1][5:-10]))
    os.system('chemprop_predict --test_path %s --features_generator rdkit_2d_normalized --no_features_scaling --checkpoint_dir %s_default_rdkit_checkpoints --preds_path %s_default_rdkit_preds.csv' %(testset,sys.argv[1][5:-10],sys.argv[1][5:-10]))
    
    test_pred_rdkit = pd.read_csv('%s_default_rdkit_preds.csv'  %sys.argv[1][5:-10])
    test_true = pd.read_csv(testset)
    r_test_rdkit = pearsonr(test_pred_rdkit['activity'],test_true['activity'])[0]
    
    models.append('MPNN2_default')
    r_test_list.append(r_test_rdkit) 

# 3. MPNN1-OPT

if mode == 'Opt':
    
    ## hypopt
    os.system('chemprop_hyperopt --data_path %s --dataset_type regression --metric r2 --num_iters 20 --config_save_path %s_hp.json' %(trainset,sys.argv[1][5:-10]))
    ## train and prediction
    os.system('chemprop_train --data_path %s --dataset_type regression --metric r2 --config_path %s_hp.json --save_dir %s_hp_checkpoints' %(trainset,sys.argv[1][5:-10],sys.argv[1][5:-10]))
    os.system('chemprop_predict --test_path %s --checkpoint_dir %s_hp_checkpoints --preds_path %s_hp_preds.csv' %(testset,sys.argv[1][5:-10],sys.argv[1][5:-10]))
    
    
    test_pred = pd.read_csv('%s_hp_preds.csv' %sys.argv[1][5:-10])
    test_true = pd.read_csv(testset)
    r_test = pearsonr(test_pred['activity'],test_true['activity'])[0]
    
    models.append('MPNN1_hyperopt')
    r_test_list.append(r_test)

    
    # 4. MPNN2-OPT
    
    # hypopt
    os.system('chemprop_hyperopt --data_path %s --dataset_type regression --metric r2 --num_iters 20 --features_generator rdkit_2d_normalized --no_features_scaling --config_save_path %s_hp_rdkit.json' %(trainset,sys.argv[1][5:-10]))
    ## train and prediction  
    os.system('chemprop_train --data_path %s --dataset_type regression  --features_generator rdkit_2d_normalized --no_features_scaling --metric r2 --config_path %s_hp_rdkit.json --save_dir %s_hp_rdkit_checkpoints' %(trainset,sys.argv[1][5:-10], sys.argv[1][5:-10]))
    os.system('chemprop_predict --test_path %s --features_generator rdkit_2d_normalized --no_features_scaling --checkpoint_dir %s_hp_rdkit_checkpoints --preds_path %s_hp_rdkit_preds.csv' %(testset,sys.argv[1][5:-10],sys.argv[1][5:-10]))  
    
    
    test_pred = pd.read_csv('%s_hp_rdkit_preds.csv' %sys.argv[1][5:-10])
    test_true = pd.read_csv(testset)
    r_test = pearsonr(test_pred['activity'],test_true['activity'])[0]
    
    models.append('MPNN2_hyperopt')
    r_test_list.append(r_test)


df = pd.DataFrame({'model':models, 'r_test':r_test_list})
df = df[['model','r_test']]
df.to_csv('%s_%s_result.csv' % (sys.argv[1][5:-10],mode), sep=",",index=False)




