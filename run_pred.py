import os,sys,time
from scipy.stats import pearsonr
import pandas as pd

# python run_pred.py trainSet_HLM.csv testSet_HLM.csv mode
# mode == default, hyperopt

trainset = sys.argv[1]
testset = sys.argv[2]
mode = sys.argv[3]

models = []
r_train_list = []
r_test_list = []
train_time = []

if mode == 'default':
    # run script using MolGraph
    t0 =time.time()
    os.system('python /home/cfang/Downloads/chemprop-master/train.py --data_path %s --dataset_type regression --metric r2 --save_dir %s_checkpoints' %(trainset,sys.argv[1][9:-4]))
    os.system('python /home/cfang/Downloads/chemprop-master/predict.py --test_path %s --checkpoint_dir %s_checkpoints --preds_path %s_train_preds.csv' %(trainset,sys.argv[1][9:-4],sys.argv[1][9:-4]))
    os.system('python /home/cfang/Downloads/chemprop-master/predict.py --test_path %s --checkpoint_dir %s_checkpoints --preds_path %s_test_preds.csv' %(testset,sys.argv[1][9:-4],sys.argv[1][9:-4]))
    t1 = time.time()
    train_t = t1-t0
    train_pred = pd.read_csv('%s_train_preds.csv' %sys.argv[1][9:-4])
    train_true = pd.read_csv(trainset)
    r_train = pearsonr(train_pred['activity'],train_true['activity'])[0]
    
    test_pred = pd.read_csv('%s_test_preds.csv' %sys.argv[1][9:-4])
    test_true = pd.read_csv(testset)
    r_test = pearsonr(test_pred['activity'],test_true['activity'])[0]
    
    models.append('MolGraph')
    r_train_list.append(r_train)
    r_test_list.append(r_test)
    train_time.append(train_t)

    # run script using MolGraph + RDKit descriptor
    t0 =time.time()
    os.system('python /home/cfang/Downloads/chemprop-master/train.py --data_path %s --dataset_type regression --features_generator rdkit_2d_normalized --no_features_scaling --metric r2 --save_dir %s_rdkit_checkpoints' %(trainset,sys.argv[1][9:-4]))
    os.system('python /home/cfang/Downloads/chemprop-master/predict.py --test_path %s --features_generator rdkit_2d_normalized --no_features_scaling --checkpoint_dir %s_rdkit_checkpoints --preds_path %s_rdkit_train_preds.csv' %(trainset,sys.argv[1][9:-4],sys.argv[1][9:-4]))
    os.system('python /home/cfang/Downloads/chemprop-master/predict.py --test_path %s --features_generator rdkit_2d_normalized --no_features_scaling --checkpoint_dir %s_rdkit_checkpoints --preds_path %s_rdkit_test_preds.csv' %(testset,sys.argv[1][9:-4],sys.argv[1][9:-4]))
    t1 = time.time()
    train_t = t1-t0
    train_pred_rdkit = pd.read_csv('%s_rdkit_train_preds.csv' %sys.argv[1][9:-4])
    train_true = pd.read_csv(trainset)
    r_train_rdkit = pearsonr(train_pred_rdkit['activity'],train_true['activity'])[0]
    
    test_pred_rdkit = pd.read_csv('%s_rdkit_test_preds.csv' %sys.argv[1][9:-4])
    test_true = pd.read_csv(testset)
    r_test_rdkit = pearsonr(test_pred_rdkit['activity'],test_true['activity'])[0]
    
    models.append('MolGraph_RDKIT')
    r_train_list.append(r_train)
    r_test_list.append(r_test)
    train_time.append(train_t)    

elif mode == 'hyperopt':
    # run scripts using MolGrah + HptOpt
    t0 =time.time()
    ## hypopt
    os.system('python /home/cfang/Downloads/chemprop-master/hyperparameter_optimization.py --data_path %s --dataset_type regression --metric r2 --num_iters 20 --config_save_path %s_hp.json' %(trainset,sys.argv[1][9:-4]))
    ## train and prediction
    os.system('python /home/cfang/Downloads/chemprop-master/train.py --data_path %s --dataset_type regression --metric r2 --config_path %s_hp.json --save_dir %s_hp_checkpoints' %(trainset,sys.argv[1][9:-4],sys.argv[1][9:-4]))
    os.system('python /home/cfang/Downloads/chemprop-master/predict.py --test_path %s --checkpoint_dir %s_hp_checkpoints --preds_path %s_hp_train_preds.csv' %(trainset,sys.argv[1][9:-4],sys.argv[1][9:-4]))
    os.system('python /home/cfang/Downloads/chemprop-master/predict.py --test_path %s --checkpoint_dir %s_hp_checkpoints --preds_path %s_hp_test_preds.csv' %(testset,sys.argv[1][9:-4],sys.argv[1][9:-4]))
    t1 = time.time()
    train_t = t1-t0
    train_pred = pd.read_csv('%s_hp_train_preds.csv' %sys.argv[1][9:-4])
    train_true = pd.read_csv(trainset)
    r_train = pearsonr(train_pred['activity'],train_true['activity'])[0]
    
    test_pred = pd.read_csv('%s_hp_test_preds.csv' %sys.argv[1][9:-4])
    test_true = pd.read_csv(testset)
    r_test = pearsonr(test_pred['activity'],test_true['activity'])[0]
    
    models.append('MolGraph_hyperopt')
    r_train_list.append(r_train)
    r_test_list.append(r_test)
    train_time.append(train_t)
    
    # run scripts using MolGrah + rdkit + HptOpt
    t0 =time.time()
    ## pre-compute rdkit features
    os.system('python /home/cfang/Downloads/chemprop-master/scripts/save_features.py --data_path %s --features_generator rdkit_2d_normalized --save_path %s_rdkit.npz --sequential' %(trainset,sys.argv[1][9:-4]))
    ## hypopt
    os.system('python /home/cfang/Downloads/chemprop-master/hyperparameter_optimization.py --data_path %s --dataset_type regression --metric r2 --num_iters 20 --features_path %s_rdkit.npz --config_save_path %s_hp_rdkit.json' %(trainset,sys.argv[1][9:-4],sys.argv[1][9:-4]))
    ## train and prediction  
    os.system('python /home/cfang/Downloads/chemprop-master/train.py --data_path %s --dataset_type regression --features_path %s_rdkit.npz --no_features_scaling --metric r2 --config_path %s_hp_rdkit.json --save_dir %s_hp_rdkit_checkpoints' %(trainset,sys.argv[1][9:-4], sys.argv[1][9:-4], sys.argv[1][9:-4]))
    os.system('python /home/cfang/Downloads/chemprop-master/predict.py --test_path %s --features_generator rdkit_2d_normalized --no_features_scaling --checkpoint_dir %s_hp_rdkit_checkpoints --preds_path %s_hp_rdkit_train_preds.csv' %(trainset,sys.argv[1][9:-4],sys.argv[1][9:-4]))
    os.system('python /home/cfang/Downloads/chemprop-master/predict.py --test_path %s --features_generator rdkit_2d_normalized --no_features_scaling --checkpoint_dir %s_hp_rdkit_checkpoints --preds_path %s_hp_rdkit_test_preds.csv' %(testset,sys.argv[1][9:-4],sys.argv[1][9:-4]))  
    
    t1 = time.time()
    train_t = t1-t0
    train_pred = pd.read_csv('%s_hp_rdkit_train_preds.csv' %sys.argv[1][9:-4])
    train_true = pd.read_csv(trainset)
    r_train = pearsonr(train_pred['activity'],train_true['activity'])[0]
    
    test_pred = pd.read_csv('%s_hp_rdkit_test_preds.csv' %sys.argv[1][9:-4])
    test_true = pd.read_csv(testset)
    r_test = pearsonr(test_pred['activity'],test_true['activity'])[0]
    
    models.append('MolGraph_RDKIT_hyperopt')
    r_train_list.append(r_train)
    r_test_list.append(r_test)
    train_time.append(train_t)    

df = pd.DataFrame({'model':models, 'r_train': r_train_list, 'r_test':r_test_list, 'time':train_time})
df = df[['model','r_train','r_test','time']]
df.to_csv('%s_%s_result.csv' % (sys.argv[1][9:-4],mode), sep=",",index=False)





