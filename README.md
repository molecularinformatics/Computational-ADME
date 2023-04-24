# Prospective Validation of Machine Learning Algorithms for ADME Prediction: An Industrial Perspective

Absorption, distribution, metabolism and excretion (ADME), which collectively define the concentration profile of a drug at the site of action, is of critical importance to the success of a drug candidate. Recent advances in machine learning (ML) algorithms and the availability of larger proprietary as well as public ADME data sets have generated renewed interest within the academic and pharmaceutical science communities in predicting pharmacokinetic and physicochemical endpoints in early drug discovery. In this study, we collected 120 internal prospective datasets over 20 months across 6 ADME in vitro endpoints: human and rat liver microsomal stability, MDR1-MDCK efflux ratio, solubility, and human and rat plasma protein binding. A variety of machine learning algorithms in combination with different molecular representations were evaluated. Our results suggest that Gradient Boosting Decision Tree and deep learning models consistently outperformed random forest over time. We also observed better performance when models were retrained on a fixed schedule and the more frequent retraining generally resulted in increased accuracy while hyperparameters tuning only improved the prospective predictions marginally.

This repository contains code and data for building and validating ML/DL models in silico ADME prediction as described in the paper of "Prospective Validation of Machine Learning Algorithms for ADME Prediction: An Industrial Perspective."

## Requirements
1. python 3.5
2. Scikit-Learn 0.20.3
3. RDKit 2018.09.3.0
4. XGBoost 0.82
5. LightGBM 2.23
6. DeepChem 2.1.0
7. ChemProp 1.5.2

## Data

To benefit the broader computational chemistry community and improve the quality and diversity of public-domain ADME data sets we have disclosed a collection of 3521 diverse compounds selected
from commercially available compound libraries (i.e. Enamine, eMolecules, WuXi LabNetwork, Mcule) and tested them against our internal six ADME in vitro assays described in this study using
the same experimental conditions as of our in-house datasets. 

The file "ADME_public_set_3521.csv" contains the compound Smiles, vendor and vendor ID, and the experimental log(properties) for six endpoints: HLM, RLM, Solubility, MDR1-MDCK ER, hPPB, and rPPB

## Codes for model training & validation
### Machine Learning Models (including Random Forest, SVM, XGBoost, LightGBM, Lasso)

The folder "ML" contains individual endpoint sdf files and the code "ADME_ML_public.py" for training and validating all Machine learning models investigated in the paper.

Take a LightGBM model trained on HLM dataset as an example. 

To train a model, you need to run:
```
python ADME_ML_public.py ADME_HLM.sdf -p 'LOG HLM_CLint (mL/min/kg)' -d FCFP4_rdMolDes -m LightGBM -w build 
```

To validate a model, run:
```
python ADME_ML_public.py ADME_HLM.sdf -p 'LOG HLM_CLint (mL/min/kg)' -d FCFP4_rdMolDes -m LightGBM -w validation
```

To do hyperparameter tuning, run:
```
python ADME_ML_public.py ADME_HLM.sdf -p 'LOG HLM_CLint (mL/min/kg)' -d FCFP4_rdMolDes -m LightGBM -w optimization 
```

### MPNN Models 
The folder "MPNN" contains individual training and test set CSV files and the code "MPNN_public.py" for training and validating message-passing neural network models investigated in the paper.

Take training MPNN models on HLM dataset as an example. 

To train and validate MPNN models with default hyperparameters, you need to run:
```
python MPNN_public.py ADME_HLM_train.csv ADME_HLM_test.csv default
```

To train and validate MPNN models with hyperparameters tuning, you need to run:
```
python MPNN_public.py ADME_HLM_train.csv ADME_HLM_test.csv hyperopt
```

Two models with either the representation of molecular graph alone (MPNN1) or the hybrid representation of a molecular graph and RDKit descriptors (MPNN2) will be built and validated in the codes. 


### FCNN Models 

The folder "FCNN" contains individual training and test set CSV files with precalculated features and the code "FCNN_public.py" for training and validating all fully-connected neural network models investigated in the paper.

Take training FCNN models on HLM dataset as an example. 

To train and validate all FCNN models, you need to run:
```
python FCNN_public.py ADME_HLM_train_feat.csv ADME_HLM_test_feat.csv
```


