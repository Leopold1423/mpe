## MPE
This repository contains the implementation of VLDB 2025 submission paper:
Mixed-Precision Embeddings for Large-Scale Recommendation Models. 


### Data Preprocessing
Please (1) download the [Avazu](https://www.kaggle.com/competitions/avazu-ctr-prediction), [Criteo](https://www.kaggle.com/competitions/criteo-display-ad-challenge), [KDD12](https://www.kaggle.com/competitions/kddcup2012-track2) datasets; (2) place the training data as follows; (3) run the corresponding shells to preprocess the datasets; (4) run 'python ./dataloader/feat_cnt.py' to count the feature frequencies.
```
├── dataprocess
    ├── avazu.sh
    ├── avazu_new
        ├── train.csv
    ├── criteo.sh    
    ├── criteo_new
        ├── train.txt
    ├── kdd12.sh
    ├── kdd12_new
        ├── training.txt
```

### Run Experiments
The 'run' folder contains scripts for testing all methods. Note that the corresponding hyperparameters in the scripts are the optimal configurations. 

```
./run/12-avazu-optfp.sh;
./run/22-criteo-optfp.sh;
./run/32-kdd12-optfp.sh;
```

### Hyperparameters
|          | Avazu                               | Criteo                              | KDD12                               |
| :------- | :---------------------------------- | :---------------------------------- | :---------------------------------- |
| Backbone | lr=1e-3, l2=0.0                     | lr=1e-3, l2=3e-6                    | lr=1e-3, l2=0.0                     |
| QR-Trick | qr_ratio=2                          | qr_ratio=2                          | qr_ratio=2                          |
| PEP      | pep_init=-11                        | pep_init=-11                        | pep_init=-11                        |
| OptFS    | tau=2e-2, optfs_l1=1.75e-10         | tau=1e-3, optfs_l1=1e-8             | tau=1e-2, optfs_l1=1e-9             |
| ALPT     | bit=8, lr_alpha=1e-6                | bit=8, lr_alpha=1e-6                | bit=8, lr_alpha=1e-6                |
| LSQ+     | bit=6, lr_alpha=1e-3                | bit=6, lr_alpha=1e-3                | bit=6, lr_alpha=1e-3                |
| MPE      | $g$=128, $\tau$=3e-3, $\gamma$=2e-6 | $g$=128, $\tau$=3e-3, $\gamma$=3e-4 | $g$=128, $\tau$=3e-3, $\gamma$=3e-6 |gang