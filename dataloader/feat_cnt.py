import os
import sys
sys.path.append('.')
sys.path.append('..')
import torch
import pickle
from tqdm import tqdm
import tensorflow as tf
from util.tool import get_data_loader
tf.config.set_visible_devices([], 'GPU')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def feature_cnt(dataset="avazu_2", batch_size=int(1e5)):
    data_loader, data_config = get_data_loader(dataset)
    cnt = torch.zeros(data_config["feature_num"])

    dataloader = data_loader.get_data("train", batch_size=batch_size)
    steps = data_config["sample_nums"]["train"] // batch_size + 1
    for feature, label in tqdm(dataloader, total=steps, ncols=80):
        res = torch.unique(feature, return_counts=True)
        cnt[res[0]] += res[1]
    
    dataloader = data_loader.get_data("valid", batch_size=batch_size)
    steps = data_config["sample_nums"]["valid"] // batch_size + 1
    for feature, label in tqdm(dataloader, total=steps, ncols=80):
        res = torch.unique(feature, return_counts=True)
        cnt[res[0]] += res[1]
    
    dataloader = data_loader.get_data("test", batch_size=batch_size)
    steps = data_config["sample_nums"]["test"] // batch_size + 1
    for feature, label in tqdm(dataloader, total=steps, ncols=80):
        res = torch.unique(feature, return_counts=True)
        cnt[res[0]] += res[1]

    stats_path = "./dataprocess/{}/stats/feat_cnt.pkl".format(dataset.replace("_", "_new/threshold_"))
    with open(stats_path, 'wb') as fi:
        pickle.dump(cnt, fi)

feature_cnt(dataset="avazu_2", batch_size=int(1e5))
feature_cnt(dataset="criteo_2", batch_size=int(1e5))
feature_cnt(dataset="kdd12_2", batch_size=int(1e5))
