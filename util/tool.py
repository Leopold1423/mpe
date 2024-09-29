import os
import sys
sys.path.append('.')
sys.path.append('..')
import json
import torch
import numpy as np
from datetime import datetime
from dataloader.tfloader import CriteoLoader, AvazuLoader, KDD12Loader


def get_device(gpu):
    device= torch.device('cpu')
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda:'+str(gpu))
    print("training on", device)
    return device

def get_data_loader(dataset):
    work_space = "/home/lishiwei/ctr_embedding/"
    if dataset == "criteo_2":
        data_config = {
            "dataset": "criteo_2",
            "data_path": work_space+"dataprocess/criteo_new/threshold_2/",
            "feature_num": 6780382,
            "field_num": 39,
            "sample_nums": {
                "train": 36672494,
                "valid": 4584061,
                "test": 4584062,
            },
        }
        data_loader = CriteoLoader(data_config["data_path"])
    elif dataset == "avazu_2":
        data_config = {
            "dataset": "avazu_2",
            "data_path": work_space+"dataprocess/avazu_new/threshold_2/",
            "feature_num": 4428293,
            "field_num": 24,
            "sample_nums": {
                "train": 32343174,
                "valid": 4042897,
                "test": 4042896,
            },
        }
        data_loader = AvazuLoader(data_config["data_path"])
    elif dataset == "kdd12_2":
        data_config = {
            "dataset": "kdd12_2",
            "data_path": work_space+"dataprocess/kdd12_new/threshold_2/",
            "feature_num": 35970485,
            "field_num": 11,
            "sample_nums": {
                "train": 119711284,
                "valid": 14963910,
                "test": 14963911,
            },
        }
        data_loader = KDD12Loader(data_config["data_path"])
    else:
        raise ValueError("not support dataset")
    return data_loader, data_config

def get_optimizer(network, params):
    backbone_params = []
    lsq_alpha_params = []
    optfp_gamma_params = []
    pep_threshold_params = []
    optfs_mask_params = []

    for name, param in network.named_parameters():
        if "embedding" in name:
            if "gamma" in name:
                optfp_gamma_params.append(param)
            elif "alpha" in name or "beta" in name:
                lsq_alpha_params.append(param)
            elif "threshold" in name:
                pep_threshold_params.append(param)
            elif "mask" in name:
                optfs_mask_params.append(param)
            else:
                backbone_params.append(param)
        else:
            backbone_params.append(param)

    backbone_group = {'params': backbone_params, 'lr': params['lr'], 'weight_decay': params["l2"]}
    lsq_alpha_group = {'params': lsq_alpha_params, 'lr': params['lr_alpha']}
    pep_threshold_group = {'params': pep_threshold_params, 'lr': params['lr']}
    optfs_mask_group = {'params': optfs_mask_params, 'lr': params['lr']}
    optfp_gamma_group = {'params': optfp_gamma_params, 'lr': params['lr']}

    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD([backbone_group, lsq_alpha_group, pep_threshold_group, optfs_mask_group, optfp_gamma_group])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam([backbone_group, lsq_alpha_group, pep_threshold_group, optfs_mask_group, optfp_gamma_group])
    else:
        raise ValueError("not support optimizer")
    return optimizer

def save_json_record(json_dir, data, name="record"):
    json_path = os.path.join(json_dir, name+".json")
    records = json.load(open(json_path, 'r')) if os.path.exists(json_path) else {}  # reuse records
    time_string = datetime.now().strftime("%m-%d %H:%M:%S")
    index, added_key = 0, time_string
    while True:     # add a number tag for the same time key
        if added_key in records.keys():
            index += 1
            added_key = time_string + " ({})".format(index)
        else:
            break
    records[added_key] = data
    with open(json_path, 'w+') as f:
        json.dump(records, f, indent=4)

    analyzed = {}       # calculate avg, std, times
    analyzed["times"] = len(records.keys())
    all_records, avg_records, std_records = {}, {}, {}
    for record in records.values():
        for key, value in record.items():
            if key not in all_records:
                all_records[key] = []
            all_records[key].append(value)
    for key in all_records:
        avg_records[key] = np.array(all_records[key]).mean()
        std_records[key] = np.array(all_records[key]).std()

    analyzed["avg"] = avg_records
    analyzed["std"] = std_records
    analyzed["all"] = all_records

    json_path = os.path.join(json_dir, name+"_avg.json")
    with open(json_path, 'w+') as f:
        json.dump(analyzed, f, indent=4)


if __name__ == "__main__":
    print("done")