from train.train_pep import Trainer_PEP
from train.train_alpt import Trainer_ALPT
from train.train_optfs import Trainer_OptFS
from train.train_optfp import Trainer_OptFP
from train.train_backbone import Trainer_BackBone

trainer_dict = {
    "fp32": Trainer_BackBone,
    "qr": Trainer_BackBone,
    "lsq": Trainer_BackBone,
    "lsq_plus": Trainer_BackBone,
    "alpt": Trainer_ALPT,
    "pep": Trainer_PEP,
    "optfs": Trainer_OptFS,
    "optfp": Trainer_OptFP,
}

def get_trainer(data_loader, data_config, model_config, train_config):
    if model_config["emb_type"] in trainer_dict.keys():
        return trainer_dict[model_config["emb_type"]](data_loader, data_config, model_config, train_config)
    else:
        raise ValueError("Invalid model type: {}".format(model_config["emb_type"]))