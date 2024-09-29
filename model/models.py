from model.ipnn import IPNN
from model.dcn import DNN, DCN
from model.deepfm import DeepFM
from model.embedding import replace_embedding


def get_model(model_config, data_config):
    if model_config["model"] == "dnn":
        model = DNN(data_config, model_config)
    elif model_config["model"] == "dcn":
        model = DCN(data_config, model_config)
    elif model_config["model"] == "deepfm":
        model = DeepFM(data_config, model_config)
    elif model_config["model"] == "ipnn":
        model = IPNN(data_config, model_config)
    else:
        raise ValueError("Invalid model type: {}".format(model_config["model"]))
    replace_embedding(model, model_config, data_config)
    return model