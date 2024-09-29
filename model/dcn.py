import torch
import torch.nn as nn
from model.embedding import Embedding
import time

class MLP(nn.Module):
    def __init__(self, input_dim, mlp_dims, dropout, use_bn=False):
        super().__init__()
        layers = list()
        for out_dim in mlp_dims:
            layers.append(nn.Linear(input_dim, out_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = out_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class DNN(nn.Module):
    def __init__(self, data_config, model_config):
        super(DNN, self).__init__()
        self.emb_dim = model_config['emb_dim']
        self.feature_num = data_config['feature_num']
        self.field_num = data_config['field_num']
        self.mlp_dims = model_config['mlp_dims']
        self.dropout = model_config['mlp_dropout']
        self.use_bn = model_config['use_bn']
        self.embed_output_dim = self.field_num * self.emb_dim

        self.embedding = Embedding(data_config['feature_num'], model_config['emb_dim'], model_config['emb_init'])
        self.mlp = MLP(self.embed_output_dim, self.mlp_dims, dropout=self.dropout, use_bn=self.use_bn)
        self.predict_dense = nn.Linear(self.mlp_dims[-1], 1)

    def forward(self, x):
        xv = self.embedding(x)
        dnn_out = self.mlp.forward(xv.view(-1, self.embed_output_dim))
        score = self.predict_dense(dnn_out)
        score = score.squeeze(1)
        return score

class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = nn.ModuleList([nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)])

    def forward(self, x):
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x

class DCN(nn.Module):
    def __init__(self, data_config, model_config):
        super(DCN, self).__init__()
        self.emb_dim = model_config['emb_dim']
        self.feature_num = data_config['feature_num']
        self.field_num = data_config['field_num']
        self.mlp_dims = model_config['mlp_dims']
        self.dropout = model_config['mlp_dropout']
        self.use_bn = model_config['use_bn']
        self.embed_output_dim = self.field_num * self.emb_dim

        self.embedding = Embedding(data_config['feature_num'], model_config['emb_dim'], model_config['emb_init'])
        self.mlp = MLP(self.embed_output_dim, self.mlp_dims, dropout=self.dropout, use_bn=self.use_bn)
        self.cross_net = CrossNetwork(input_dim=self.embed_output_dim, num_layers=len(self.mlp_dims))
        self.predict_dense = nn.Linear(self.mlp_dims[-1] + self.embed_output_dim, 1)

    def forward(self, x):
        xv = self.embedding(x)
        dnn_out = self.mlp.forward(xv.view(-1, self.embed_output_dim))
        cross_out = self.cross_net(xv.view(-1, self.embed_output_dim))
        out = torch.cat([dnn_out, cross_out], dim=1)
        score = self.predict_dense(out)
        score = score.squeeze(1)
        return score
