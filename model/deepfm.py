import torch
import torch.nn as nn
from model.embedding import Embedding


class FeaturesLinear(nn.Module):
    def __init__(self, vocab_size, output_dim=1):
        super().__init__()
        self.fc = nn.Embedding(vocab_size, output_dim)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return torch.sum(self.fc(x), dim=1) + self.bias

class FactorizationMachine(nn.Module):
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum
    
    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, mlp_dims, dropout, output_layer=True, use_bn=False):
        super().__init__()
        layers = list()
        for out_dim in mlp_dims:
            layers.append(nn.Linear(input_dim, out_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = out_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class FM(nn.Module):
    """Factorization Machines"""
    def __init__(self, data_config, model_config):
        super(FM, self).__init__()
        self.emb_dim = model_config['emb_dim']
        self.field_num = data_config['field_num']
        self.feature_num = data_config['feature_num']

        self.embedding = Embedding(data_config['feature_num'], model_config['emb_dim'], model_config['emb_init'])
        # self.linear = FeaturesLinear(self.feature_num)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        # linear_score = self.linear.forward(x)
        xv = self.embedding(x)
        fm_score = self.fm.forward(xv)
        score = fm_score
        return score.squeeze(1)

class DeepFM(FM):
    def __init__(self, data_config, model_config):
        super(DeepFM, self).__init__(data_config, model_config)
        self.embed_output_dim = self.field_num * self.emb_dim
        self.mlp_dims = model_config['mlp_dims']
        self.dropout = model_config['mlp_dropout']
        self.use_bn = model_config['use_bn']
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, self.mlp_dims, dropout=self.dropout, use_bn=self.use_bn)

    def forward(self, x):
        # linear_score = self.linear.forward(x)
        xv = self.embedding(x)
        fm_score = self.fm.forward(xv)
        dnn_score = self.mlp.forward(xv.view(-1, self.embed_output_dim))
        # score = linear_score + fm_score + dnn_score
        score = fm_score + dnn_score
        return score.squeeze(1)

