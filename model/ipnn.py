import torch
import torch.nn as nn
from model.embedding import Embedding
from model.deepfm import MultiLayerPerceptron


class IPNN(torch.nn.Module):
    def __init__(self, data_config, model_config):
        super(IPNN, self).__init__()
        self.emb_dim = model_config['emb_dim']
        self.feature_num = data_config['feature_num']
        self.field_num = data_config['field_num']
        self.mlp_dims = model_config['mlp_dims']
        self.dropout = model_config['mlp_dropout']
        self.use_bn = model_config['use_bn']
        
        self.embed_output_dim = self.field_num * self.emb_dim
        self.product_output_dim = int(self.field_num * (self.field_num - 1) / 2)
        self.dnn_input_dim = self.embed_output_dim + self.product_output_dim

        self.embedding = Embedding(data_config['feature_num'], model_config['emb_dim'], model_config['emb_init'])
        self.mlp = MultiLayerPerceptron(self.dnn_input_dim, self.mlp_dims, dropout=self.dropout, use_bn=self.use_bn)       

        rows, cols = [], []
        for i in range(self.field_num):
            for j in range(i+1, self.field_num):
                rows.append(i)
                cols.append(j)
        self.rows = nn.Parameter(torch.tensor(rows), requires_grad=False)
        self.cols = nn.Parameter(torch.tensor(cols), requires_grad=False)
    
    def calc_product(self, xe):
        batch_size = xe.shape[0]
        trans = torch.transpose(xe, 1, 2)
        gather_rows = torch.gather(trans, 2, self.rows.expand(batch_size, trans.shape[1], self.rows.shape[0]))
        gather_cols = torch.gather(trans, 2, self.cols.expand(batch_size, trans.shape[1], self.rows.shape[0]))
        p = torch.transpose(gather_rows, 1, 2)
        q = torch.transpose(gather_cols, 1, 2)
        product_embedding = torch.mul(p, q)
        product_embedding = torch.sum(product_embedding, 2)
        return product_embedding
    
    def forward(self, x):
        xv = self.embedding(x)
        product = self.calc_product(xv)
        xv = xv.view(-1, self.embed_output_dim)
        xe = torch.cat((xv, product), 1)
        score = self.mlp.forward(xe)
        return score.squeeze(1)

