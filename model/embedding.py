import math
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, feature_num, emb_dim, emb_init="normal_0.003"):
        super().__init__()
        self.feature_num, self.emb_dim = feature_num, emb_dim
        self.weight = nn.Parameter(torch.randn(self.feature_num, self.emb_dim))
        if emb_init == "xavier":
            nn.init.xavier_uniform_(self.weight)
        elif "uniform" in emb_init:
            value = float(emb_init.split("_")[1])
            nn.init.uniform_(self.weight, a=-value, b=value)
        elif "normal" in emb_init:
            value = float(emb_init.split("_")[1])
            nn.init.normal_(self.weight, mean=0, std=value)

    def forward(self, x):
        return F.embedding(x, self.weight)

class QR_Embedding(nn.Module):
    def __init__(self, feature_num, emb_dim, emb_init="normal_0.003", config=None):
        super(QR_Embedding, self).__init__()
        self.feature_num, self.emb_dim = feature_num, emb_dim
        self.qr_ratio = config["qr_ratio"]
        self.embedding_q = torch.nn.Parameter(torch.randn(int(self.feature_num / self.qr_ratio)+2, self.emb_dim))
        self.embedding_r = torch.nn.Parameter(torch.randn(self.qr_ratio, self.emb_dim))
        nn.init.normal_(self.embedding_q, mean=0, std=float(emb_init.split("_")[1]))
        nn.init.normal_(self.embedding_r, mean=0, std=float(emb_init.split("_")[1]))

    def forward(self, x):
        x_q = (x / self.qr_ratio).long()
        x_r = torch.remainder(x, self.qr_ratio).long()
        embed_q = F.embedding(x_q, self.embedding_q)
        embed_r = F.embedding(x_r, self.embedding_r)
        return embed_q + embed_r

class lsq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, alpha, bit, grad_scale):
        low = -2 ** (bit - 1)
        high = 2 ** (bit - 1) - 1
        weight_alpha = weight / alpha
        quantized_weight = weight_alpha.round().clamp(low, high) * alpha

        ctx.save_for_backward(weight_alpha, alpha)
        ctx.other = low, high, grad_scale
        return quantized_weight

    @staticmethod
    def backward(ctx, dy):
        weight_alpha, alpha = ctx.saved_tensors
        low, high, grad_scale = ctx.other
        
        small = (weight_alpha < low).float()
        big = (weight_alpha > high).float()
        middle = 1.0 - small - big 

        dweight = middle * dy
        dalpha = ((small*low + big*high + middle*(-weight_alpha+weight_alpha.round()))*dy*grad_scale).sum()
        return dweight, dalpha, None, None, None

class Lsq_Embedding(Embedding):
    def __init__(self, feature_num, emb_dim, emb_init="normal_0.003", config=None):
        super().__init__(feature_num, emb_dim, emb_init)
        self.bit = config["bit"]
        self.grad_scale = torch.tensor(1 / (2**(self.bit-1))**0.5 / self.weight.numel()**0.5)
        self.alpha = nn.Parameter(self.weight.abs().mean()*2 / (2**(self.bit-1))**0.5)

    def forward(self, x):
        weight = F.embedding(x, self.weight)
        alpha = self.alpha.abs() + torch.tensor(1e-10)
        out = lsq.apply(weight, alpha, self.bit, self.grad_scale)
        return out

class lsq_plus(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, alpha, beta, bit, grad_scale):
        low = -2 ** (bit - 1)
        high = 2 ** (bit - 1) - 1
        weight_alpha = (weight - beta) / alpha
        quantized_weight = weight_alpha.round().clamp(low, high) * alpha + beta

        ctx.save_for_backward(weight_alpha, alpha)
        ctx.other = low, high, grad_scale
        return quantized_weight

    @staticmethod
    def backward(ctx, dy):
        weight_alpha, alpha = ctx.saved_tensors
        low, high, grad_scale = ctx.other

        small = (weight_alpha < low).float()
        big = (weight_alpha > high).float()
        middle = 1.0 - small - big 
        
        dweight = middle * dy

        dalpha = ((small * low + big * high + middle * (-weight_alpha + weight_alpha.round())) * dy * grad_scale).sum()
        dbeta = ((small + big) * dy * grad_scale).sum(dim=(0,1))
        return dweight, dalpha, dbeta, None, None, None
    
class Lsq_Plus_Embedding(Embedding):
    def __init__(self, feature_num, emb_dim, emb_init="normal_0.003", config=None):
        super().__init__(feature_num, emb_dim, emb_init)
        self.bit = config["bit"]
        
        self.grad_scale = torch.tensor(1 / (2**(self.bit-1))**0.5 / self.weight.numel()**0.5)
        self.alpha = nn.Parameter(self.weight.abs().mean()*2 / (2**(self.bit-1))**0.5)
        self.beta = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        weight = F.embedding(x, self.weight)
        alpha = self.alpha.abs() + torch.tensor(1e-10)
        out = lsq_plus.apply(weight, alpha, self.beta, self.bit, self.grad_scale)
        return out

class PEP_Embedding(Embedding):
    def __init__(self, feature_num, emb_dim, emb_init="normal_0.003", config=None):
        super().__init__(feature_num, emb_dim, emb_init)
        self.threshold = nn.Parameter(config["pep_init"] * torch.ones([self.emb_dim]))
        self.mask = nn.Parameter(torch.ones_like(self.weight), requires_grad=False)
        self.retraining = False
            
    def forward(self, x):
        weight = F.embedding(x, self.weight)
        if self.retraining:
            mask = F.embedding(x, self.mask)
            sparse_weight = weight * mask
        else:
            sparse_weight = self.soft_threshold(weight, self.threshold)
        return sparse_weight

    def soft_threshold(self, v, s):
        return torch.sign(v) * torch.relu(torch.abs(v) - torch.sigmoid(s))
        
    def get_sparsity(self):
        sparse_v = self.soft_threshold(self.weight.data.cpu(), self.threshold.data.cpu())
        non_zero_values = torch.nonzero(sparse_v).size(0)
        sparsity = 1 - (non_zero_values / self.weight.numel())
        return sparsity
    
    def get_mask(self):
        sparse_v = self.soft_threshold(self.weight.data.cpu(), self.threshold.data.cpu())
        return (sparse_v != 0).float() 

class OptFS_Embedding(Embedding):
    def __init__(self, feature_num, emb_dim, emb_init="normal_0.003", config=None):
        super().__init__(feature_num, emb_dim, emb_init)
        self.mask_init = torch.tensor(config["mask_init"])
        self.mask = nn.Parameter(torch.ones(self.feature_num, 1) * self.mask_init)
        self.tau = config["tau"]
        self.epoch, self.total_epoch = 1, 1     # to be set before training
        self.retraining = False
       
    def forward(self, x):
        weight = F.embedding(x, self.weight)
        mask = self.get_mask(x)
        return weight * mask
    
    def get_mask(self, x):
        mask = F.embedding(x, self.mask)
        if self.retraining:
            mask = (mask > 0).float().detach()
        else:
            tau = self.tau**(self.epoch/self.total_epoch)
            mask = torch.sigmoid(mask / tau)
            mask = mask / torch.sigmoid(self.mask_init)
        return mask

    def regularization(self):
        if self.retraining:
            return 0
        tau = self.tau**(self.epoch/self.total_epoch)
        return torch.sum(torch.sigmoid(self.mask / tau))
    
    def get_sparsity(self):
        return 1 - float((self.mask > 0).float().mean())

class alpt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, alpha, bit, grad_scale):
        low = -2 ** (bit - 1)
        high = 2 ** (bit - 1) - 1
        weight_alpha = weight / alpha
        quantized_weight = weight_alpha.round().clamp(low, high) * alpha

        ctx.save_for_backward(weight_alpha, alpha)
        ctx.other = low, high, grad_scale
        return quantized_weight

    @staticmethod
    def backward(ctx, dy):
        weight_alpha, alpha = ctx.saved_tensors
        low, high, grad_scale = ctx.other
        
        small = (weight_alpha < low).float()
        big = (weight_alpha > high).float()
        middle = 1.0 - small - big 

        dweight = middle * dy
        dalpha = ((small*low + big*high + middle*(-weight_alpha+weight_alpha.round()))*dy*grad_scale)
        return dweight, dalpha.sum(dim=-1).unsqueeze(-1), None, None, None

class ALPT_Embedding(Embedding):
    def __init__(self, feature_num, emb_dim, emb_init="normal_0.003", config=None):
        super().__init__(feature_num, emb_dim, emb_init)
        self.bit = config["bit"]
        self.grad_scale = torch.tensor(1 / (2**(self.bit-1))**0.5 / self.weight.numel()**0.5)
        self.alpha = nn.Parameter(self.weight.abs().mean()*2 / (2**(self.bit-1))**0.5)
        self.post_quantizer()
        self.order = 1

    def forward(self, x):
        weight = F.embedding(x, self.weight)
        if self.order == 1:
            output = weight
        elif self.order == 2:
            alpha = self.alpha.abs() + torch.tensor(1e-10)
            output = lsq.apply(weight, alpha, self.bit, self.grad_scale)
        return output

    def post_quantizer(self):
        with torch.no_grad():
            low = -2 ** (self.bit - 1)
            high = 2 ** (self.bit - 1) - 1
            alpha = self.alpha.abs() + torch.tensor(1e-10)
            self.weight.data = torch.floor(self.weight / self.alpha + torch.rand_like(self.weight))
            self.weight.data = self.weight.data.clamp(low, high) * alpha

def optfp_get_groups(group_num, group_size, feature_num, dataset):
    feature_group = torch.arange(group_num).unsqueeze(dim=-1) * torch.ones((group_num, group_size))
    feature_group = feature_group.reshape(-1)[:feature_num]
    path1 = "/home/lishiwei/ctr_embedding/dataprocess/"
    path2 = dataset.replace("_", "_new/threshold_")
    path3 = "/stats/feat_cnt.pkl"
    with open(path1+path2+path3, 'rb') as fi:
        feat_cnt = pickle.load(fi)
    feature_index = torch.sort(feat_cnt)[1]
    group_index = torch.ones(feature_num)
    group_index[feature_index] = feature_group

    feature_sortcnt = torch.sort(feat_cnt)[0]
    group_cnt = [feature_sortcnt[i*group_size: (i+1)*group_size].sum() for i in range(group_num-1)]
    group_cnt.append(feature_sortcnt[group_size*(group_num-1):].sum())
    group_cnt = torch.tensor(group_cnt)
    group_cnt = (group_cnt / group_cnt.min()).unsqueeze(-1).unsqueeze(-1)
    return group_index, group_cnt

class OptFP_Embedding(Embedding):
    def __init__(self, feature_num, emb_dim, emb_init="normal_0.003", dataset=None, config=None):
        super().__init__(feature_num, emb_dim, emb_init)
        # group
        self.group_num = int(config["group"])
        self.group_size = int(math.ceil(feature_num / self.group_num))
        group_index, group_cnt = optfp_get_groups(self.group_num, self.group_size, feature_num, dataset)
        self.group_index = nn.Parameter(group_index.long(), requires_grad=False)
        self.group_cnt = nn.Parameter(group_cnt, requires_grad=False)
        # quantization
        bitsets = torch.tensor([int(i) for i in config["bitsets"]])
        levels = 2**(bitsets.float()-1)  # .float() must be used in case of inf for bit=0
        grad_scale = 1 / levels**0.5 / self.weight.numel()**0.5
        gamma = torch.zeros((self.group_num, 1, len(bitsets)))
        alpha = torch.ones((len(bitsets))) * 2*self.weight.abs().mean() / levels**0.5
        beta = torch.zeros((self.emb_dim))
        
        self.tau = config["tau"]
        self.gamma = nn.Parameter(gamma)
        self.alpha = nn.Parameter(alpha)
        self.beta = nn.Parameter(beta)
        self.bitsets = nn.Parameter(bitsets, requires_grad=False)
        self.grad_scale = nn.Parameter(grad_scale, requires_grad=False)
        self.retraining = False

    def forward(self, x):
        weight = F.embedding(x, self.weight)
        group = self.group_index[x]
        if self.retraining:
            gamma = self.gamma
        else:
            gamma = F.softmax(self.gamma / self.tau, dim=-1)
        gamma = gamma[group]
        alpha = self.alpha.abs() + torch.tensor(1e-10)
        output = torch.zeros_like(weight)
        for i, bit in enumerate(self.bitsets):
            if bit == 0: continue
            quantized_weight = lsq_plus.apply(weight, alpha[i], self.beta, bit, self.grad_scale[i])
            output += quantized_weight * gamma[:,:,:,i]
        return output

    def regularization(self):
        gamma = F.softmax(self.gamma / self.tau, dim=-1)
        reg_loss = (gamma * self.bitsets / self.group_cnt).sum()
        return reg_loss

    def get_bit(self):
        threshold = 1/2/len(self.bitsets)
        gamma = F.softmax(self.gamma / self.tau, dim=-1)
        larger_gamma = (gamma >= threshold).float()
        possible_bits = larger_gamma * self.bitsets
        bit = possible_bits.max(dim=-1)[0].view(-1)
        index = possible_bits.max(dim=-1)[1].view(-1)
        logit = torch.tensor([gamma[g,:,i] for g, i in enumerate(index)])
        return bit, logit

    def discrete_gamma(self):
        bit, _ = self.get_bit()
        self.gamma.data = (bit.reshape(-1,1,1) == self.bitsets).float()
        self.gamma = nn.Parameter(self.gamma.data, requires_grad=False)
        self.retraining = True


def replace_embedding(model, model_config, data_config):
    emb_dict = {
        "qr": QR_Embedding,
        "lsq": Lsq_Embedding,
        "lsq_plus": Lsq_Plus_Embedding,
        "pep": PEP_Embedding,
        "alpt": ALPT_Embedding,
        "optfs": OptFS_Embedding,
    }
    emb_data_dict = {
        "optfp": OptFP_Embedding,
    }
    if model_config["emb_type"] in emb_dict.keys():
        model.embedding = emb_dict[model_config["emb_type"]](model.feature_num, model.emb_dim, model_config['emb_init'], model_config)
    elif model_config["emb_type"] in emb_data_dict.keys():
        model.embedding = emb_data_dict[model_config["emb_type"]](model.feature_num, model.emb_dim, model_config['emb_init'], data_config['dataset'], model_config)

if __name__ == "__main__":
    emb = Embedding(4428293, 16, "normal_0.003", None)
    print(emb)