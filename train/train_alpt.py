import os
import torch
from tqdm import tqdm
from util.tool import *
from train.train_backbone import Trainer_BackBone

def get_optimizer_alpt(network, params):
    backbone_params, alpha_params = [], []
    for name, param in network.named_parameters():
        if "embedding" in name and "alpha" in name:
            alpha_params.append(param)
        else:
            backbone_params.append(param)

    backbone_group = {'params': backbone_params, 'lr': params['lr'], 'weight_decay': params["l2"]}
    alpha_group = {'params': alpha_params, 'lr': params['lr_alpha']}

    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD([backbone_group])
        optimizer_alpha = torch.optim.SGD([alpha_group])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam([backbone_group])
        optimizer_alpha = torch.optim.Adam([alpha_group])
    else:
        raise ValueError("not support optimizer")
    return optimizer, optimizer_alpha

class Trainer_ALPT(Trainer_BackBone):
    def __init__(self, data_loader, data_config, model_config, train_config):
        super().__init__(data_loader, data_config, model_config, train_config)
        self.optimizer, self.optimizer_alpha = get_optimizer_alpt(self.model, train_config)

    def update_second(self, feature, label):
        self.model.train()
        self.optimizer_alpha.zero_grad()
        feature, label = feature.to(self.device), label.to(self.device)
        prob = self.model.forward(feature)
        loss = self.criterion(prob, label.squeeze())
        loss.backward()        
        self.optimizer_alpha.step()
        return loss.item()
    
    def train_epoch(self):
        self.logger.info('begin searching ...')
        for epoch in range(1, self.train_config["epoch"]+1, 1):
            step = 0
            train_data_loader = self.data_loader.get_data("train", batch_size=self.batch_size)
            for feature, label in tqdm(train_data_loader, total=self.train_steps, ncols=80):
                step += 1
                loss = self.update(feature, label)
                self.model.embedding.order = 2
                loss_alpha = self.update_second(feature, label)
                self.model.embedding.order = 1
                self.model.embedding.post_quantizer()
                if step in self.print_points:
                    self.logger.info("epoch: {}; step: {}; train loss: {:.4f}".format(epoch, step, loss))
                if step in self.eval_points:
                    val_auc, val_loss = self.evaluate(eval_type="valid")
                    self.logger.info("* valid epoch: {}; step: {}; auc: {:.4f}; loss: {:.4f}.".format(epoch, step, val_auc, val_loss))

            if val_auc > self.best_auc:
                self.best_auc, self.best_loss, self.best_epoch = val_auc, val_loss, epoch
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "best.pt"))            
            if epoch - self.best_epoch >= self.early_stop:
                break

        record = self.test_best()
        self.record.update(record)
        save_json_record(self.save_path, self.record, name="record")
        
