import os
import copy
import torch
from tqdm import tqdm
from util.tool import *
import torch.nn.functional as F
from train.train_backbone import Trainer_BackBone


class Trainer_OptFS(Trainer_BackBone):
    def __init__(self, data_loader, data_config, model_config, train_config):
        super().__init__(data_loader, data_config, model_config, train_config)
        self.validation_count = 0
        if data_config['dataset'] == 'avazu_2':
            self.save_point = 2  
        elif data_config['dataset'] == 'kdd12_2':
            self.save_point = 1 
        else:
            self.save_point = 2

    def update(self, feature, label):
        self.model.train()
        self.optimizer.zero_grad()
        feature, label = feature.to(self.device), label.to(self.device)
        prob = self.model.forward(feature)
        loss = self.criterion(prob, label.squeeze())
        # l2 on mask
        loss += self.train_config['optfs_l1'] * self.model.embedding.regularization()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def searching(self):
        self.logger.info('begin searching ...')
        self.model.embedding.total_epoch = self.train_config["epoch"]    # setting for tau
        torch.save(self.model.state_dict(), os.path.join(self.save_path, "search-init.pt"))
        for epoch in range(1, self.train_config["epoch"]+1, 1):
            self.model.embedding.epoch = epoch     # setting for tau
            step = 0
            train_data_loader = self.data_loader.get_data("train", batch_size=self.batch_size)
            for feature, label in tqdm(train_data_loader, total=self.train_steps, ncols=80):
                step += 1
                loss = self.update(feature, label)
                if step in self.print_points:
                    self.logger.info("epoch:{}; step:{}; train loss:{:.4f}".format(epoch, step, loss))
                if step in self.eval_points:
                    val_auc, val_loss = self.evaluate(eval_type="valid")
                    sparsity = self.model.embedding.get_sparsity()
                    self.logger.info("* valid epoch:{}; step:{}; auc:{:.4f}; loss:{:.4f}; sparsity:{:.4f}.".format(epoch, step, val_auc, val_loss, sparsity))
                    self.validation_count += 1
                    if self.validation_count == self.save_point:
                        torch.save(self.model.state_dict(), os.path.join(self.save_path, "search-middle.pt"))

            if val_auc > self.best_auc:
                self.best_auc, self.best_loss, self.best_epoch = val_auc, val_loss, epoch
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "search-best.pt"))
            if epoch - self.best_epoch >= self.early_stop:
                break
        
        self.logger.info("* best search epoch: {}; loss: {:.4f}; auc: {:.4f}; sparsity: {:.4f}".format(
            self.best_epoch, self.best_loss, self.best_auc, sparsity))
        search_record = {
            "search_epoch": self.best_epoch, 
            "search_auc": self.best_auc, 
            "search_loss": self.best_loss, 
            "search_sparsity": sparsity,
            }
        self.record.update(search_record)

    def prepare_retrain(self):
        # load model
        init_dict = torch.load(os.path.join(self.save_path, "search-middle.pt"), map_location=self.device)
        init_dict['embedding.mask'] = copy.deepcopy(self.model.embedding.mask.data)
        self.model.load_state_dict(init_dict)
        self.model.embedding.retraining = True
        self.model.embedding.mask.requires_grad = False
        # reinit optimizer
        self.best_auc, self.best_loss, self.best_epoch = 0.0, 0.0, 0
        self.optimizer = get_optimizer(self.model, self.train_config)
        
    def train_epoch_new(self):
        self.searching()
        self.prepare_retrain()
        self.train_epoch()
        self.logger.info("* search sparsity: {:.4f}".format(self.record["search_sparsity"]))
        