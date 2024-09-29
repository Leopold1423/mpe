import os
import copy
import torch
from tqdm import tqdm
from util.tool import *
from train.train_backbone import Trainer_BackBone

class Trainer_OptFP(Trainer_BackBone):
    def __init__(self, data_loader, data_config, model_config, train_config):
        super().__init__(data_loader, data_config, model_config, train_config)
        self.best_bit = 0

    def update_search(self, feature, label):
        self.model.train()
        self.optimizer.zero_grad()
        feature, label = feature.to(self.device), label.to(self.device)
        prob = self.model.forward(feature)
        loss = self.criterion(prob, label.squeeze())
        # l2 on gamma
        loss += self.train_config['l2_gamma'] * self.model.embedding.regularization()
        loss.backward()        
        self.optimizer.step()
        return loss.item()
    
    def searching(self):
        self.logger.info('begin searching ...')
        torch.save(self.model.state_dict(), os.path.join(self.save_path, "search-init.pt"))
        for epoch in range(1, self.train_config["epoch"]+1, 1):
            step = 0
            train_data_loader = self.data_loader.get_data("train", batch_size=self.batch_size)
            for feature, label in tqdm(train_data_loader, total=self.train_steps, ncols=80):
                step += 1
                loss = self.update_search(feature, label)
                if step in self.print_points:
                    self.logger.info("epoch:{}; step:{}; train loss:{:.4f}".format(epoch, step, loss))
                if step in self.eval_points:
                    val_auc, val_loss = self.evaluate(eval_type="valid")
                    val_bit, _ = self.model.embedding.get_bit()
                    val_bit = float(val_bit.mean())
                    self.logger.info("* valid epoch:{}; step:{}; auc:{:.4f}; loss:{:.4f}; bit:{:.4f}.".format(epoch, step, val_auc, val_loss, val_bit))

            if val_auc > self.best_auc:
                self.best_bit, self.best_auc, self.best_loss, self.best_epoch = val_bit, val_auc, val_loss, epoch
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "search-best.pt"))
            if epoch - self.best_epoch >= self.early_stop:
                break
        
        self.logger.info("* best search epoch: {}; loss: {:.4f}; auc: {:.4f}; bit: {:.4f}".format(
            self.best_epoch, self.best_loss, self.best_auc, self.best_bit))
        search_record = {
            "search_epoch": self.best_epoch, 
            "search_auc": self.best_auc, 
            "search_loss": self.best_loss, 
            "search_bit": self.best_bit,
            }
        self.record.update(search_record)
        
    def prepare_retrain(self):
        # load model
        init_dict = torch.load(os.path.join(self.save_path, "search-init.pt"), map_location=self.device)
        best_dict = torch.load(os.path.join(self.save_path, "search-best.pt"), map_location=self.device)
        reuse_keys = ["embedding.gamma"]
        reuse_keys = reuse_keys + [k for k in init_dict.keys() if "embedding" not in k]
        reuse_keys.append("embedding.alpha")
        reuse_keys.append("embedding.beta")
        for k in reuse_keys:
            init_dict[k] = copy.deepcopy(best_dict[k])
        self.model.load_state_dict(init_dict)
        # print bits information
        bit, logit = self.model.embedding.get_bit()
        self.logger.info(bit)
        self.logger.info(logit)
        # convert gamma
        self.model.embedding.discrete_gamma()
        # reinit optimizer
        self.best_auc, self.best_loss, self.best_epoch = 0.0, 0.0, 0
        self.optimizer = get_optimizer(self.model, self.train_config)
        
    def train_epoch_new(self):
        self.searching()
        self.prepare_retrain()
        self.train_epoch()
        self.logger.info("* avg bit: {:.4f}".format(self.best_bit))