import os
import copy
import torch
from tqdm import tqdm
from util.tool import *
from train.train_backbone import Trainer_BackBone

class Trainer_PEP(Trainer_BackBone):
    def __init__(self, data_loader, data_config, model_config, train_config):
        super().__init__(data_loader, data_config, model_config, train_config)

    def searching(self):
        self.logger.info('begin searching ...')
        torch.save(self.model.state_dict(), os.path.join(self.save_path, "search-init.pt"))

        for epoch in range(1, self.train_config["epoch"]+1, 1):
            step = 0
            train_data_loader = self.data_loader.get_data("train", batch_size=self.batch_size)
            for feature, label in tqdm(train_data_loader, total=self.train_steps, ncols=80):
                step += 1
                loss = self.update(feature, label)
                if step in self.print_points:
                    self.logger.info("epoch: {}; step: {}; train loss: {:.4f}.".format(epoch, step, loss))
                if step in self.eval_points:
                    val_auc, val_loss = self.evaluate(eval_type="valid")
                    sparsity = self.model.embedding.get_sparsity()
                    self.logger.info("* valid epoch: {}; step: {}; auc: {:.4f}; loss: {:.4f}; sparsity: {:.4f}.".format(epoch, step, val_auc, val_loss, sparsity))
                
            if val_auc > self.best_auc:
                self.best_auc, self.best_loss, self.best_epoch = val_auc, val_loss, epoch
                torch.save(self.model.state_dict(), os.path.join(self.save_path, "search-best.pt"))
            if epoch - self.best_epoch >= self.early_stop:
                break
        
        sparsity = self.model.embedding.get_sparsity()
        search_record = {
            "search_epoch": epoch,
            "search_auc": val_auc,
            "search_loss": val_loss,
            "search_sparsity": sparsity,
            }
        self.record.update(search_record)

    def prepare_retrain(self):
        # load model
        init_dict = torch.load(os.path.join(self.save_path, "search-init.pt"), map_location=self.device)
        init_dict['embedding.mask'] = copy.deepcopy(self.model.embedding.get_mask())
        self.model.load_state_dict(init_dict)
        self.model.embedding.retraining = True
        # reinit optimizer
        self.best_auc, self.best_loss, self.best_epoch = 0.0, 0.0, 0
        self.optimizer = get_optimizer(self.model, self.train_config)

    def train_epoch_new(self):
        self.searching()
        self.prepare_retrain()
        self.train_epoch()
