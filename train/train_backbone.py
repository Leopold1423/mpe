import os
import torch
import numpy as np
from tqdm import tqdm
from util.tool import *
from util.logger import get_log
from model.models import get_model
from sklearn.metrics import roc_auc_score, log_loss

class Trainer_BackBone(object):
    def __init__(self, data_loader, data_config, model_config, train_config):
        self.data_loader, self.data_config = data_loader, data_config
        self.model_config, self.train_config = model_config, train_config
        # model
        self.model = get_model(model_config, data_config)
        self.device = get_device(train_config['gpu'])
        self.model.to(self.device)
        # loss
        self.criterion = torch.nn.functional.binary_cross_entropy_with_logits
        self.optimizer = get_optimizer(self.model, train_config)
        # parameter
        self.sample_nums, self.batch_size = data_config["sample_nums"], train_config["batch_size"]
        self.val_per_epoch, self.early_stop = self.train_config['val_per_epoch'], self.train_config['early_stop']
        # log
        self.log_path = self.train_config["log_path"]
        self.log_name = self.train_config["log_name"]
        self.save_path = os.path.join(self.log_path, self.log_name)
        os.makedirs(self.save_path, exist_ok=True)
        self.logger = get_log(self.log_path, self.log_name)
        self.logger.info("data_config: {}".format(self.data_config))
        self.logger.info("model_config: {}".format(self.model_config))
        self.logger.info("train_config: {}".format(self.train_config))
        # log points
        self.train_steps = self.sample_nums["train"] // self.batch_size + 1
        self.print_points = torch.linspace(1, self.train_steps, 11).int()[1:]
        self.eval_points = torch.linspace(1, self.train_steps, self.val_per_epoch+1).int()[1:]
        self.best_auc, self.best_loss, self.best_epoch = 0.0, 0.0, 0    # record the best validation auc
        self.record = {}     # final json record including validation and test

    def evaluate(self, eval_type="valid", eval_steps=None):
        self.model.eval()
        if eval_steps is None:
            eval_steps = int(self.sample_nums[eval_type] // self.batch_size) + 1
        data_loader = self.data_loader.get_data(eval_type, batch_size=self.batch_size)
        val_prob, val_true, steps = [], [], 0
        for feature, label in tqdm(data_loader, total=eval_steps, ncols=80):
            feature = feature.to(self.device)
            prob = self.model.forward(feature)
            prob = torch.sigmoid(prob).detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            val_prob.append(prob)
            val_true.append(label)
            if steps < eval_steps:
                steps = steps + 1
            else:
                break
        y_prob = np.concatenate(val_prob).astype("float64")
        y_true = np.concatenate(val_true).astype("float64")
        val_auc = round(roc_auc_score(y_true, y_prob), 6)
        val_loss = round(log_loss(y_true, y_prob), 6)
        return val_auc, val_loss

    def update(self, feature, label):
        self.model.train()
        self.optimizer.zero_grad()
        feature, label = feature.to(self.device), label.to(self.device)
        prob = self.model.forward(feature)
        loss = self.criterion(prob, label.squeeze())
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test_best(self):
        self.logger.info("* best epoch: {}; loss: {:.4f}; auc: {:.4f}.".format(
            self.best_epoch, self.best_loss, self.best_auc))
        best_dict = torch.load(os.path.join(self.save_path, "best.pt"))
        self.model.load_state_dict(best_dict)
        test_auc, test_loss = self.evaluate(eval_type="test")
        self.logger.info("* test loss: {:.4f}; test auc: {:.4f}.".format(test_loss, test_auc))
        record = {
            "epoch": self.best_epoch, 
            "valid_loss": self.best_loss, 
            "valid_auc": self.best_auc, 
            "test_loss": test_loss, 
            "test_auc": test_auc,
            }
        return record

    def train_epoch(self):
        self.logger.info('begin training ...')
        for epoch in range(1, self.train_config["epoch"]+1, 1):
            step = 0
            train_data_loader = self.data_loader.get_data("train", batch_size=self.batch_size)
            for feature, label in tqdm(train_data_loader, total=self.train_steps, ncols=80):
                step += 1
                loss = self.update(feature, label)
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
