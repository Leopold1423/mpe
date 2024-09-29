import os
import argparse
import tensorflow as tf
from util.tool import get_data_loader
from train.trainers import get_trainer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.set_visible_devices([], 'GPU')

parser = argparse.ArgumentParser()
# args for model.
parser.add_argument('--dataset', type=str, default='avazu_2', help='criteo/avazu/kdd12')
parser.add_argument('--model', type=str, default='dnn', help='dnn/dcn/deepfm/ipnn')
parser.add_argument('--emb_type', type=str, default='fp32', help='embedding type')
parser.add_argument('--emb_dim', type=int, default=16, help='embedding dimension')
parser.add_argument('--mlp_dims', type=int, nargs='+', default=[1024, 512, 256], help='mlp dimensions')
# args for training.
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--l2', type=float, default=0.0, help='weight decay')
parser.add_argument('--optimizer', type=str, default='adam', help='sgd/adam')
parser.add_argument('--batch_size', type=float, default=10000, help='batch size')
parser.add_argument('--epoch', type=int, default=1, help='max epoch for training')
parser.add_argument('--val_per_epoch', type=int, default=2, help='')
parser.add_argument('--early_stop', type=int, default=5, help='epoch to stop')
parser.add_argument('--gpu', type=int, default=0, help='specify gpu')
parser.add_argument('--log_path', type=str, default='./log/', help='log file save path')
parser.add_argument('--log_name', type=str, default='debug', help='log file name')
# args for compression methods.
parser.add_argument('--qr_ratio', type=int, default=3, help='qr ratio')             # args for qr.
parser.add_argument('--bit', type=int, default=8, help='bit width')                 # args for lsq.
parser.add_argument('--lr_alpha', type=float, default=1e-3, help='lr of alpha')     # args for alpt.
parser.add_argument('--pep_init', type=float, default=-10, help='threshold init')   # args for pep.
parser.add_argument('--optfs_l1', type=float, default=1e-9, help='weight decay')    # ars for optfs.
parser.add_argument('--mask_init', type=float, default=0.5, help='mask init')
parser.add_argument('--l2_gamma', type=float, default=1e-6, help='weight decay')    # args for optfp.
parser.add_argument('--tau', type=float, default=3e-3, help='tau of softmax')
parser.add_argument('--group', type=float, default=128, help='number of group')
parser.add_argument('--bitsets', type=str, default="0123456", help='optimization space of bits')
args = parser.parse_args()

data_loader, data_config = get_data_loader(args.dataset)
model_config = {
    "model": args.model,
    "emb_type": args.emb_type,
    "emb_dim": args.emb_dim,
    "mlp_dims": args.mlp_dims,
    "emb_init": 'normal_0.003',
    "mlp_dropout": 0.0,
    "use_bn": True,
    
    "qr_ratio": args.qr_ratio,          # qr
    "bit": args.bit,                    # lsq
    "pep_init": args.pep_init,          # pep
    "mask_init": args.mask_init,        # optfs
    "tau": args.tau,                    # optfp
    "group": args.group,
    "bitsets": args.bitsets,
}
train_config = {
    "optimizer": args.optimizer,
    "batch_size": int(args.batch_size),
    "lr": args.lr,
    "l2": args.l2,
    "epoch": args.epoch,
    "val_per_epoch": args.val_per_epoch,
    "early_stop": args.early_stop,
    "gpu": args.gpu,
    "log_path": args.log_path,
    "log_name": args.log_name,  

    "lr_alpha": args.lr_alpha,          # alpt
    "optfs_l1": args.optfs_l1,          # optfs
    "l2_gamma": args.l2_gamma,          # optfp
}

trainer = get_trainer(data_loader, data_config, model_config, train_config)
if hasattr(trainer, 'train_epoch_new'):
    trainer.train_epoch_new()
else:
    trainer.train_epoch()
