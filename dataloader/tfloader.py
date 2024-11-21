import glob
import torch
import numpy as np
import tensorflow as tf


class CriteoLoader(object):
    def __init__(self, tfrecord_path):
        self.SAMPLES = 1
        self.FIELDS = 39
        self.tfrecord_path = tfrecord_path
        self.description = {
            "label": tf.io.FixedLenFeature([self.SAMPLES], tf.float32),
            "feature": tf.io.FixedLenFeature([self.FIELDS], tf.int64),
        }

    def get_data(self, data_type, batch_size=1):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return example['feature'], example['label']
        files = glob.glob(self.tfrecord_path + "{}*".format(data_type))
        if not files:
            raise ValueError("no criteo files")
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        for x, y in ds:
            x = torch.from_numpy(x.numpy())
            y = torch.from_numpy(y.numpy())
            yield x, y

class AvazuLoader(object):
    def __init__(self, tfrecord_path):
        self.SAMPLES = 1
        self.FIELDS = 24
        self.tfrecord_path = tfrecord_path
        self.description = {
            "label": tf.io.FixedLenFeature([self.SAMPLES], tf.float32),
            "feature": tf.io.FixedLenFeature([self.FIELDS], tf.int64),
        }

    def get_data(self, data_type, batch_size=1):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return example['feature'], example['label']
        files = glob.glob(self.tfrecord_path + "{}*".format(data_type))
        if not files:
            raise ValueError("no avazu files")
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        for x, y in ds:
            x = torch.from_numpy(x.numpy())
            y = torch.from_numpy(y.numpy())
            yield x, y

class KDD12Loader(object):
    def __init__(self, tfrecord_path):
        self.SAMPLES = 1
        self.FIELDS = 11
        self.tfrecord_path = tfrecord_path
        self.description = {
            "label": tf.io.FixedLenFeature([self.SAMPLES], tf.float32),
            "feature": tf.io.FixedLenFeature([self.FIELDS], tf.int64),
        }

    def get_data(self, data_type, batch_size=1):
        @tf.autograph.experimental.do_not_convert
        def read_data(raw_rec):
            example = tf.io.parse_single_example(raw_rec, self.description)
            return example['feature'], example['label']
        files = glob.glob(self.tfrecord_path + "{}*".format(data_type))
        if not files:
            raise ValueError("no kdd12 files")
        ds = tf.data.TFRecordDataset(files).map(read_data, num_parallel_calls=tf.data.experimental.AUTOTUNE). \
            batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        for x, y in ds:
            x = torch.from_numpy(x.numpy())
            y = torch.from_numpy(y.numpy())
            yield x, y


def count_features_examples(data_loader, batch_size=int(1e5)):
    train_iter = data_loader.get_data("train", batch_size=batch_size)
    step, max_id = 0, 0
    for x, y in train_iter:
        step += 1
        if max_id < np.max(x.cpu().numpy().tolist()):
            max_id = np.max(x.cpu().numpy().tolist())
    train_exapmles = batch_size * (step-1) + len(x)
    print("train examples: {}".format(train_exapmles))
    print("train max_id: {}".format(max_id))

    val_iter = data_loader.get_data("valid", batch_size=batch_size)
    step, max_id = 0, 0
    for x, y in val_iter:
        step += 1
        if max_id < np.max(x.cpu().numpy().tolist()):
            max_id = np.max(x.cpu().numpy().tolist())
    valid_exapmles = batch_size * (step-1) + len(x)
    print("valid examples: {}".format(valid_exapmles))
    print("valid max_id: {}".format(max_id))

    test_iter = data_loader.get_data("test", batch_size=batch_size)
    step, max_id = 0, 0
    for x, y in test_iter:
        step += 1
        if max_id < np.max(x.cpu().numpy().tolist()):
            max_id = np.max(x.cpu().numpy().tolist())
    test_exapmles = batch_size * (step-1) + len(x)
    print("test examples: {}".format(test_exapmles))
    print("test max_id: {}".format(max_id))

def test_criteo_loader(data_path):
    data_loader = CriteoLoader(data_path)
    print("criteo data_path: {} ".format(data_path))
    count_features_examples(data_loader)

def test_avazu_loader(data_path):
    data_loader = AvazuLoader(data_path)
    print("avazu data_path: {} ".format(data_path))
    count_features_examples(data_loader)

def test_kdd12_loader(data_path):
    data_loader = KDD12Loader(data_path)
    print("kdd12 data_path: {} ".format(data_path))
    count_features_examples(data_loader)


# avazu data_path: ./dataprocess/avazu_new/threshold_2/ 
# train examples: 32343174
# train max_id: 4428292
# valid examples: 4042897
# valid max_id: 4428292
# test examples: 4042896
# test max_id: 4428292
# criteo data_path: ./dataprocess/criteo_new/threshold_2/ 
# train examples: 36672494
# train max_id: 6780381
# valid examples: 4584061
# valid max_id: 6780381
# test examples: 4584062
# test max_id: 6780381
# kdd12 data_path: ./dataprocess/kdd12_new/threshold_2/
# train examples: 119711284
# train max_id: 35970484
# valid examples: 14963911
# valid max_id: 35970484
# test examples: 14963910
# test max_id: 35970484


if __name__ == "__main__":
    test_avazu_loader(data_path="./dataprocess/avazu_new/threshold_2/")
    test_criteo_loader(data_path="./dataprocess/criteo_new/threshold_2/")
    test_kdd12_loader(data_path="./dataprocess/kdd12_new/threshold_2/")

