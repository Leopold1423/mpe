import os
import tqdm
import argparse
import tensorflow as tf
from pathlib import Path
from datatransform import DataTransform, feature_example

parser = argparse.ArgumentParser(description='Transfrom original data to TFRecord')
parser.add_argument('--label', default="Label", type=str)
parser.add_argument("--store_stat", action="store_true")
parser.add_argument("--threshold", type=int, default=2)
parser.add_argument("--dataset", type=Path, default="./kdd12_new/training.txt")
parser.add_argument("--record", type=Path, default="./kdd12_new/threshold_")
parser.add_argument("--ratio", nargs='+', type=float, default=[0.8, 0.1, 0.1])
args = parser.parse_args()
if args.record == Path("./kdd12_new/threshold_"):
    args.record = Path("./kdd12_new/threshold_"+str(args.threshold)+"/")
os.makedirs(args.record, exist_ok=True)


class KDDTransform(DataTransform):
    def __init__(self, dataset_path, path, min_threshold, label_index, ratio, store_stat=False, seed=2022):
        super(KDDTransform, self).__init__(dataset_path, path, store_stat=store_stat, seed=seed)
        self.threshold = min_threshold
        self.label = label_index
        self.split = ratio
        self.path = path
        self.name ="Label,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11".split(",")
    
    def process(self):
        self._read(name = self.name, header = None, label_index = self.label,sep="\t")
        if self.store_stat:
            self.generate_and_filter(threshold=self.threshold, label_index=self.label)
        tr, te, val = self.random_split(ratio=self.split)
        self.transform_tfrecord(tr, self.path, "train", label_index=self.label)
        self.transform_tfrecord(te, self.path, "test", label_index=self.label)
        self.transform_tfrecord(val, self.path, "validation", label_index=self.label)

    def _process_x(self):
        print(self.data[self.data["Label"] == 1].shape)
    
    def _process_y(self):
        self.data["Label"] = self.data["Label"].apply(lambda x: 0 if x == 0 else 1)

    def transform_tfrecord(self, data, record_path, flag, records = 5e6, label_index=""):
        part = 0
        instance_num = 0
        while records * part <= data.shape[0]:
            tf_writer = tf.io.TFRecordWriter(os.path.join(record_path, "{}_{:04d}.tfrecord".format(flag, part)))
            print("===write part {:04d}===".format(part))
            #pbar = tqdm.tqdm(total = int(records))
            tmp_data = data[int(part * records): int((part + 1) * records)]
            pbar = tqdm.tqdm(total = tmp_data.shape[0])
            for index,row in tmp_data.iterrows():
                label = None
                feature = []
                #oov = True
                for i in self.field_name:
                    if i == label_index:
                        label = float(row[i])
                        continue
                    #print(i+"_"+str(int(row[i])))
                    feat_id = self.feat_map.setdefault(i+"_"+str(int(row[i])), self.field_offset[i] - 1)  # only differe with DataTransform: int() 
                    #oov  = oov and (feat_id == self.field_offset[i])
                    feature.append(feat_id)
                #if oov:
                    #continue
                tf_writer.write(feature_example(label, feature))
                pbar.update(1)
                instance_num += 1
            tf_writer.close()
            pbar.close()
            part += 1
        print("real instance number:", instance_num)


if __name__ == "__main__":
    tranformer = KDDTransform(args.dataset, args.record, args.threshold, args.label, args.ratio, args.store_stat, 2022)
    tranformer.process()
