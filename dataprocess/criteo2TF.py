import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datatransform import DataTransform

parser = argparse.ArgumentParser(description='Transform original data to TFRecord')
parser.add_argument('--label', type=str, default='Label')
parser.add_argument("--store_stat", action="store_true")
parser.add_argument("--threshold", type=int, default=2)
parser.add_argument("--dataset", type=Path, default="./criteo_new/train.txt")
parser.add_argument("--record", type=Path, default="./criteo_new/threshold_")
parser.add_argument("--ratio", nargs='+', type=float, default=[0.8, 0.1, 0.1])
args = parser.parse_args()
if args.record == Path("./criteo_new/threshold_"):
    args.record = Path("./criteo_new/threshold_"+str(args.threshold)+"/")
os.makedirs(args.record, exist_ok=True)


class CriteoTransform(DataTransform):
    def __init__(self, dataset_path, path, min_threshold, label_index, ratio, store_stat=False, seed=2022):
        super(CriteoTransform, self).__init__(dataset_path, path, store_stat=store_stat, seed=seed)
        self.threshold = min_threshold
        self.label = label_index
        self.split = ratio
        self.path = path
        self.name = "Label,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26".split(",")

    def process(self):
        self._read(name=self.name, header=None, sep="\t", label_index=self.label)
        if self.store_stat:
            self.generate_and_filter(threshold=self.threshold, label_index=self.label,
                                     white_list="I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13".split(","))
        tr, val, te = self.random_split(ratio=self.split)
        self.transform_tfrecord(tr, self.path, "train", label_index=self.label)
        self.transform_tfrecord(val, self.path, "test", label_index=self.label)
        self.transform_tfrecord(te, self.path, "validation", label_index=self.label)

    def _process_x(self):
        def bucket(value):
            if not pd.isna(value):
                if value > 2:
                    value = int(np.floor(np.log(value) ** 2))
                else:
                    value = int(value)
            return value

        for i in range(1, 14):
            col_name = "I{}".format(i)
            self.data[col_name] = self.data[col_name].apply(bucket)

    def _process_y(self):
        pass


if __name__ == "__main__":
    tranformer = CriteoTransform(args.dataset, args.record, args.threshold, args.label, args.ratio, args.store_stat, 2022)
    tranformer.process()
