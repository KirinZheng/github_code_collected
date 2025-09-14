import copy
import os.path
import random

import numpy as np
from mmengine.dataset import force_full_init
from spikingjelly.datasets.n_caltech101 import NCaltech101 as SJNCaltech101

from .builder import DATASETS
# from amzcls.datasets import BaseDataset
from .mmpretrain_datasets import BaseDataset

__all__ = ['NCaltech101']
GLOBAL_DVS_DATASET = None


@DATASETS.register_module()
class NCaltech101(BaseDataset):
    data_infos = None
    pipeline = None

    def __init__(self, data_prefix, test_mode, time_step, data_type='frame',
                 split_by='number', use_ckpt=False, *args, **kwargs):
        super(NCaltech101, self).__init__(
            ann_file='', data_prefix=data_prefix, test_mode=test_mode, lazy_init=True, *args, **kwargs
        )
        self.time_step = time_step
        self.data_type = data_type
        self.split_by = split_by
        dvs_dataset = load_ncaltech101(data_prefix, test_mode, time_step, data_type, split_by)
        self.train_data, self.test_data = load_data_infos(dvs_dataset)
        self.data_infos = self.test_data if test_mode else self.train_data

    def prepare_data(self, idx):
        data_info = copy.deepcopy(self.data_infos[idx])
        data_info['img'] = load_npz(data_info['img'])
        return self.pipeline(data_info)

    def _join_prefix(self):
        pass

    def full_init(self):
        pass

    @force_full_init
    def __len__(self) -> int:
        return len(self.data_infos)


def load_ncaltech101(data_prefix, test_mode, time_step, data_type='frame', split_by='number'):
    global GLOBAL_DVS_DATASET
    if GLOBAL_DVS_DATASET is None:
        GLOBAL_DVS_DATASET = SJNCaltech101(
            root=data_prefix,
            data_type=data_type,
            frames_number=time_step,
            split_by=split_by)
        print(f'[INFO] [AMZCLS] Processing {"testing" if test_mode else "training"} dataset...')
    return GLOBAL_DVS_DATASET


def load_npz(file_name, data_type='frames'):
    return np.load(file_name, allow_pickle=True)[data_type].astype(np.float32)


# implement without checkpoint
def load_data_infos(dvs_dataset, split_rate=0.9, shuffle=True):
    filepath = dvs_dataset.root
    class_list = os.listdir(filepath)
    class_list.sort()
    train_data_infos = []
    test_data_infos = []
    for class_index, class_name in enumerate(class_list):
        file_list = os.listdir(os.path.join(filepath, class_name))
        if shuffle:
            np.random.seed(class_index)
            np.random.shuffle(file_list)
            # random.shuffle(file_list)
        split = int(len(file_list) * split_rate)
        for file in file_list[:split]:
            train_data_infos.append({
                'img': os.path.join(filepath, class_name, file),
                'gt_label': np.array(class_index, dtype=np.int64)
            })
        for file in file_list[split:]:
            test_data_infos.append({
                'img': os.path.join(filepath, class_name, file),
                'gt_label': np.array(class_index, dtype=np.int64)
            })
    return train_data_infos, test_data_infos
