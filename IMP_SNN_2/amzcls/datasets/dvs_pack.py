import copy
from typing import Sequence

import numpy as np
from mmengine.dataset import force_full_init

from .builder import DATASETS, build_dataset
from .mmpretrain_datasets import BaseDataset


@DATASETS.register_module()
class DVSPack(BaseDataset):
    data_infos = None
    pipeline = None

    def __init__(self, dataset_cfgs: Sequence, pipeline: Sequence = None, **kwargs):
        super(DVSPack, self).__init__(
            ann_file='', data_prefix='', pipeline=pipeline, lazy_init=True, **kwargs
        )
        self.datasets = [build_dataset(dataset_cfg) for dataset_cfg in dataset_cfgs]
        self.data_infos = []
        # todo: optimize here
        self.dataset_order = [0]
        for dataset in self.datasets:
            self.data_infos += dataset.data_infos
            self.dataset_order.append(len(dataset.data_infos) + self.dataset_order[-1])
        self.dataset_order = self.dataset_order[1:]

    def prepare_data(self, idx):
        data_info = copy.deepcopy(self.data_infos[idx])
        data_info['img'] = load_npz(data_info['img'])
        # pipeline in origin dataset
        data_info = self.datasets[self.get_dataset_index(idx)].pipeline(data_info)
        # pipeline in this dataset
        data_info = self.pipeline(data_info)
        return data_info

    def _join_prefix(self):
        pass

    def full_init(self):
        pass

    @force_full_init
    def __len__(self) -> int:
        return len(self.data_infos)

    def get_dataset_index(self, sample_idx):
        for dataset_idx, order in enumerate(self.dataset_order):
            if sample_idx < order:
                return dataset_idx


def load_npz(file_name, data_type='frames'):
    return np.load(file_name, allow_pickle=True)[data_type].astype(np.float32)
