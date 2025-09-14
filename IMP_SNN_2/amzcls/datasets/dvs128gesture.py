import copy

import numpy as np
from mmengine.dataset import force_full_init
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture as SJDVS128Gesture

from .builder import DATASETS
from .mmpretrain_datasets import BaseDataset

__all__ = ['DVS128Gesture']


@DATASETS.register_module()
class DVS128Gesture(BaseDataset):
    data_infos = None
    pipeline = None

    def __init__(self, data_prefix, test_mode, time_step, data_type='frame',
                 split_by='number', *args, **kwargs):
        super(DVS128Gesture, self).__init__(
            ann_file='', data_prefix=data_prefix, test_mode=test_mode, lazy_init=True, *args, **kwargs
        )
        self.time_step = time_step
        self.data_type = data_type
        self.split_by = split_by
        train = False if test_mode else True
        dvs_dataset = load_dvs_gesture(data_prefix, train, time_step, data_type, split_by)
        self.data_infos = load_data_infos(dvs_dataset)

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


def load_dvs_gesture(data_prefix, train, time_step, data_type='frame', split_by='number'):
    dvs_dataset = SJDVS128Gesture(
        root=data_prefix,
        train=train,
        data_type=data_type,
        frames_number=time_step,
        split_by=split_by)
    print(f'[INFO] [AMZCLS] Processing {"train" if train else "test"} set...\n'
          f'[INFO] [AMZCLS] The {"train" if train else "test"} set contains {len(dvs_dataset.samples)} samples')
    return dvs_dataset


def load_npz(file_name, data_type='frames'):
    return np.load(file_name, allow_pickle=True)[data_type].astype(np.float32)


# implement without checkpoint
def load_data_infos(dvs_dataset):
    data_infos = []
    for sample in dvs_dataset.samples:
        data_infos.append({
            'img': sample[0],
            'gt_label': np.array(sample[1], dtype=np.int64)
        })
    return data_infos
