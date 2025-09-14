import copy
import multiprocessing
import os
import os.path
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from mmengine.dataset import force_full_init
from spikingjelly.datasets import NeuromorphicDatasetFolder
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS, load_events
from torchvision.datasets.utils import extract_archive
from tqdm import tqdm

from .builder import DATASETS
from .mmpretrain_datasets import BaseDataset


@DATASETS.register_module()
class CIFAR10DVSLegacy(BaseDataset):
    def __init__(self, data_prefix, train: bool, time_step: int, data_type='frame',
                 split_by='number', *args, **kwargs):
        super(CIFAR10DVSLegacy, self).__init__(
            ann_file='', data_prefix=data_prefix, test_mode=~train, lazy_init=True, *args, **kwargs
        )
        split_cifar10dvs(data_prefix, time_step, split_by)
        self._cifar10dvs = _CIFAR10DVSLegacy(data_prefix, train, data_type, time_step, split_by)
        self.data_infos = []
        for sample in tqdm(self._cifar10dvs.samples):
            self.data_infos.append({
                'img': sample[0],
                'gt_label': np.array(sample[1], dtype=np.int64)
            })

    def prepare_data(self, idx):
        data_info = copy.deepcopy(self.data_infos[idx])
        data_info['img'] = np.load(
            data_info['img'], allow_pickle=True
        )['frames'].astype(np.float32)
        return self.pipeline(data_info)

    def _join_prefix(self):
        pass

    def full_init(self):
        pass

    @force_full_init
    def __len__(self) -> int:
        return len(self.data_infos)


def split_cifar10dvs(data_prefix, frames_number, split_by):
    root = os.path.join(data_prefix, f'frames_number_{frames_number}_split_by_{split_by}')
    # root = '/home/stephen/Desktop/workspace/Parallel-Spiking-Neuron-main/' \
    #        'cifar10dvs/datasets/CIFAR10DVS/frames_number_10_split_by_number'
    if os.path.exists(os.path.join(root, 'train')) and os.path.exists(os.path.join(root, 'test')):
        return
    for name in ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'):
        source = os.path.join(root, name)
        target = os.path.join(root, 'test', name)
        if not os.path.exists(target):
            os.makedirs(target)
        for i in range(100):
            os.symlink(os.path.join(source, f'cifar10_{name}_{i}.npz'), os.path.join(target, f'cifar10_{name}_{i}.npz'))
        target = os.path.join(root, 'train', name)
        if not os.path.exists(target):
            os.makedirs(target)
        for i in range(100, 1000):
            os.symlink(os.path.join(source, f'cifar10_{name}_{i}.npz'), os.path.join(target, f'cifar10_{name}_{i}.npz'))


class _CIFAR10DVSLegacy(NeuromorphicDatasetFolder):
    def __init__(
            self,
            root: str,
            train: bool = None,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        '''
        :param root: root path of the dataset
        :type root: str
        :param data_type: `event` or `frame`
        :type data_type: str
        :param frames_number: the integrated frame number
        :type frames_number: int
        :param split_by: `time` or `number`
        :type split_by: str
        :param duration: the time duration of each frame
        :type duration: int
        :param transform: a function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        :type transform: callable
        :param target_transform: a function/transform that takes
            in the target and transforms it.
        :type target_transform: callable

        If ``data_type == 'event'``
            the sample in this dataset is a dict whose keys are ['t', 'x', 'y', 'p'] and values are ``numpy.ndarray``.

        If ``data_type == 'frame'`` and ``frames_number`` is not ``None``
            events will be integrated to frames with fixed frames number. ``split_by`` will define how to split events.
            See :class:`cal_fixed_frames_number_segment_index` for
            more details.

        If ``data_type == 'frame'``, ``frames_number`` is ``None``, and ``duration`` is not ``None``
            events will be integrated to frames with fixed time duration.

        '''
        super().__init__(root, train, data_type, frames_number, split_by, duration, transform,
                         target_transform)

    @staticmethod
    def resource_url_md5() -> list:
        '''
        :return: A list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        '''
        return [
            ('airplane.zip', 'https://ndownloader.figshare.com/files/7712788', '0afd5c4bf9ae06af762a77b180354fdd'),
            ('automobile.zip', 'https://ndownloader.figshare.com/files/7712791', '8438dfeba3bc970c94962d995b1b9bdd'),
            ('bird.zip', 'https://ndownloader.figshare.com/files/7712794', 'a9c207c91c55b9dc2002dc21c684d785'),
            ('cat.zip', 'https://ndownloader.figshare.com/files/7712812', '52c63c677c2b15fa5146a8daf4d56687'),
            ('deer.zip', 'https://ndownloader.figshare.com/files/7712815', 'b6bf21f6c04d21ba4e23fc3e36c8a4a3'),
            ('dog.zip', 'https://ndownloader.figshare.com/files/7712818', 'f379ebdf6703d16e0a690782e62639c3'),
            ('frog.zip', 'https://ndownloader.figshare.com/files/7712842', 'cad6ed91214b1c7388a5f6ee56d08803'),
            ('horse.zip', 'https://ndownloader.figshare.com/files/7712851', 'e7cbbf77bec584ffbf913f00e682782a'),
            ('ship.zip', 'https://ndownloader.figshare.com/files/7712836', '41c7bd7d6b251be82557c6cce9a7d5c9'),
            ('truck.zip', 'https://ndownloader.figshare.com/files/7712839', '89f3922fd147d9aeff89e76a2b0b70a7')
        ]

    @staticmethod
    def downloadable() -> bool:
        '''
        :return: Whether the dataset can be directly downloaded by python codes. If not, the user have to download it manually
        :rtype: bool
        '''
        return True

    @staticmethod
    def extract_downloaded_files(download_root: str, extract_root: str):
        '''
        :param download_root: Root directory path which saves downloaded dataset files
        :type download_root: str
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :return: None

        This function defines how to extract download files.
        '''
        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 10)) as tpe:
            for zip_file in os.listdir(download_root):
                zip_file = os.path.join(download_root, zip_file)
                print(f'Extract [{zip_file}] to [{extract_root}].')
                tpe.submit(extract_archive, zip_file, extract_root)

    @staticmethod
    def load_origin_data(file_name: str) -> Dict:
        '''
        :param file_name: path of the events file
        :type file_name: str
        :return: a dict whose keys are ['t', 'x', 'y', 'p'] and values are ``numpy.ndarray``
        :rtype: Dict

        This function defines how to read the origin binary data.
        '''
        with open(file_name, 'rb') as fp:
            t, x, y, p = load_events(fp,
                                     x_mask=0xfE,
                                     x_shift=1,
                                     y_mask=0x7f00,
                                     y_shift=8,
                                     polarity_mask=1,
                                     polarity_shift=None)
            # return {'t': t, 'x': 127 - x, 'y': y, 'p': 1 - p.astype(int)}  # this will get the same data with http://www2.imse-cnm.csic.es/caviar/MNIST_DVS/dat2mat.m
            # see https://github.com/jackd/events-tfds/pull/1 for more details about this problem
            return {'t': t, 'x': 127 - y, 'y': 127 - x, 'p': 1 - p.astype(int)}

    @staticmethod
    def get_H_W() -> Tuple:
        '''
        :return: A tuple ``(H, W)``, where ``H`` is the height of the data and ``W` is the weight of the data.
            For example, this function returns ``(128, 128)`` for the DVS128 Gesture dataset.
        :rtype: tuple
        '''
        return 128, 128

    @staticmethod
    def read_aedat_save_to_np(bin_file: str, np_file: str):
        events = CIFAR10DVS.load_origin_data(bin_file)
        np.savez(np_file,
                 t=events['t'],
                 x=events['x'],
                 y=events['y'],
                 p=events['p']
                 )
        print(f'Save [{bin_file}] to [{np_file}].')

    @staticmethod
    def create_events_np_files(extract_root: str, events_np_root: str):
        '''
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :param events_np_root: Root directory path which saves events files in the ``npz`` format
        :type events_np_root:
        :return: None

        This function defines how to convert the origin binary data in ``extract_root`` to ``npz`` format and save converted files in ``events_np_root``.
        '''
        t_ckp = time.time()
        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 64)) as tpe:
            for class_name in os.listdir(extract_root):
                aedat_dir = os.path.join(extract_root, class_name)
                np_dir = os.path.join(events_np_root, class_name)
                os.mkdir(np_dir)
                print(f'Mkdir [{np_dir}].')
                for bin_file in os.listdir(aedat_dir):
                    source_file = os.path.join(aedat_dir, bin_file)
                    target_file = os.path.join(np_dir, os.path.splitext(bin_file)[0] + '.npz')
                    print(f'Start to convert [{source_file}] to [{target_file}].')
                    tpe.submit(CIFAR10DVS.read_aedat_save_to_np, source_file,
                               target_file)
        print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
