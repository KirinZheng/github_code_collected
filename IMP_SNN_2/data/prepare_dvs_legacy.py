import os


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


if __name__ == '__main__':
    split_cifar10dvs('/home/stephen/Desktop/workspace/2023/amz-cls-0.0.2/data/dvs-cifar10', 4, 'number')
