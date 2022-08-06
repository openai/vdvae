import glob
import gzip
import os
import pickle
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torchvision.datasets import ImageFolder


def set_up_data(H):
    shift_loss = -127.5
    scale_loss = 1. / 127.5
    if H.dataset == 'imagenet32':
        trX, vaX, teX = imagenet32(H.data_root)
        H.image_size = 32
        H.image_channels = 3
        shift = -116.2373
        scale = 1. / 69.37404
    elif H.dataset == 'imagenet64':
        trX, vaX, teX = imagenet64(H.data_root)
        H.image_size = 64
        H.image_channels = 3
        shift = -115.92961967
        scale = 1. / 69.37404
    elif H.dataset == 'ffhq_256':
        trX, vaX, teX = ffhq256(H.data_root)
        H.image_size = 256
        H.image_channels = 3
        shift = -112.8666757481
        scale = 1. / 69.84780273
    elif H.dataset == 'ffhq_1024':
        trX, vaX, teX = ffhq1024(H.data_root)
        H.image_size = 1024
        H.image_channels = 3
        shift = -0.4387
        scale = 1.0 / 0.2743
        shift_loss = -0.5
        scale_loss = 2.0
    elif H.dataset == 'cifar10':
        (trX, _), (vaX, _), (teX, _) = cifar10(H.data_root, one_hot=False)
        H.image_size = 32
        H.image_channels = 3
        shift = -120.63838
        scale = 1. / 64.16736
    elif H.dataset == 'bev':
        trX, vaX, teX = bev(H.data_root)
        H.image_size = 64
        H.image_channels = 1
        shift = -127.5  # Range (0, 255) --> (-1, +1)
        scale = 1. / 127.5
    else:
        raise ValueError('unknown dataset: ', H.dataset)

    do_low_bit = H.dataset in ['ffhq_256']

    if H.test_eval:
        print('DOING TEST')
        eval_dataset = teX
    else:
        eval_dataset = vaX

    shift = torch.tensor([shift]).cuda().view(1, 1, 1, 1)
    scale = torch.tensor([scale]).cuda().view(1, 1, 1, 1)
    shift_loss = torch.tensor([shift_loss]).cuda().view(1, 1, 1, 1)
    scale_loss = torch.tensor([scale_loss]).cuda().view(1, 1, 1, 1)

    if H.dataset == 'ffhq_1024':
        train_data = ImageFolder(trX, transforms.ToTensor())
        valid_data = ImageFolder(eval_dataset, transforms.ToTensor())
        untranspose = True
    else:
        train_data = TensorDataset(torch.as_tensor(trX))
        valid_data = TensorDataset(torch.as_tensor(eval_dataset))
        untranspose = False

    def preprocess_func(x):
        nonlocal shift
        nonlocal scale
        nonlocal shift_loss
        nonlocal scale_loss
        nonlocal do_low_bit
        nonlocal untranspose
        'takes in a data example and returns the preprocessed input'
        'as well as the input processed for the loss'
        if untranspose:
            x[0] = x[0].permute(0, 2, 3, 1)
        inp = x[0].cuda(non_blocking=True).float()
        out = inp.clone()
        inp.add_(shift).mul_(scale)
        if do_low_bit:
            # 5 bits of precision
            out.mul_(1. / 8.).floor_().mul_(8.)
        out.add_(shift_loss).mul_(scale_loss)
        return inp, out

    return H, train_data, valid_data, preprocess_func


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)


def flatten(outer):
    return [el for inner in outer for el in inner]


def unpickle_cifar10(file):
    fo = open(file, 'rb')
    data = pickle.load(fo, encoding='bytes')
    fo.close()
    data = dict(zip([k.decode() for k in data.keys()], data.values()))
    return data


def imagenet32(data_root):
    trX = np.load(os.path.join(data_root, 'imagenet32-train.npy'),
                  mmap_mode='r')
    np.random.seed(42)
    tr_va_split_indices = np.random.permutation(trX.shape[0])
    train = trX[tr_va_split_indices[:-5000]]
    valid = trX[tr_va_split_indices[-5000:]]
    test = np.load(os.path.join(data_root, 'imagenet32-valid.npy'),
                   mmap_mode='r')
    return train, valid, test


def imagenet64(data_root):
    trX = np.load(os.path.join(data_root, 'imagenet64-train.npy'),
                  mmap_mode='r')
    np.random.seed(42)
    tr_va_split_indices = np.random.permutation(trX.shape[0])
    train = trX[tr_va_split_indices[:-5000]]
    valid = trX[tr_va_split_indices[-5000:]]
    test = np.load(os.path.join(data_root, 'imagenet64-valid.npy'),
                   mmap_mode='r')  # this is test.
    return train, valid, test


def ffhq1024(data_root):
    # we did not significantly tune hyperparameters on ffhq-1024, and so simply evaluate on the test set
    return os.path.join(data_root, 'ffhq1024/train'), os.path.join(
        data_root, 'ffhq1024/valid'), os.path.join(data_root, 'ffhq1024/valid')


def ffhq256(data_root):
    trX = np.load(os.path.join(data_root, 'ffhq-256.npy'), mmap_mode='r')
    np.random.seed(5)
    tr_va_split_indices = np.random.permutation(trX.shape[0])
    train = trX[tr_va_split_indices[:-7000]]
    valid = trX[tr_va_split_indices[-7000:]]
    # we did not significantly tune hyperparameters on ffhq-256, and so simply evaluate on the test set
    return train, valid, valid


def cifar10(data_root, one_hot=True):
    tr_data = [
        unpickle_cifar10(
            os.path.join(data_root, 'cifar-10-batches-py/',
                         'data_batch_%d' % i)) for i in range(1, 6)
    ]
    trX = np.vstack(data['data'] for data in tr_data)
    trY = np.asarray(flatten([data['labels'] for data in tr_data]))
    te_data = unpickle_cifar10(
        os.path.join(data_root, 'cifar-10-batches-py/', 'test_batch'))
    teX = np.asarray(te_data['data'])
    teY = np.asarray(te_data['labels'])
    trX = trX.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    teX = teX.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    trX, vaX, trY, vaY = train_test_split(trX,
                                          trY,
                                          test_size=5000,
                                          random_state=11172018)
    if one_hot:
        trY = np.eye(10, dtype=np.float32)[trY]
        vaY = np.eye(10, dtype=np.float32)[vaY]
        teY = np.eye(10, dtype=np.float32)[teY]
    else:
        trY = np.reshape(trY, [-1, 1])
        vaY = np.reshape(vaY, [-1, 1])
        teY = np.reshape(teY, [-1, 1])
    return (trX, trY), (vaX, vaY), (teX, teY)  # (N, H, W, C), (N, 1)


def read_compressed_pickle(path):
    try:
        with gzip.open(path, "rb") as f:
            pkl_obj = f.read()
            obj = pickle.loads(pkl_obj)
            return obj
    except IOError as error:
        print(error)


def bev(data_root, test_size=256):
    '''
    NOTE: Input data range is transformed (0, 255) --> (-1., +1.) during
          preprocessing.

    Returns:
        trX: Training data matrix (N, H, W, C) of images (uint8) with value
             range (0, 255).
    '''
    sample_paths = glob.glob(os.path.join(data_root, '*', '*.pkl.gz'))
    random.shuffle(sample_paths)

    samples = []

    for sample_path in sample_paths:
        sample = read_compressed_pickle(sample_path)
        road_present = sample['road_present']  # (H, W)
        # Convert (0., 1.) prob values --> (0, 255) image values
        road_present = np.round(road_present * 255).astype(np.uint8)

        # road_present = np.stack([road_present, road_present, road_present], -1)
        samples.append(road_present)

    samples = np.stack(samples)  # (N, H, W)
    samples = np.expand_dims(samples, -1)  # (N, H, W, 1)

    trX = samples[:-test_size]
    vaX = samples[-test_size:]
    teX = samples[-test_size:]

    # (N, H, W, C)
    return trX, vaX, teX
