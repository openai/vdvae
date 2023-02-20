import copy
import glob
import gzip
import os
import pickle
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from scipy import ndimage
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
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
    elif H.dataset == 'bev64' or H.dataset == 'bev128' or H.dataset == 'bev256':
        train_data = CompletedBEVDataset(H.data_train_root,
                                         do_rand_rot=H.rotate_samples,
                                         do_masking=H.do_masking,
                                         do_intensity_zeroing=True)
        valid_data = CompletedBEVDataset(H.data_val_root,
                                         do_masking=H.do_masking,
                                         do_intensity_zeroing=True)
        # Create a data matrix
        # dataloader = DataLoader(valid_data, len(valid_data))
        # vaX = next(iter(dataloader)).numpy()
        # teX = vaX
        H.image_channels = 5
        H.image_channels_post_match = 6  # Incl. mask
        shift = 0.
        scale = 1.
        if H.dataset == 'bev64':
            H.image_size = 64
        elif H.dataset == 'bev128':
            H.image_size = 128
        elif H.dataset == 'bev256':
            H.image_size = 256
    else:
        raise ValueError('unknown dataset: ', H.dataset)

    do_low_bit = H.dataset in ['ffhq_256']

    if H.test_eval:
        print('DOING TEST')
        eval_dataset = teX
    else:
        eval_dataset = None  # vaX

    shift = torch.tensor([shift]).cuda().view(1, 1, 1, 1)
    scale = torch.tensor([scale]).cuda().view(1, 1, 1, 1)
    shift_loss = torch.tensor([shift_loss]).cuda().view(1, 1, 1, 1)
    scale_loss = torch.tensor([scale_loss]).cuda().view(1, 1, 1, 1)

    d = H.dataset
    if d == 'ffhq_1024':
        train_data = ImageFolder(trX, transforms.ToTensor())
        valid_data = ImageFolder(eval_dataset, transforms.ToTensor())
        untranspose = True
    elif d == 'bev64' or d == 'bev128' or d == 'bev256':
        untranspose = False
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
        inp = x.cuda(non_blocking=True).float()
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


# def bev(data_root, test_size=256):
#     '''
#     NOTE: Input data range is transformed (0, 255) --> (-1., +1.) during
#           preprocessing.
#     Returns:
#         trX: Training data matrix (N, H, W, C) of images (uint8) with value
#              range (0, 255).
#     '''
#     sample_paths = glob.glob(os.path.join(data_root, '*', '*.pkl.gz'))
#     random.shuffle(sample_paths)
#
#     samples = []
#
#     for sample_path in sample_paths:
#         sample = read_compressed_pickle(sample_path)
#         road_present = sample['road_present']  # (H, W)
#         road_future = sample['road_future']
#         # Convert (0., 1.) prob values --> (0, 255) image values
#         # road_present = np.round(road_present * 255).astype(np.uint8)
#         road_present = road_present.astype(np.float32)
#         road_future = road_future.astype(np.float32)
#
#         sample = np.stack([road_present, road_future], -1)
#         samples.append(sample)
#
#     samples = np.stack(samples)  # (N, H, W)
#
#     trX = samples[:-test_size]
#     vaX = samples[-test_size:]
#     teX = samples[-test_size:]
#
#     # (N, H, W, C)
#     return trX, vaX, teX


class BEVDataset(Dataset):

    def __init__(
        self,
        root_dir,
        do_rand_rot=False,
        reduced_subset_size=-1,
        do_extrapolation=False,
        do_masking=False,
        mask_p_min=0.95,
        mask_p_max=0.99,
    ):
        '''
        Args:
            root_dir: 
            do_rand_rot:
            reduced_subset_size: If positive integer ==> Sample size
            do_extrapolation: Extrapolate beyond observed region by assigning
                              unobserved elements to their nearest neighbor.
            mask_p_min: Probability range for extrapolation masking operation.
            mask_p_max:

        '''
        self.sample_paths = glob.glob(os.path.join(root_dir, '*', '*.pkl.gz'))
        if reduced_subset_size > 0:
            # random.shuffle(self.sample_paths)
            self.sample_paths = self.sample_paths[:reduced_subset_size]

        self.do_rand_rot = do_rand_rot
        self.do_extrapolation = do_extrapolation
        self.do_masking = do_masking
        self.mask_p_min = mask_p_min
        self.mask_p_max = mask_p_max

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        '''
        Returns:
            sample: BEV sample (H,W,C) in (0, 1) interval.
        '''
        sample_path = self.sample_paths[idx]

        sample = self.read_compressed_pickle(sample_path)
        road_present = sample['road_present']  # (H, W)
        road_future = sample['road_future']
        road_full = sample['road_full']
        road_present = road_present.astype(np.float32)
        road_future = road_future.astype(np.float32)
        road_full = road_full.astype(np.float32)

        if self.do_extrapolation:
            pseudo_road_full, _, _ = self.gen_pseudo_bev(road_full)

            if self.do_masking:
                p = np.random.random()
                mask_prob = self.mask_p_max - (1. - self.mask_p_min) * p
                h, w = pseudo_road_full.shape
                mask = np.random.rand(h, w) < mask_prob
                pseudo_road_full[mask] = 0.5

            mask = ~(pseudo_road_full == 0.5)
            pseudo_road_full[mask] = pseudo_road_full[mask]

            mask = road_full == 0.5
            road_full[mask] = pseudo_road_full[mask]

        sample = np.stack([road_present, road_future, road_full], -1)

        sample = torch.tensor(sample)

        if self.do_rand_rot:
            k = random.randint(0, 3)
            sample = torch.rot90(sample, k, [0, 1])

        return sample

    @staticmethod
    def read_compressed_pickle(path):
        try:
            with gzip.open(path, "rb") as f:
                pkl_obj = f.read()
                obj = pickle.loads(pkl_obj)
                return obj
        except IOError as error:
            print(error)

    @staticmethod
    def distance_field(mat: np.array,
                       dist_from_pos: bool,
                       metric: str = 'chebyshev'):
        '''
        Assumes mat has a single semantic class with probablistic measurements
        p(x_{i,j}=True).
        '''
        if dist_from_pos:
            input = mat > 0.5
        else:
            input = mat < 0.5
        not_input = ~input

        input_inner = ndimage.binary_erosion(input)
        contour = input - input_inner.astype(int)

        dist_field = contour.astype(int)

        dist_field[not_input] = distance.cdist(
            np.argwhere(contour), np.argwhere(not_input), metric).min(0) + 1

        return dist_field

    def gen_pseudo_bev(self, bev, metric: str = 'sqeuclidean'):

        pos_dist_field = self.distance_field(bev, True, metric)
        neg_dist_field = self.distance_field(bev, False, metric)

        pseudo_bev = copy.deepcopy(bev)

        for i in range(128):
            for j in range(128):

                obs = bev[i, j]
                if obs > 0.5 or obs < 0.5:
                    continue

                pos_dist = pos_dist_field[i, j]
                neg_dist = neg_dist_field[i, j]

                if pos_dist < neg_dist:
                    pseudo_obs = 1.
                else:
                    pseudo_obs = 0.
                pseudo_bev[i, j] = pseudo_obs

        return pseudo_bev, pos_dist_field, neg_dist_field


class CompletedBEVDataset(Dataset):

    def __init__(
        self,
        root_dir,
        do_rand_rot=False,
        reduced_subset_size=-1,
        do_masking=False,
        mask_p_min=0.95,
        mask_p_max=0.99,
        do_intensity_zeroing=False,
    ):
        '''
        Args:
            root_dir:
            do_rand_rot:
            reduced_subset_size: If positive integer ==> Sample size
            mask_p_min: Probability range for extrapolation masking operation.
            mask_p_max:
            do_intensity_zeroing: Make empty intensity elements zero

        '''
        self.sample_paths = glob.glob(os.path.join(root_dir, '*', '*.pkl.gz'))
        if reduced_subset_size > 0:
            # random.shuffle(self.sample_paths)
            self.sample_paths = self.sample_paths[:reduced_subset_size]

        self.do_rand_rot = do_rand_rot
        self.do_masking = do_masking
        self.mask_p_min = mask_p_min
        self.mask_p_max = mask_p_max
        self.do_intensity_zeroing = do_intensity_zeroing

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        '''
        Returns:
            sample: BEV sample (H,W,C) in (-1, 1) interval.
        '''
        sample_path = self.sample_paths[idx]
        sample = self.read_compressed_pickle(sample_path)

        # Remove redundant batch dimensions (added later)
        sample = sample[0]

        # if self.do_intensity_zeroing:
        #     road_idx = 0
        #     mask = sample[road_idx] <= 0
        #     int = sample[1]
        #     int[mask] = 0.
        #     sample[1] = int
        #
        #     oracle_road_idx = 2
        #     mask = sample[oracle_road_idx] <= 0
        #     int = sample[3]
        #     int[mask] = 0.
        #     sample[3] = int

        if self.do_rand_rot:
            k = random.randint(0, 3)
            sample = torch.rot90(sample, k, [-2, -1])

        # Oracle road, int (0,1) --> (-1,1)
        sample[5:7] = 2 * sample[5:7] - 1

        # RGB (0,1) --> (-1,1)
        sample[2:5] = 2 * sample[2:5] - 1
        sample[7:10] = 2 * sample[7:10] - 1

        return sample

    @staticmethod
    def read_compressed_pickle(path):
        try:
            with gzip.open(path, "rb") as f:
                pkl_obj = f.read()
                obj = pickle.loads(pkl_obj)
                return obj
        except IOError as error:
            print(error)


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    do_rand_rot = False
    dataset = CompletedBEVDataset(
        './c_bevs',
        do_rand_rot,
        do_masking=False,
        do_intensity_zeroing=True,
    )
    print(len(dataset))

    batch_size = 3

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for idx, sample in enumerate(dataloader):

        x, x_oracle = sample.chunk(2, dim=1)

        x_road, x_int = x.chunk(2, dim=1)
        x_oracle_road, x_oracle_int = x_oracle.chunk(2, dim=1)

        batch_idx = 0

        print(idx, sample.shape)
        for batch_idx in range(batch_size):
            plt.subplot(4, batch_size, 0 * batch_size + batch_idx + 1)
            plt.imshow(x_road[batch_idx, 0].numpy())
            plt.subplot(4, batch_size, 1 * batch_size + batch_idx + 1)
            plt.imshow(x_int[batch_idx, 0].numpy())
            plt.subplot(4, batch_size, 2 * batch_size + batch_idx + 1)
            plt.imshow(x_oracle_road[batch_idx, 0].numpy())
            plt.subplot(4, batch_size, 3 * batch_size + batch_idx + 1)
            plt.imshow(x_oracle_int[batch_idx, 0].numpy())

        plt.show()
        # plt.savefig(f'dataset_viz_{idx}.png')

        break
