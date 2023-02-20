import json
import os
import subprocess
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist

# from mpi4py import MPI

NUM_GPUS = 6


def allreduce(x, average):
    if mpi_size() > 1:
        dist.all_reduce(x, dist.ReduceOp.SUM)
    return x / mpi_size() if average else x


def get_cpu_stats_over_ranks(stat_dict):
    keys = sorted(stat_dict.keys())
    allreduced = allreduce(torch.stack(
        [torch.as_tensor(stat_dict[k]).detach().cuda().float() for k in keys]),
                           average=True).cpu()
    return {k: allreduced[i].item() for (i, k) in enumerate(keys)}


class Hyperparams(dict):

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value


def logger(log_prefix):
    'Prints the arguments out to stdout, .txt, and .jsonl files'

    jsonl_path = f'{log_prefix}.jsonl'
    txt_path = f'{log_prefix}.txt'

    def log(*args, pprint=False, **kwargs):
        if mpi_rank() != 0:
            return
        t = time.ctime()
        argdict = {'time': t}
        if len(args) > 0:
            argdict['message'] = ' '.join([str(x) for x in args])
        argdict.update(kwargs)

        txt_str = []
        args_iter = sorted(argdict) if pprint else argdict
        for k in args_iter:
            val = argdict[k]
            if isinstance(val, np.ndarray):
                val = val.tolist()
            elif isinstance(val, np.integer):
                val = int(val)
            elif isinstance(val, np.floating):
                val = float(val)
            argdict[k] = val
            if isinstance(val, float):
                val = f'{val:.5f}'
            txt_str.append(f'{k}: {val}')
        txt_str = ', '.join(txt_str)

        if pprint:
            json_str = json.dumps(argdict, sort_keys=True)
            txt_str = json.dumps(argdict, sort_keys=True, indent=4)
        else:
            json_str = json.dumps(argdict)

        print(txt_str, flush=True)

        with open(txt_path, "a+") as f:
            print(txt_str, file=f, flush=True)
        with open(jsonl_path, "a+") as f:
            print(json_str, file=f, flush=True)

    return log


def maybe_download(path, filename=None):
    '''If a path is a gsutil path, download it and return the local link,
    otherwise return link'''
    if not path.startswith('gs://'):
        return path
    if filename:
        local_dest = f'/tmp/'
        out_path = f'/tmp/{filename}'
        if os.path.isfile(out_path):
            return out_path
        subprocess.check_output(['gsutil', '-m', 'cp', '-R', path, out_path])
        return out_path
    else:
        local_dest = tempfile.mkstemp()[1]
        subprocess.check_output(['gsutil', '-m', 'cp', path, local_dest])
    return local_dest


def tile_images(images, d1=4, d2=4, border=1):
    id1, id2, c = images[0].shape
    out = np.ones(
        [d1 * id1 + border * (d1 + 1), d2 * id2 + border * (d2 + 1), c],
        dtype=np.uint8)
    out *= 255
    if len(images) != d1 * d2:
        raise ValueError('Wrong num of images')
    for imgnum, im in enumerate(images):
        num_d1 = imgnum // d2
        num_d2 = imgnum % d2
        start_d1 = num_d1 * id1 + border * (num_d1 + 1)
        start_d2 = num_d2 * id2 + border * (num_d2 + 1)
        out[start_d1:start_d1 + id1, start_d2:start_d2 + id2, :] = im
    return out


def image_grid(img, grid_h, grid_w, color=(255, 255, 255)):

    img_h, img_w, _ = img.shape

    for idx_h in range(1, img_h // grid_h):
        for idx_w in range(1, img_w // grid_w):
            border_h = idx_h * grid_h
            border_w = idx_w * grid_w
            img[border_h, :, :3] = color
            if idx_w % 2 == 0:
                margin = 1
            else:
                margin = 0
            img[:, border_w - margin:border_w + margin + 1, :3] = color

    return img


def arrange_side_by_side(array_a, array_b):
    a = []
    num_arrays = array_a.shape[0]
    for idx in range(num_arrays):
        elem_a = array_a[idx]
        elem_b = array_b[idx]
        a.append(elem_a)
        a.append(elem_b)
    a = np.stack(a)

    return a


def mpi_size():
    return int(os.environ['WORLD_SIZE'])  # MPI.COMM_WORLD.Get_size()


def mpi_rank():
    return int(os.environ['RANK'])  # MPI.COMM_WORLD.Get_rank()


def num_nodes():
    nn = mpi_size()
    if nn % NUM_GPUS == 0:
        return nn // NUM_GPUS
    return nn // NUM_GPUS + 1


def gpus_per_node():
    size = mpi_size()
    if size > 1:
        return max(size // num_nodes(), 1)
    return 1


def local_mpi_rank():
    return mpi_rank() % gpus_per_node()
