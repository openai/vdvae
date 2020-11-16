import torch
import numpy as np
from mpi4py import MPI
import socket
import argparse
import os
import json
import subprocess
from hps import Hyperparams, parse_args_and_update_hparams, add_vae_arguments
from utils import (logger,
                   local_mpi_rank,
                   mpi_size,
                   maybe_download,
                   mpi_rank)
from data import mkdir_p
from contextlib import contextmanager
import torch.distributed as dist
from apex.optimizers import FusedAdam as AdamW
from vae import VAE
from torch.nn.parallel.distributed import DistributedDataParallel


def update_ema(vae, ema_vae, ema_rate):
    for p1, p2 in zip(vae.parameters(), ema_vae.parameters()):
        p2.data.mul_(ema_rate)
        p2.data.add_(p1.data * (1 - ema_rate))


def save_model(path, vae, ema_vae, optimizer, H):
    torch.save(vae.state_dict(), f'{path}-model.th')
    torch.save(ema_vae.state_dict(), f'{path}-model-ema.th')
    torch.save(optimizer.state_dict(), f'{path}-opt.th')
    from_log = os.path.join(H.save_dir, 'log.jsonl')
    to_log = f'{os.path.dirname(path)}/{os.path.basename(path)}-log.jsonl'
    subprocess.check_output(['cp', from_log, to_log])


def accumulate_stats(stats, frequency):
    z = {}
    for k in stats[-1]:
        if k in ['distortion_nans', 'rate_nans', 'skipped_updates', 'gcskip']:
            z[k] = np.sum([a[k] for a in stats[-frequency:]])
        elif k == 'grad_norm':
            vals = [a[k] for a in stats[-frequency:]]
            finites = np.array(vals)[np.isfinite(vals)]
            if len(finites) == 0:
                z[k] = 0.0
            else:
                z[k] = np.max(finites)
        elif k == 'elbo':
            vals = [a[k] for a in stats[-frequency:]]
            finites = np.array(vals)[np.isfinite(vals)]
            z['elbo'] = np.mean(vals)
            z['elbo_filtered'] = np.mean(finites)
        elif k == 'iter_time':
            z[k] = stats[-1][k] if len(stats) < frequency else np.mean([a[k] for a in stats[-frequency:]])
        else:
            z[k] = np.mean([a[k] for a in stats[-frequency:]])
    return z


def linear_warmup(warmup_iters):
    def f(iteration):
        return 1.0 if iteration > warmup_iters else iteration / warmup_iters
    return f


def setup_mpi(H):
    H.mpi_size = mpi_size()
    H.local_rank = local_mpi_rank()
    H.rank = mpi_rank()
    os.environ["RANK"] = str(H.rank)
    os.environ["WORLD_SIZE"] = str(H.mpi_size)
    os.environ["MASTER_PORT"] = str(H.port)
    # os.environ["NCCL_LL_THRESHOLD"] = "0"
    os.environ["MASTER_ADDR"] = MPI.COMM_WORLD.bcast(socket.gethostname(), root=0)
    torch.cuda.set_device(H.local_rank)
    dist.init_process_group(backend='nccl', init_method=f"env://")


def distributed_maybe_download(path, local_rank, mpi_size):
    if not path.startswith('gs://'):
        return path
    filename = path[5:].replace('/', '-')
    with first_rank_first(local_rank, mpi_size):
        fp = maybe_download(path, filename)
    return fp


@contextmanager
def first_rank_first(local_rank, mpi_size):
    if mpi_size > 1 and local_rank > 0:
        dist.barrier()

    try:
        yield
    finally:
        if mpi_size > 1 and local_rank == 0:
            dist.barrier()


def setup_save_dirs(H):
    H.save_dir = os.path.join(H.save_dir, H.desc)
    mkdir_p(H.save_dir)
    H.logdir = os.path.join(H.save_dir, 'log')


def set_up_hyperparams(s=None):
    H = Hyperparams()
    parser = argparse.ArgumentParser()
    parser = add_vae_arguments(parser)
    parse_args_and_update_hparams(H, parser, s=s)
    setup_mpi(H)
    setup_save_dirs(H)
    logprint = logger(H.logdir)
    for i, k in enumerate(sorted(H)):
        logprint(type='hparam', key=k, value=H[k])
    np.random.seed(H.seed)
    torch.manual_seed(H.seed)
    torch.cuda.manual_seed(H.seed)
    logprint('training model', H.desc, 'on', H.dataset)
    return H, logprint


def restore_params(model, path, local_rank, mpi_size, map_ddp=True, map_cpu=False):
    state_dict = torch.load(distributed_maybe_download(path, local_rank, mpi_size), map_location='cpu' if map_cpu else None)
    if map_ddp:
        new_state_dict = {}
        l = len('module.')
        for k in state_dict:
            if k.startswith('module.'):
                new_state_dict[k[l:]] = state_dict[k]
            else:
                new_state_dict[k] = state_dict[k]
        state_dict = new_state_dict
    model.load_state_dict(state_dict)


def restore_log(path, local_rank, mpi_size):
    loaded = [json.loads(l) for l in open(distributed_maybe_download(path, local_rank, mpi_size))]
    try:
        cur_eval_loss = min([z['elbo'] for z in loaded if 'type' in z and z['type'] == 'eval_loss'])
    except ValueError:
        cur_eval_loss = float('inf')
    starting_epoch = max([z['epoch'] for z in loaded if 'type' in z and z['type'] == 'train_loss'])
    iterate = max([z['step'] for z in loaded if 'type' in z and z['type'] == 'train_loss'])
    return cur_eval_loss, iterate, starting_epoch


def load_vaes(H, logprint):
    vae = VAE(H)
    if H.restore_path:
        logprint(f'Restoring vae from {H.restore_path}')
        restore_params(vae, H.restore_path, map_cpu=True, local_rank=H.local_rank, mpi_size=H.mpi_size)

    ema_vae = VAE(H)
    if H.restore_ema_path:
        logprint(f'Restoring ema vae from {H.restore_ema_path}')
        restore_params(ema_vae, H.restore_ema_path, map_cpu=True, local_rank=H.local_rank, mpi_size=H.mpi_size)
    else:
        ema_vae.load_state_dict(vae.state_dict())
    ema_vae.requires_grad_(False)

    vae = vae.cuda(H.local_rank)
    ema_vae = ema_vae.cuda(H.local_rank)

    vae = DistributedDataParallel(vae, device_ids=[H.local_rank], output_device=H.local_rank)

    if len(list(vae.named_parameters())) != len(list(vae.parameters())):
        raise ValueError('Some params are not named. Please name all params.')
    total_params = 0
    for name, p in vae.named_parameters():
        total_params += np.prod(p.shape)
    logprint(total_params=total_params, readable=f'{total_params:,}')
    return vae, ema_vae


def load_opt(H, vae, logprint):
    optimizer = AdamW(vae.parameters(), weight_decay=H.wd, lr=H.lr, betas=(H.adam_beta1, H.adam_beta2))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_warmup(H.warmup_iters))

    if H.restore_optimizer_path:
        optimizer.load_state_dict(
            torch.load(distributed_maybe_download(H.restore_optimizer_path, H.local_rank, H.mpi_size), map_location='cpu'))
    if H.restore_log_path:
        cur_eval_loss, iterate, starting_epoch = restore_log(H.restore_log_path, H.local_rank, H.mpi_size)
    else:
        cur_eval_loss, iterate, starting_epoch = float('inf'), 0, 0
    logprint('starting at epoch', starting_epoch, 'iterate', iterate, 'eval loss', cur_eval_loss)
    return optimizer, scheduler, cur_eval_loss, iterate, starting_epoch
