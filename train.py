import os
import time

import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data import set_up_data
from train_helpers import (accumulate_stats, load_opt, load_vaes, save_model,
                           set_up_hyperparams, update_ema)
from utils import get_cpu_stats_over_ranks


def training_step(H, data_input, target, vae, ema_vae, optimizer, iterate):
    t0 = time.time()
    
    vae.zero_grad()

    B = data_input.shape[0]
    device = data_input.get_device()

    x_1_prob = data_input[:, :, :, 0:1]  # Extract 'road_present' tensors
    # x_2_prob = data_input[:, :, :, 1:2]  # Extract 'road_future' tensors

    # Value range (0, 1) --> (-1, +1)
    x_1 = 2 * x_1_prob - 1
    # x_2 = 2 * x_2_prob - 1

    x_1_target = x_1.detach().clone()

    # Observable mask
    mask_1s = torch.logical_or(x_1 < 0., x_1 > 0.).to(device)
    # mask_2s = torch.logical_or(x_2 < 0., x_2 > 0.).to(device)
    mask_1s = mask_1s[:, :, :, 0]  # (B, H, W)
    # mask_2s = mask_2s[:, :, :, 0]

    # Past + Future sample
    # x_2_full = x_1.detach().clone()
    # x_2_full[mask_2s] = 0
    # x_2_full += x_2
    # x_2 = x_2_full
    # mask_2s = torch.logical_or(mask_1s, mask_2s)

    # x_cat = torch.concat((x_1, x_2), dim=0)
    # x_prob_cat = torch.concat((x_1_prob, x_2_prob), dim=0)
    # mask_cat = torch.concat((mask_1s, mask_2s), dim=0)

    if H.rnd_noise_ratio > 0.:
        B, h, w, c = x_1.shape
        mask_prob = H.rnd_noise_ratio * torch.rand(1, device=torch.device(device))
        mask = torch.rand(
            (B, h, w, c), device=torch.device(device)) < mask_prob
        x_1[mask] = 0

    stats = vae.forward(x_1, x_1_target, mask_1s)

    stats['elbo'].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(),
                                               H.grad_clip).item()
    distortion_nans = torch.isnan(stats['distortion']).sum()
    rate_nans = torch.isnan(stats['rate']).sum()
    stats.update(
        dict(rate_nans=0 if rate_nans == 0 else 1,
             distortion_nans=0 if distortion_nans == 0 else 1))
    stats = get_cpu_stats_over_ranks(stats)

    skipped_updates = 1
    # only update if no rank has a nan and if the grad norm is below a specific threshold
    if stats['distortion_nans'] == 0 and stats['rate_nans'] == 0 and (
            H.skip_threshold == -1 or grad_norm < H.skip_threshold):
        optimizer.step()
        skipped_updates = 0
        update_ema(vae, ema_vae, H.ema_rate)

    t1 = time.time()
    stats.update(skipped_updates=skipped_updates,
                 iter_time=t1 - t0,
                 grad_norm=grad_norm)
    return stats


def eval_step(data_input, target, ema_vae):
    with torch.no_grad():
        device = data_input.get_device()
        x_1_prob = data_input[:, :, :, 0:1]  # Extract 'road_present' tensors
        # Value range (0, 1) --> (-1, +1)
        x_1 = 2 * x_1_prob - 1
        # Observable mask
        mask_1s = torch.logical_or(x_1 < 0., x_1 > 0.).to(device)
        mask_1s = mask_1s[:, :, :, 0]  # (B, H, W)
        stats = ema_vae.forward(x_1, x_1, mask_1s)
    stats = get_cpu_stats_over_ranks(stats)
    return stats


def get_sample_for_visualization(data, preprocess_fn, num, dataset):
    '''
    Returns
        orig_image: RGB (np.uint8) image w. dim (B, H, W, C).
        preprocessed: Tensor in range (-1, 1) w. dim (B, H, W, C).
    '''
    for x in DataLoader(data, batch_size=num):
        break
    # Convert to image value range
    orig_image = (x * 255.0).to(torch.uint8)
    preprocessed = preprocess_fn(x)[0]
    # TODO Centralize the (0, 1) --> (-1, 1) transformation in the preprocessor
    preprocessed = 2 * preprocessed -1
    # Remove 'future' sample
    orig_image = orig_image[:,:,:,0:1]
    preprocessed = preprocessed[:,:,:,0:1]
    return orig_image, preprocessed


def train_loop(H, data_train, data_valid, preprocess_fn, vae, ema_vae,
               logprint):
    optimizer, scheduler, cur_eval_loss, iterate, starting_epoch = load_opt(
        H, vae, logprint)
    train_sampler = DistributedSampler(data_train,
                                       num_replicas=H.mpi_size,
                                       rank=H.rank)
    viz_batch_original, viz_batch_processed = get_sample_for_visualization(
        data_valid, preprocess_fn, H.num_images_visualize, H.dataset)
    early_evals = set([1] + [2**exp for exp in range(3, 14)])
    stats = []
    iters_since_starting = 0
    H.ema_rate = torch.as_tensor(H.ema_rate).cuda()
    for epoch in range(starting_epoch, H.num_epochs):
        train_sampler.set_epoch(epoch)
        for x in DataLoader(data_train,
                            batch_size=H.n_batch,
                            drop_last=True,
                            pin_memory=True,
                            sampler=train_sampler):
            data_input, target = preprocess_fn(x)
            training_stats = training_step(H, data_input, target, vae, ema_vae,
                                           optimizer, iterate)
            stats.append(training_stats)
            scheduler.step()
            if iterate % H.iters_per_print == 0 or iters_since_starting in early_evals:
                logprint(model=H.desc,
                         type='train_loss',
                         lr=scheduler.get_last_lr()[0],
                         epoch=epoch,
                         step=iterate,
                         **accumulate_stats(stats, H.iters_per_print))

            if iterate % H.iters_per_images == 0 or (
                    iters_since_starting in early_evals
                    and H.dataset != 'ffhq_1024') and H.rank == 0:
                write_images(H, ema_vae, viz_batch_original,
                             viz_batch_processed,
                             f'{H.save_dir}/samples-{iterate}.png', logprint)

            iterate += 1
            iters_since_starting += 1
            if iterate % H.iters_per_save == 0 and H.rank == 0:
                if np.isfinite(stats[-1]['elbo']):
                    logprint(model=H.desc,
                             type='train_loss',
                             epoch=epoch,
                             step=iterate,
                             **accumulate_stats(stats, H.iters_per_print))
                    fp = os.path.join(H.save_dir, 'latest')
                    logprint(f'Saving model@ {iterate} to {fp}')
                    save_model(fp, vae, ema_vae, optimizer, H)

            if iterate % H.iters_per_ckpt == 0 and H.rank == 0:
                save_model(os.path.join(H.save_dir, f'iter-{iterate}'), vae,
                           ema_vae, optimizer, H)

        if epoch % H.epochs_per_eval == 0:
            valid_stats = evaluate(H, ema_vae, data_valid, preprocess_fn)
            logprint(model=H.desc,
                     type='eval_loss',
                     epoch=epoch,
                     step=iterate,
                     **valid_stats)


def evaluate(H, ema_vae, data_valid, preprocess_fn):
    stats_valid = []
    valid_sampler = DistributedSampler(data_valid,
                                       num_replicas=H.mpi_size,
                                       rank=H.rank)
    for x in DataLoader(data_valid,
                        batch_size=H.n_batch,
                        drop_last=True,
                        pin_memory=True,
                        sampler=valid_sampler):
        data_input, target = preprocess_fn(x)
        stats_valid.append(eval_step(data_input, target, ema_vae))

    if len(stats_valid) < 1:
        raise Exception('Evaluation output list is empty. ' \
                        f'Check sufficient val data size' \
                        f'\n    len(valid_sampler) {len(valid_sampler)}' \
                        f'\n    stats_valid {len(stats_valid)}')

    vals = [a['elbo'] for a in stats_valid]
    finites = np.array(vals)[np.isfinite(vals)]
    stats = dict(
        n_batches=len(vals),
        filtered_elbo=np.mean(finites),
        **{k: np.mean([a[k] for a in stats_valid])
           for k in stats_valid[-1]})
    return stats


def write_images(H, ema_vae, viz_batch_original, viz_batch_processed, fname,
                 logprint):
    '''
    Args:
        viz_batch_original: RGB (np.uint8) tensor (B,H,W,C).
        viz_batch_processed: Float tensor (B,H,W,C) in the (-1, 1) interval.
    '''
    zs = [
        s['z'].cuda() for s in ema_vae.forward_get_latents(viz_batch_processed)
    ]
    batches = [viz_batch_original.numpy()]
    mb = viz_batch_processed.shape[0]
    lv_points = np.floor(
        np.linspace(0, 1, H.num_variables_visualize + 2) *
        len(zs)).astype(int)[1:-1]
    for i in lv_points:
        batches.append(ema_vae.forward_samples_set_latents(mb, zs[:i], t=0.1))
    for t in [1.0, 0.9, 0.8, 0.7][:H.num_temperatures_visualize]:
        batches.append(ema_vae.forward_uncond_samples(mb, t=t))
    n_rows = len(batches)
    ch = batches[0].shape[-1]
    im = np.concatenate(batches, axis=0).reshape(
        (n_rows, mb,
         *viz_batch_processed.shape[1:])).transpose([0, 2, 1, 3, 4]).reshape([
             n_rows * viz_batch_processed.shape[1],
             mb * viz_batch_processed.shape[2], ch
         ])
    logprint(f'printing samples to {fname}')
    imageio.imwrite(fname, im)


def run_test_eval(H, ema_vae, data_test, preprocess_fn, logprint):
    print('evaluating')
    viz_batch_original, viz_batch_processed = get_sample_for_visualization(
        data_test, preprocess_fn, H.num_images_visualize, H.dataset)
    write_images(H, ema_vae, viz_batch_original, viz_batch_processed,
                 f'{H.save_dir}/samples-eval.png', logprint)
    stats = evaluate(H, ema_vae, data_test, preprocess_fn)
    print('test results')
    for k in stats:
        print(k, stats[k])
    logprint(type='test_loss', **stats)


def main():
    H, logprint = set_up_hyperparams()
    H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
    vae, ema_vae = load_vaes(H, logprint)
    if H.test_eval:
        run_test_eval(H, ema_vae, data_valid_or_test, preprocess_fn, logprint)
    else:
        train_loop(H, data_train, data_valid_or_test, preprocess_fn, vae,
                   ema_vae, logprint)


if __name__ == "__main__":
    main()
