import os
import time

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data import set_up_data
# from debug import viz_vae_forward_input
from train_helpers import (accumulate_stats, load_opt, load_vaes, save_model,
                           set_up_hyperparams, update_ema)
from utils import arrange_side_by_side, get_cpu_stats_over_ranks, image_grid


def training_step(H, data_input, target, vae, ema_vae, optimizer, iterate):
    '''

    Both input and targets need to be in the interval (-1, 1).
    Intensity that is blank is 0 for both oracle and posterior matching.

    Args:
        data_input: (B,H,W,C) (-1,1)
    '''
    t0 = time.time()

    vae.zero_grad()

    # x:               Partial [-1,1]
    # x_oracle:        Completed [-1,1]
    # x_oracle_target: Completed [-1,1]
    x = data_input[:, 0:5]
    x_oracle_target = data_input[:, 5:10]
    x_oracle = x_oracle_target.clone()

    # Renaming for paper nomenclature
    x_post_match = x

    # Dummy fully observed completed mask
    m_target = torch.ones_like(x_oracle_target[:, 0:1])

    # Add input observability mask
    m_in = ~(x_post_match[:, 0:1] == 0)
    x_post_match = torch.cat((x_post_match, m_in), dim=1)

    x_oracle = torch.permute(x_oracle, (0, 2, 3, 1))
    x_post_match = torch.permute(x_post_match, (0, 2, 3, 1))
    x_oracle_target = torch.permute(x_oracle_target, (0, 2, 3, 1))
    m_target = torch.permute(m_target, (0, 2, 3, 1))

    # x_oracle:        (2B,H,W,5) (-1,1) <-- Duplicates of B samples
    # x_post_match:    (2B,H,W,6) (-1,1)
    # x_oracle_target: (2B,H,W,5) (-1,1)
    # m_target:        (2B,H,W,1)
    stats = vae.forward(x_oracle, x_post_match, x_oracle_target, m_target)

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
        # x:               Partial [-1,1]
        # x_oracle:        Completed [-1,1]
        # x_oracle_target: Completed [-1,1]
        x = data_input[:, 0:5]  # (-1,1)
        x_oracle_target = data_input[:, 5:10]  # (-1,1)
        x_oracle = x_oracle_target.clone()

        # Renaming for paper nomenclature
        x_post_match = x

        # Dummy fully observed completed mask
        m_target = torch.ones_like(x_oracle_target[:, 0:1])

        # Add input observability mask
        m_in = ~(x_post_match[:, 0:1] == 0)
        x_post_match = torch.cat((x_post_match, m_in), dim=1)

        x_oracle = torch.permute(x_oracle, (0, 2, 3, 1))
        x_post_match = torch.permute(x_post_match, (0, 2, 3, 1))
        x_oracle_target = torch.permute(x_oracle_target, (0, 2, 3, 1))
        m_target = torch.permute(m_target, (0, 2, 3, 1))

        # x_oracle:        (2B,H,W,5) (-1,1) <-- Duplicates of B samples
        # x_post_match:    (2B,H,W,6) (-1,1)
        # x_oracle_target: (2B,H,W,5) (-1,1)
        # m_target:        (2B,H,W,1)
        stats = ema_vae.forward(x_oracle, x_post_match, x_oracle_target,
                                m_target)

    stats = get_cpu_stats_over_ranks(stats)
    return stats


def get_sample_for_visualization(data, preprocess_fn, num, dataset):
    '''
    Returns
        orig_image: RGB (np.uint8) image w. dim (B, H, W, C).
        preprocessed: Tensor in range (-1, 1) w. dim (B, H, W, C).
    '''
    for data_input in DataLoader(data, batch_size=num):
        break
    x = data_input[:, 0:5]  # (B,5,H,W) (-1,1)
    x_oracle_target = data_input[:, 5:10]  # (B,5,H,W) (0,1)

    x_oracle = x_oracle_target.clone()

    x = torch.permute(x, (0, 2, 3, 1))
    x_oracle = torch.permute(x_oracle, (0, 2, 3, 1))

    # Convert to image value range
    orig_image = x.clone()
    orig_image = 0.5 * (orig_image + 1)
    orig_image = (orig_image * 255.0).to(torch.uint8)

    orig_image_oracle = x_oracle.clone()
    orig_image_oracle = 0.5 * (orig_image_oracle + 1)
    orig_image_oracle = (orig_image_oracle * 255.0).to(torch.uint8)
    preprocessed = preprocess_fn(x)[0]
    preprocessed_oracle = preprocess_fn(x_oracle)[0]

    return orig_image, orig_image_oracle, preprocessed, preprocessed_oracle


def train_loop(H, data_train, data_valid, preprocess_fn, vae, ema_vae,
               logprint):
    optimizer, scheduler, cur_eval_loss, iterate, starting_epoch = load_opt(
        H, vae, logprint)
    train_sampler = DistributedSampler(data_train,
                                       num_replicas=H.mpi_size,
                                       rank=H.rank)
    viz_batch_original, viz_batch_original_oracle, viz_batch_processed, viz_batch_processed_oracle = get_sample_for_visualization(
        data_valid, preprocess_fn, H.num_images_visualize, H.dataset)

    # Skip early evaluations to save time
    early_evals = set([9999999])  # set([1] + [2**exp for exp in range(3, 14)])
    stats = []
    iters_since_starting = 0
    H.ema_rate = torch.as_tensor(H.ema_rate).cuda()
    for epoch in range(starting_epoch, H.num_epochs):
        train_sampler.set_epoch(epoch)
        for x in DataLoader(data_train,
                            batch_size=H.n_batch,
                            drop_last=True,
                            pin_memory=True,
                            num_workers=H.dataloader_workers,
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

                m_in = ~(viz_batch_processed[:, :, :, 0:1] == 0)
                viz_batch_processed_w_mask = torch.cat(
                    (viz_batch_processed, m_in), dim=-1)

                for temp in H.viz_temps:
                    write_images(H,
                                 ema_vae,
                                 viz_batch_original,
                                 viz_batch_processed_w_mask,
                                 viz_batch_original_oracle,
                                 viz_batch_processed_oracle,
                                 f'{H.save_dir}/samples-{iterate}_t{temp}.png',
                                 logprint,
                                 temp=temp)

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
        raise Exception('Evaluation output list is empty. '
                        f'Check sufficient val data size'
                        f'\n    len(valid_sampler) {len(valid_sampler)}'
                        f'\n    stats_valid {len(stats_valid)}')

    vals = [a['elbo'] for a in stats_valid]
    finites = np.array(vals)[np.isfinite(vals)]
    stats = dict(
        n_batches=len(vals),
        filtered_elbo=np.mean(finites),
        **{k: np.mean([a[k] for a in stats_valid])
           for k in stats_valid[-1]})
    return stats


def write_images(H,
                 ema_vae,
                 viz_batch_original,
                 viz_batch_processed,
                 viz_batch_original_oracle,
                 viz_batch_processed_oracle,
                 fname,
                 logprint,
                 temp=0.1):
    '''
    Args:
        viz_batch_original: RGB (np.uint8) tensor (B,H,W,C).
        viz_batch_processed: Float tensor (B,H,W,C) in the (-1, 1) interval.
    '''
    ###########
    #  Input
    ###########
    zs = [
        s['z'].cuda() for s in ema_vae.forward_get_latents(viz_batch_processed,
                                                           mode='post_match')
    ]
    zs_oracle = [
        s['z'].cuda()
        for s in ema_vae.forward_get_latents(viz_batch_processed_oracle,
                                             mode='oracle')
    ]
    # Input viz
    obs_viz = viz_batch_original.numpy()
    oracle_viz = viz_batch_original_oracle.numpy()
    input_viz = arrange_side_by_side(obs_viz, oracle_viz)
    batches = [input_viz]

    # Posterior sampling viz
    acts = ema_vae.encoder_post_match.forward(viz_batch_processed)
    px_z, _ = ema_vae.decoder.forward(acts, mode='post_match')
    post_obs = ema_vae.decoder.out_net.sample(px_z)

    acts = ema_vae.encoder.forward(viz_batch_processed_oracle)
    px_z, _ = ema_vae.decoder.forward(acts, mode='oracle')
    post_oracle = ema_vae.decoder.out_net.sample(px_z)

    post_viz = arrange_side_by_side(post_obs, post_oracle)
    batches.append(post_viz)

    mb = input_viz.shape[0]
    lv_points = np.floor(
        np.linspace(0, 1, H.num_variables_visualize + 2) *
        len(zs)).astype(int)[1:-1]
    # Latent sampling viz
    for i in lv_points:
        latent_obs = ema_vae.forward_samples_set_latents(
            mb // 2,
            zs[:i],
            t=temp,
        )
        latent_oracle = ema_vae.forward_samples_set_latents(
            mb // 2,
            zs_oracle[:i],
            t=temp,
        )
        latent_viz = arrange_side_by_side(latent_obs, latent_oracle)
        batches.append(latent_viz)

    # Unconditional viz
    viz_temps_list = H.viz_temps
    for t in viz_temps_list[:H.num_temperatures_visualize]:
        batches.append(ema_vae.forward_uncond_samples(mb, t=t))
    n_rows = len(batches)

    # Add (road, intensity) visualizations (single channel)
    ch = 2  # latent_obs[0].shape[-1]
    batches_tmp = [viz[:, :, :, :2] for viz in batches]
    viz_batch_processed_tmp = viz_batch_processed[:, :, :, 0:2]
    im = np.concatenate(batches_tmp, axis=0).reshape(
        (n_rows, mb, *viz_batch_processed_tmp.shape[1:])).transpose(
            [0, 2, 1, 3, 4]).reshape([
                n_rows * viz_batch_processed_tmp.shape[1],
                mb * viz_batch_processed_tmp.shape[2], ch
            ])
    logprint(f'printing samples to {fname}')

    # Concatenate 'road' and 'intensity' visualizations to side-by-side img
    im = np.concatenate([im[:, :, 0], im[:, :, 1]], axis=1)

    cm = plt.get_cmap('viridis')
    im = cm(im)
    im = im[:, :, :3]  # Remove alpha channel

    im = (255 * im).astype(np.uint8)

    # Add RGB visualizations (three channels)
    ch_rgb = 3
    batches_tmp = [viz[:, :, :, 2:5] for viz in batches]
    viz_batch_processed_tmp = viz_batch_processed[:, :, :, 2:5]
    im_rgb = np.concatenate(batches_tmp, axis=0).reshape(
        (n_rows, mb, *viz_batch_processed_tmp.shape[1:])).transpose(
            [0, 2, 1, 3, 4]).reshape([
                n_rows * viz_batch_processed_tmp.shape[1],
                mb * viz_batch_processed_tmp.shape[2], ch_rgb
            ])

    im = np.concatenate((im, im_rgb), axis=1)

    _, grid_h, grid_w, _ = viz_batch_processed.shape
    im = image_grid(im, grid_h, grid_w)

    imageio.imwrite(fname, im)


def run_test_eval(H, ema_vae, data_test, preprocess_fn, logprint):
    print('evaluating')
    viz_batch_original, viz_batch_original_oracle, viz_batch_processed, viz_batch_processed_oracle = get_sample_for_visualization(
        data_test, preprocess_fn, H.num_images_visualize, H.dataset)

    m_in = ~(viz_batch_processed[:, :, :, 0:1] == 0)
    viz_batch_processed_w_mask = torch.cat((viz_batch_processed, m_in), dim=-1)

    for temp in [0.1, 0.4, 1.0]:
        write_images(H,
                     ema_vae,
                     viz_batch_original,
                     viz_batch_processed_w_mask,
                     viz_batch_original_oracle,
                     viz_batch_processed_oracle,
                     f'{H.save_dir}/samples-eval_t{temp}.png',
                     logprint,
                     temp=temp)

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
