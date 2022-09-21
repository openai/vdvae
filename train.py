import os
import time

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data import set_up_data
from train_helpers import (accumulate_stats, load_opt, load_vaes, save_model,
                           set_up_hyperparams, update_ema)
from utils import arrange_side_by_side, get_cpu_stats_over_ranks, image_grid


def training_step(H, data_input, target, vae, ema_vae, optimizer, iterate):
    '''
    Args:
        data_input: (B,H,W,C)
    '''
    t0 = time.time()

    vae.zero_grad()

    # B = data_input.shape[0]
    # device = data_input.get_device()

    x_1_prob = data_input[:, :, :, 0:1]  # Extract 'road_present' tensors
    x_2_prob = data_input[:, :, :, 1:2]  # Extract 'road_future' tensors
    x_3_prob = data_input[:, :, :, 2:3]  # Extract 'road_full' tensors

    # Value range (0, 1) --> (-1, +1)
    x_1 = 2 * x_1_prob - 1
    x_2 = 2 * x_2_prob - 1
    x_3 = 2 * x_3_prob - 1

    # x_2_rot = torch.rot90(x_2, 2, [1, 2])
    # x_3_rot = torch.rot90(x_3, 2, [1, 2])
    # x_3_prob_rot = torch.rot90(x_3_prob, 2, [1, 2])

    # x = torch.concat((x_1, x_2_rot, x_3, x_3_rot))
    # x_prob = torch.concat((x_3_prob, x_3_prob_rot, x_3_prob, x_3_prob_rot))
    x_post_match = torch.concat((x_1, x_2))
    x_oracle = torch.concat((x_3, x_3))

    x_prob = torch.concat((x_3_prob, x_3_prob))

    # Target value thresholding
    POS_THRESH = 0.75
    NEG_THRESH = 0.25

    x_prob[x_prob > POS_THRESH] = 1.
    x_prob[x_prob < NEG_THRESH] = 0.

    # if H.fully_observable:
    #     x = torch.concat((x_3, x_3_rot, x_3, x_3_rot))
    #     x_prob = torch.concat((x_3_prob, x_3_prob_rot))
    #     m_pred = torch.ones_like(x_prob, dtype=torch.bool)
    #     m_in = m_pred
    # elif H.fully_observable_pred:
    #     x = torch.concat((x_1, x_2_rot, x_3, x_3_rot))
    #     x_prob = torch.concat((x_3_prob, x_3_prob_rot))
    #     m_pred = torch.ones_like(x_prob, dtype=torch.bool)
    #     m_in = m_pred
    # else:
    m_pred = ~(x_prob == 0.5)
    m_pred[(x_prob < POS_THRESH) & (x_prob > NEG_THRESH)] = False

    # if H.rnd_noise_ratio > 0.:
    #     B, h, w, c = x.shape
    #     # Do not mask the oracle
    #     B = B // 2
    #     mask_prob = H.rnd_noise_ratio * torch.rand(1, device=device)
    #     mask = torch.rand((B, h, w, c), device=device) < mask_prob
    #     dummy_mask = torch.zeros_like(mask, dtype=torch.bool, device=device)
    #     mask = torch.concat((mask, dummy_mask))
    #     x[mask] = 0

    # Add input observability mask
    m_in = ~(x_post_match == 0)
    x_post_match = torch.cat((x_post_match, m_in), dim=-1)

    # x_oracle:     (2B,H,W,1) <-- Duplicates of B samples
    # x_post_match: (2B,H,W,2)
    # x_prob:       (2B,H,W,1)
    # m_pred:       (2B,H,W,1)
    stats = vae.forward(x_oracle, x_post_match, x_prob, m_pred)

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
        # device = data_input.get_device()

        x_1_prob = data_input[:, :, :, 0:1]  # Extract 'road_present' tensors
        x_2_prob = data_input[:, :, :, 1:2]  # Extract 'road_future' tensors
        x_3_prob = data_input[:, :, :, 2:3]  # Extract 'road_full' tensors

        # Value range (0, 1) --> (-1, +1)
        x_1 = 2 * x_1_prob - 1
        x_2 = 2 * x_2_prob - 1
        x_3 = 2 * x_3_prob - 1

        # x_2_rot = torch.rot90(x_2, 2, [1, 2])
        # x_3_prob_rot = torch.rot90(x_3_prob, 2, [1, 2])

        x_post_match = torch.concat((x_1, x_2))
        x_oracle = torch.concat((x_3, x_3))

        x_prob = torch.concat((x_3_prob, x_3_prob))

        # x = x_3
        # x_prob = x_3_prob

        # Target value thresholding
        POS_THRESH = 0.75
        NEG_THRESH = 0.25

        x_prob[x_prob > POS_THRESH] = 1.
        x_prob[x_prob < NEG_THRESH] = 0.
        m_pred = ~(x_prob == 0.5)
        m_pred[(x_prob < POS_THRESH) & (x_prob > NEG_THRESH)] = False

        # Add input observability mask
        m_in = ~(x_post_match == 0)
        x_post_match = torch.cat((x_post_match, m_in), dim=-1)

        stats = ema_vae.forward(x_oracle, x_post_match, x_prob, m_pred)

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
    preprocessed = 2 * preprocessed - 1
    # Remove 'future' sample
    # orig_image = orig_image[:, :, :, 0:1]
    # preprocessed = preprocessed[:, :, :, 0:1]

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

    # Remove 'future' sample
    # viz_batch_original = viz_batch_original[:, :, :, 0:1]
    # viz_batch_processed = viz_batch_processed[:, :, :, 0:1]

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

                # Visualize both 'partial' and 'oracle' samples
                # POINT: 'Oracle' samples used to get 'z' during training
                #        ==> Compare how 'partial' does on its own
                B = viz_batch_original.shape[-1] // 3
                viz_batch_original_obs = viz_batch_original[:, :, :, :B]
                viz_batch_original_oracle = viz_batch_original[:, :, :, 2 * B:]
                viz_batch_processed_obs = viz_batch_processed[:, :, :, :B]
                viz_batch_processed_oracle = viz_batch_processed[:, :, :,
                                                                 2 * B:]

                m_in = ~(viz_batch_processed_obs == 0)
                viz_batch_processed_obs = torch.cat(
                    (viz_batch_processed_obs, m_in), dim=-1)
                # m = torch.ones_like(viz_batch_processed_oracle)
                # viz_batch_processed_oracle = torch.cat(
                #     (viz_batch_processed_oracle, m), dim=-1)

                for temp in [0.1, 0.4, 1.0]:
                    write_images(H,
                                 ema_vae,
                                 viz_batch_original_obs,
                                 viz_batch_processed_obs,
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
    ch = latent_obs[0].shape[-1]
    # Remove observation mask input layer
    viz_batch_processed = viz_batch_processed[:, :, :, 0:1]
    im = np.concatenate(batches, axis=0).reshape(
        (n_rows, mb,
         *viz_batch_processed.shape[1:])).transpose([0, 2, 1, 3, 4]).reshape([
             n_rows * viz_batch_processed.shape[1],
             mb * viz_batch_processed.shape[2], ch
         ])
    logprint(f'printing samples to {fname}')

    cm = plt.get_cmap('viridis')
    im = cm(im)
    im = im[:, :, 0]  # (W,H,C)

    im = (255 * im).astype(np.uint8)

    _, grid_h, grid_w, _ = viz_batch_processed.shape
    im = image_grid(im, grid_h, grid_w)

    imageio.imwrite(fname, im)


def run_test_eval(H, ema_vae, data_test, preprocess_fn, logprint):
    print('evaluating')
    viz_batch_original, viz_batch_processed = get_sample_for_visualization(
        data_test, preprocess_fn, H.num_images_visualize, H.dataset)

    # Visualize both 'partial' and 'oracle' samples
    # POINT: 'Oracle' samples used to get 'z' during training
    #        ==> Compare how 'partial' does on its own
    B = viz_batch_processed.shape[0] // 2
    viz_batch_original_oracle = viz_batch_original[B:]
    viz_batch_processed_oracle = viz_batch_processed[B:]
    viz_batch_processed_obs = viz_batch_processed[:B]
    viz_batch_original_obs = viz_batch_original[:B]

    for temp in [0.1, 0.4, 1.0]:
        write_images(H,
                     ema_vae,
                     viz_batch_original_obs,
                     viz_batch_processed_obs,
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
