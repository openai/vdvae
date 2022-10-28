import itertools
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from vae_helpers import (DmolNet, HModule, draw_gaussian_diag_samples,
                         gaussian_analytical_kl, get_1x1, get_3x3)


class Block(nn.Module):

    def __init__(self,
                 in_width,
                 middle_width,
                 out_width,
                 down_rate=None,
                 residual=False,
                 use_3x3=True,
                 zero_last=False):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(
            middle_width, middle_width)
        self.c3 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(
            middle_width, middle_width)
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out,
                               kernel_size=self.down_rate,
                               stride=self.down_rate)
        return out


def parse_layer_string(s):
    layers = []
    for ss in s.split(','):
        if 'x' in ss:
            res, num = ss.split('x')
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif 'm' in ss:
            res, mixin = [int(a) for a in ss.split('m')]
            layers.append((res, mixin))
        elif 'd' in ss:
            res, down_rate = [int(a) for a in ss.split('d')]
            layers.append((res, down_rate))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers


def pad_channels(t, width):
    d1, d2, d3, d4 = t.shape
    empty = torch.zeros(d1, width, d3, d4, device=t.device)
    empty[:, :d2, :, :] = t
    return empty


def get_width_settings(width, s):
    mapping = defaultdict(lambda: width)
    if s:
        s = s.split(',')
        for ss in s:
            k, v = ss.split(':')
            mapping[int(k)] = int(v)
    return mapping


class Encoder(HModule):

    def build(self):
        H = self.H
        self.in_conv = get_3x3(H.image_channels, H.width)
        self.widths = get_width_settings(H.width, H.custom_width_str)
        enc_blocks = []
        blockstr = parse_layer_string(H.enc_blocks)
        for res, down_rate in blockstr:
            use_3x3 = res > 2  # Don't use 3x3s for 1x1, 2x2 patches
            enc_blocks.append(
                Block(self.widths[res],
                      int(self.widths[res] * H.bottleneck_multiple),
                      self.widths[res],
                      down_rate=down_rate,
                      residual=True,
                      use_3x3=use_3x3))
        n_blocks = len(blockstr)
        for b in enc_blocks:
            b.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.enc_blocks = nn.ModuleList(enc_blocks)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.in_conv(x)
        activations = {}
        activations[x.shape[2]] = x
        for block in self.enc_blocks:
            x = block(x)
            res = x.shape[2]
            x = x if x.shape[1] == self.widths[res] else pad_channels(
                x, self.widths[res])
            activations[res] = x
        return activations


class EncoderPostMatch(HModule):

    def build(self):
        H = self.H
        self.in_conv = get_3x3(H.image_channels_post_match, H.width)
        self.widths = get_width_settings(H.width, H.custom_width_str)
        enc_blocks = []
        blockstr = parse_layer_string(H.enc_blocks)
        for res, down_rate in blockstr:
            use_3x3 = res > 2  # Don't use 3x3s for 1x1, 2x2 patches
            enc_blocks.append(
                Block(self.widths[res],
                      int(self.widths[res] * H.bottleneck_multiple),
                      self.widths[res],
                      down_rate=down_rate,
                      residual=True,
                      use_3x3=use_3x3))
        n_blocks = len(blockstr)
        for b in enc_blocks:
            b.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.enc_blocks = nn.ModuleList(enc_blocks)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.in_conv(x)
        activations = {}
        activations[x.shape[2]] = x
        for block in self.enc_blocks:
            x = block(x)
            res = x.shape[2]
            x = x if x.shape[1] == self.widths[res] else pad_channels(
                x, self.widths[res])
            activations[res] = x
        return activations


class DecBlock(nn.Module):

    def __init__(self, H, res, mixin, n_blocks):
        super().__init__()
        self.base = res
        self.mixin = mixin
        self.H = H
        self.widths = get_width_settings(H.width, H.custom_width_str)
        width = self.widths[res]
        use_3x3 = res > 2
        cond_width = int(width * H.bottleneck_multiple)
        self.zdim = H.zdim
        self.enc = Block(width * 2,
                         cond_width,
                         H.zdim * 2,
                         residual=False,
                         use_3x3=use_3x3)
        self.enc_post_match = Block(width * 2,
                                    cond_width,
                                    H.zdim * 2,
                                    residual=False,
                                    use_3x3=use_3x3)
        self.prior = Block(width,
                           cond_width,
                           H.zdim * 2 + width,
                           residual=False,
                           use_3x3=use_3x3,
                           zero_last=True)
        self.z_proj = get_1x1(H.zdim, width)
        self.z_proj.weight.data *= np.sqrt(1 / n_blocks)
        self.resnet = Block(width,
                            cond_width,
                            width,
                            residual=True,
                            use_3x3=use_3x3)
        self.resnet.c4.weight.data *= np.sqrt(1 / n_blocks)
        self.z_fn = lambda x: self.z_proj(x)

    def sample(self, x, acts, mode='train'):

        # qm, qv = self.enc(torch.cat([x, acts], dim=1)).chunk(2, dim=1)

        if mode == 'train':

            # Split into oracle and posterior matching activations
            B = acts.shape[0] // 2
            acts_oracle = acts[:B]
            acts_post_match = acts[B:]

            qm_oracle, qv_oracle = self.enc(torch.cat([x, acts_oracle],
                                                      dim=1)).chunk(2, dim=1)
            qm_post_match, qv_post_match = self.enc_post_match(
                torch.cat([x, acts_post_match], dim=1)).chunk(2, dim=1)

            z = draw_gaussian_diag_samples(qm_oracle, qv_oracle)

            # Posterior matching objective
            kl_post_match = gaussian_analytical_kl(qm_oracle, qm_post_match,
                                                   qv_oracle, qv_post_match)

            qm = qm_oracle
            qv = qv_oracle

        elif mode == 'post_match':

            qm_post_match, qv_post_match = self.enc_post_match(
                torch.cat([x, acts], dim=1)).chunk(2, dim=1)

            z = draw_gaussian_diag_samples(qm_post_match, qv_post_match)

            kl_post_match = torch.zeros_like(z)
            qm = qm_post_match
            qv = qv_post_match

        elif mode == 'oracle':
            qm_oracle, qv_oracle = self.enc(torch.cat([x, acts],
                                                      dim=1)).chunk(2, dim=1)

            z = draw_gaussian_diag_samples(qm_oracle, qv_oracle)

            qm = qm_oracle
            qv = qv_oracle
            kl_post_match = torch.zeros_like(z)

        feats = self.prior(x)
        pm, pv, xpp = feats[:, :self.zdim,
                            ...], feats[:, self.zdim:self.zdim * 2,
                                        ...], feats[:, self.zdim * 2:, ...]
        x = x + xpp

        # Match prior (same order as Vanilla VDVAE)
        # Order: p(x) || q(x) ==> Optimize 'prior' to cover all 'posteriors'
        kl_prior = gaussian_analytical_kl(qm, pm, qv, pv)

        return z, x, kl_prior, kl_post_match

    def sample_uncond(self, x, t=None, lvs=None):
        n, c, h, w = x.shape
        feats = self.prior(x)
        pm, pv, xpp = feats[:, :self.zdim,
                            ...], feats[:, self.zdim:self.zdim * 2,
                                        ...], feats[:, self.zdim * 2:, ...]
        x = x + xpp
        if lvs is not None:
            z = lvs
        else:
            if t is not None:
                pv = pv + torch.ones_like(pv) * np.log(t)
            z = draw_gaussian_diag_samples(pm, pv)
        return z, x

    def get_inputs(self, xs, activations, mode):
        acts = activations[self.base]
        try:
            x = xs[self.base]
        except KeyError:
            x = torch.zeros_like(acts)
        if acts.shape[0] // 2 != x.shape[0] and mode == 'train':
            x = x.repeat(acts.shape[0] // 2, 1, 1, 1)
        elif acts.shape[0] != x.shape[0] and (mode != 'train'):
            x = x.repeat(acts.shape[0], 1, 1, 1)
        return x, acts

    def forward(self, xs, activations, get_latents=False, mode='train'):
        x, acts = self.get_inputs(xs, activations, mode)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...],
                                  scale_factor=self.base // self.mixin)
        z, x, kl_prior, kl_post_match = self.sample(x, acts, mode)
        x = x + self.z_fn(z)
        x = self.resnet(x)
        xs[self.base] = x
        if get_latents:
            return xs, dict(z=z.detach(),
                            kl=kl_prior,
                            kl_post_match=kl_post_match)
        return xs, dict(kl=kl_prior, kl_post_match=kl_post_match)

    def forward_uncond(self, xs, t=None, lvs=None):
        try:
            x = xs[self.base]
        except KeyError:
            ref = xs[list(xs.keys())[0]]
            x = torch.zeros(dtype=ref.dtype,
                            size=(ref.shape[0], self.widths[self.base],
                                  self.base, self.base),
                            device=ref.device)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...],
                                  scale_factor=self.base // self.mixin)
        z, x = self.sample_uncond(x, t, lvs=lvs)
        x = x + self.z_fn(z)
        x = self.resnet(x)
        xs[self.base] = x
        return xs


class Decoder(HModule):

    def build(self):
        H = self.H
        resos = set()
        dec_blocks = []
        self.widths = get_width_settings(H.width, H.custom_width_str)
        blocks = parse_layer_string(H.dec_blocks)
        for idx, (res, mixin) in enumerate(blocks):
            dec_blocks.append(DecBlock(H, res, mixin, n_blocks=len(blocks)))
            resos.add(res)
        self.resolutions = sorted(resos)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.bias_xs = nn.ParameterList([
            nn.Parameter(torch.zeros(1, self.widths[res], res, res))
            for res in self.resolutions if res <= H.no_bias_above
        ])
        self.out_net = DmolNet(H)
        self.gain = nn.Parameter(torch.ones(1, H.width, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, H.width, 1, 1))
        self.final_fn = lambda x: x * self.gain + self.bias

    def forward(self, activations, get_latents=False, mode='train'):
        stats = []
        xs = {a.shape[2]: a for a in self.bias_xs}
        for block in self.dec_blocks:
            xs, block_stats = block(xs,
                                    activations,
                                    get_latents=get_latents,
                                    mode=mode)
            stats.append(block_stats)
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size], stats

    def forward_uncond(self, n, t=None, y=None):
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
        for idx, block in enumerate(self.dec_blocks):
            try:
                temp = t[idx]
            except TypeError:
                temp = t
            xs = block.forward_uncond(xs, temp)
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size]

    def forward_manual_latents(self, n, latents, t=None):
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[2]] = bias.repeat(n, 1, 1, 1)
        for block, lvs in itertools.zip_longest(self.dec_blocks, latents):
            xs = block.forward_uncond(xs, t, lvs=lvs)
        xs[self.H.image_size] = self.final_fn(xs[self.H.image_size])
        return xs[self.H.image_size]


class VAE(HModule):

    def build(self):
        self.encoder = Encoder(self.H)
        self.decoder = Decoder(self.H)

        self.encoder_post_match = EncoderPostMatch(self.H)

        self.w_kl_oracle = self.H.w_kl_oracle

    def inference(self, x):
        acts_post_match = self.encoder_post_match.forward(x)
        px_z, _ = self.decoder.forward(acts_post_match, mode='post_match')
        x_hat = self.decoder.out_net.sample(px_z)

        return x_hat

    def forward(self,
                x_oracle,
                x_post_match,
                x_target,
                mask=None,
                mode='train'):
        '''
        Args:
            x: Interval (-1, 1)
            x_target: Interval (-1, 1)
        '''

        device = x_target.get_device()

        acts_oracle = self.encoder.forward(x_oracle)
        acts_post_match = self.encoder_post_match.forward(x_post_match)

        # Stack activations for each feature map resolution
        activations = {}
        for key in acts_oracle.keys():
            a = acts_oracle[key]
            b = acts_post_match[key]
            activations[key] = torch.concat([a, b])

        px_z, stats = self.decoder.forward(activations, mode=mode)

        # Masked ELBO computation
        if mask is None:
            B, H, W, _ = x_target.shape
            mask = torch.ones((B, H, W))
            mask = mask.to(device)
        distortion_per_pixel = self.decoder.out_net.nll(px_z, x_target, mask)

        rate_per_pixel = torch.zeros_like(distortion_per_pixel)
        rate_per_pixel_post_match = torch.zeros(
            stats[0]['kl_post_match'].shape[0], device=device)
        ndims = np.prod(x_oracle.shape[1:])
        for statdict in stats:
            rate_per_pixel += statdict['kl'].sum(dim=(1, 2, 3))
            rate_per_pixel_post_match += statdict['kl_post_match'].sum(dim=(1,
                                                                            2,
                                                                            3))
        rate_per_pixel /= ndims
        rate_per_pixel_post_match /= ndims

        rate_per_pixel_post_match = self.w_kl_oracle * rate_per_pixel_post_match

        elbo = (distortion_per_pixel +
                rate_per_pixel).mean() + rate_per_pixel_post_match.mean()
        return dict(elbo=elbo,
                    distortion=distortion_per_pixel.mean(),
                    rate=rate_per_pixel.mean(),
                    rate_post_match=rate_per_pixel_post_match.mean())

    def forward_get_latents(self, x, mode='post_match'):
        if mode == 'oracle':
            activations = self.encoder.forward(x)
        elif mode == 'post_match':
            activations = self.encoder_post_match.forward(x)
        _, stats = self.decoder.forward(activations,
                                        get_latents=True,
                                        mode=mode)
        return stats

    def forward_uncond_samples(self, n_batch, t=None):
        px_z = self.decoder.forward_uncond(n_batch, t=t)
        return self.decoder.out_net.sample(px_z)

    def forward_samples_set_latents(
        self,
        n_batch,
        latents,
        t=None,
    ):
        px_z = self.decoder.forward_manual_latents(n_batch, latents, t=t)
        return self.decoder.out_net.sample(px_z)
