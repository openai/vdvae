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

    def sample(self, x, acts, train_mode=False):
        qm, qv = self.enc(torch.cat([x, acts], dim=1)).chunk(2, dim=1)
        feats = self.prior(x)
        pm, pv, xpp = feats[:, :self.zdim,
                            ...], feats[:, self.zdim:self.zdim * 2,
                                        ...], feats[:, self.zdim * 2:, ...]
        x = x + xpp
        z = draw_gaussian_diag_samples(qm, qv)

        # Unpack 'partial' and 'oracle' sets
        if train_mode:
            B = qm.shape[0] // 3
            qm_1 = qm[0:B]
            qv_1 = qv[0:B]
            qm_2 = qm[B:2 * B]
            qv_2 = qv[B:2 * B]
            qm_oracle = qm[2 * B:]
            qv_oracle = qv[2 * B:]

            # Match oracle (same order as Denton)
            # Order: ( p(x) || q(x) ) ==> Optimize 'partial' to cover 'oracle'
            kl_oracle_1 = gaussian_analytical_kl(qm_oracle, qm_1, qv_oracle,
                                                 qv_1)
            kl_oracle_2 = gaussian_analytical_kl(qm_oracle, qm_2, qv_oracle,
                                                 qv_2)
            kl_oracle = torch.concat((kl_oracle_1, kl_oracle_2))

        else:
            kl_oracle = torch.zeros_like(qm)

        # Match prior (same order as Vanilla VDVAE)
        # Order: p(x) || q(x) ==> Optimize 'prior' to cover all 'posteriors'
        kl_prior = gaussian_analytical_kl(qm, pm, qv, pv)
        return z, x, kl_prior, kl_oracle

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

    def get_inputs(self, xs, activations):
        acts = activations[self.base]
        try:
            x = xs[self.base]
        except KeyError:
            x = torch.zeros_like(acts)
        if acts.shape[0] != x.shape[0]:
            x = x.repeat(acts.shape[0], 1, 1, 1)
        return x, acts

    def forward(self, xs, activations, get_latents=False, train_mode=False):
        x, acts = self.get_inputs(xs, activations)
        if self.mixin is not None:
            x = x + F.interpolate(xs[self.mixin][:, :x.shape[1], ...],
                                  scale_factor=self.base // self.mixin)
        z, x, kl_prior, kl_oracle = self.sample(x, acts, train_mode)
        x = x + self.z_fn(z)
        x = self.resnet(x)
        xs[self.base] = x
        if get_latents:
            return xs, dict(z=z.detach(), kl=kl_prior, kl_oracle=kl_oracle)
        return xs, dict(kl=kl_prior, kl_oracle=kl_oracle)

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

    def forward(self, activations, get_latents=False, train_mode=False):
        stats = []
        xs = {a.shape[2]: a for a in self.bias_xs}
        for block in self.dec_blocks:
            xs, block_stats = block(xs,
                                    activations,
                                    get_latents=get_latents,
                                    train_mode=train_mode)
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

    def forward(self, x, x_target, mask=None, train_mode=False):
        '''
        Args:
            x: Interval (-1, 1)
            x_target: Interval (-1, 1)
        '''

        device = x_target.get_device()

        activations = self.encoder.forward(x)
        px_z, stats = self.decoder.forward(activations, train_mode=train_mode)

        # Masked ELBO computation
        if mask is None:
            B, H, W, _ = x_target.shape
            mask = torch.ones((B, H, W))
            mask = mask.to(device)
        distortion_per_pixel = self.decoder.out_net.nll(px_z, x_target, mask)

        rate_per_pixel = torch.zeros_like(distortion_per_pixel)
        rate_per_pixel_oracle = torch.zeros(stats[0]['kl_oracle'].shape[0],
                                            device=device)
        ndims = np.prod(x.shape[1:])
        for statdict in stats:
            rate_per_pixel += statdict['kl'].sum(dim=(1, 2, 3))
            rate_per_pixel_oracle += statdict['kl_oracle'].sum(dim=(1, 2, 3))
        rate_per_pixel /= ndims
        rate_per_pixel_oracle /= ndims

        elbo = (distortion_per_pixel +
                rate_per_pixel).mean() + rate_per_pixel_oracle.mean()
        return dict(elbo=elbo,
                    distortion=distortion_per_pixel.mean(),
                    rate=rate_per_pixel.mean(),
                    rate_oracle=rate_per_pixel_oracle.mean())

    def forward_get_latents(self, x):
        activations = self.encoder.forward(x)
        _, stats = self.decoder.forward(activations, get_latents=True)
        return stats

    def forward_uncond_samples(self, n_batch, t=None):
        px_z = self.decoder.forward_uncond(n_batch, t=t)
        return self.decoder.out_net.sample(px_z)

    def forward_samples_set_latents(self, n_batch, latents, t=None):
        px_z = self.decoder.forward_manual_latents(n_batch, latents, t=t)
        return self.decoder.out_net.sample(px_z)
