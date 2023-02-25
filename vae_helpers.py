import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# DISTR_PARAM_NUM = 3
NUM_CLASSES = 255


@torch.jit.script
def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
    return -0.5 + logsigma2 - logsigma1 + 0.5 * (
        logsigma1.exp()**2 + (mu1 - mu2)**2) / (logsigma2.exp()**2)


@torch.jit.script
def draw_gaussian_diag_samples(mu, logsigma):
    eps = torch.empty_like(mu).normal_(0., 1.)
    return torch.exp(logsigma) * eps + mu


def get_conv(in_dim,
             out_dim,
             kernel_size,
             stride,
             padding,
             zero_bias=True,
             zero_weights=False,
             groups=1,
             scaled=False):
    c = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, groups=groups)
    if zero_bias:
        c.bias.data *= 0.0
    if zero_weights:
        c.weight.data *= 0.0
    return c


def get_3x3(in_dim,
            out_dim,
            zero_bias=True,
            zero_weights=False,
            groups=1,
            scaled=False):
    return get_conv(in_dim,
                    out_dim,
                    3,
                    1,
                    1,
                    zero_bias,
                    zero_weights,
                    groups=groups,
                    scaled=scaled)


def get_1x1(in_dim,
            out_dim,
            zero_bias=True,
            zero_weights=False,
            groups=1,
            scaled=False):
    return get_conv(in_dim,
                    out_dim,
                    1,
                    1,
                    0,
                    zero_bias,
                    zero_weights,
                    groups=groups,
                    scaled=scaled)


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.shape) - 1
    m = x.max(dim=axis, keepdim=True)[0]
    return x - m - torch.log(torch.exp(x - m).sum(dim=axis, keepdim=True))


def const_max(t, constant):
    other = torch.ones_like(t) * constant
    return torch.max(t, other)


def const_min(t, constant):
    other = torch.ones_like(t) * constant
    return torch.min(t, other)


def binary_cross_entropy(x_hat, x, obs_mask=None, eps=1e-12):
    '''
    Args:
        x_hat: Model output tensor (N,C,H,W) in interval (0,1)
        x: Model input tensor (N,C,H,W) in interval (0,1)
    '''
    log_pxz = x * torch.log(x_hat + eps) + (1 - x) * torch.log(1 - x_hat + eps)

    if obs_mask is not None:
        log_pxz = obs_mask * log_pxz

    # Mean over all observed elements
    return log_pxz.sum(dim=(1, 2, 3)) / obs_mask.sum(dim=(1, 2, 3))


def mse(x_hat, x, obs_mask=None):
    mse = (x_hat - x)**2

    if obs_mask is not None:
        mse = obs_mask * mse

    # Mean over all observed elements
    return mse.sum(dim=(1, 2, 3)) / obs_mask.sum(dim=(1, 2, 3))


def get_num_mix_distr_params(num_ch):
    '''
    Returns the number of paramters required for each mixture distribution.

    Ex: RGB (3 ch) w. 10 mixtures
        (3 + 3 + 3 + 1) * 10 = 100
                10 <-- returned

        WM (5 ch) w. 10 mixtures
        (10 + 5 + 5 + 1) * 10 = 210
                21 <-- returned
    '''
    if num_ch == 3:
        num_coeffs = 3
    elif num_ch == 5:
        # num_coeffs = 10  # Fully conditional
        # num_coeffs = 7  # Intensity decoupled
        num_coeffs = 4  # Road only conditional
    else:
        raise NotImplementedError()

    num_mus = num_ch
    num_scales = num_ch
    num_pis = 1

    return num_coeffs + num_mus + num_scales + num_pis, num_coeffs


def unpack_pred_params(pred, ch, num_mix, num_coeffs):
    '''
    Returns a tuple of conditional logistic mixture distribution paramters.

    NOTE: Removes extracted coefficients for easy indexing.

    Args:
        pred: (B,H,W,N)
        ch:
        num_mix:
        num_coeffs:

    Returns:

    '''
    B = pred.shape[0]
    H = pred.shape[1]
    W = pred.shape[2]

    # Mixture probabilities (B,H,W,#mix)
    logit_probs = pred[:, :, :, :num_mix]

    pred = pred[:, :, :, num_mix:]

    # Conditional linear dependence model coefficients (B,H,W,#coeffs,#mix)
    coeffs = pred[:, :, :, :num_coeffs * num_mix]
    coeffs = torch.reshape(coeffs, (B, H, W, num_coeffs, num_mix))
    coeffs = torch.tanh(coeffs)

    pred = pred[:, :, :, num_coeffs * num_mix:]

    # Reshape paramters to channel-wise order
    pred = torch.reshape(pred, (B, H, W, ch, -1))

    # Mean and scale parameters
    means = pred[:, :, :, :, :num_mix]
    log_scales = pred[:, :, :, :, num_mix:2 * num_mix]

    return logit_probs, coeffs, means, log_scales


def conditional_distr_train_5ch(m, c, x):
    '''
    Args:
        m: Mean paramters (B,H,W,C,#mix)
        c: Conditional linar depedence model paramters (B,H,#coeff,#mix)
        x: Target tensor w. duplicated elements in #mix dim (B,H,W,C,#mix)

    Fully conditional p(B|G,R,i,r)p(G|R,i,r)p(R|i,r)p(i|r)p(r)
    m0 = means[:, :, :, 0, :]
    m1 = means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :]
    m2 = means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:,:,:, 2, :] * x[:, :, :, 1, :]
    m3 = means[:, :, :, 3, :] + coeffs[:, :, :, 3, :] * x[:, :, :, 0, :] + coeffs[:,:,:, 4, :] * x[:, :, :, 1, :] + coeffs[:,:,:, 5, :] * x[:, :, :, 2, :]
    m4 = means[:, :, :, 4, :] + coeffs[:, :, :, 6, :] * x[:, :, :, 0, :] + coeffs[:,:,:, 7, :] * x[:, :, :, 1, :] + coeffs[:,:,:, 8, :] * x[:, :, :, 2, :] + coeffs[:,:,:, 9, :] * x[:, :, :, 3, :]

    Road conditional p(B|G,R,r)P(G|R,r)p(R|r)p(i|r)p(r)
    m0 = means[:, :, :, 0, :]
    m1 = means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :]
    m2 = means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :]
    m3 = means[:, :, :, 3, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 0, :] + coeffs[:,:,:, 3, :] * x[:, :, :, 2, :]
    m4 = means[:, :, :, 4, :] + coeffs[:, :, :, 4, :] * x[:, :, :, 0, :] + coeffs[:,:,:, 5, :] * x[:, :, :, 2, :] + coeffs[:,:,:, 6, :] * x[:, :, :, 3, :]

    Road only conditional p(B|r)P(G|r)p(R|r)p(i|r)p(r)
    m0 = means[:, :, :, 0, :]
    m1 = means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :]
    m2 = means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :]
    m3 = means[:, :, :, 3, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 0, :]
    m4 = means[:, :, :, 4, :] + coeffs[:, :, :, 3, :] * x[:, :, :, 0, :]

    '''
    # Fully conditional p(B|G,R,i,r)p(G|R,i,r)p(R|i,r)p(i|r)p(r)
    # m0 = m[:, :, :, 0, :]
    # m1 = m[:, :, :, 1, :] + c[:, :, :, 0, :] * x[:, :, :, 0, :]
    # m2 = m[:, :, :,
    #        2, :] + c[:, :, :, 1, :] * x[:, :, :, 0, :] + c[:, :, :,
    #                                                        2, :] * x[:, :, :,
    #                                                                  1, :]
    # m3 = m[:, :, :,
    #        3, :] + c[:, :, :,
    #                  3, :] * x[:, :, :,
    #                            0, :] + c[:, :, :,
    #                                      4, :] * x[:, :, :,
    #                                                1, :] + c[:, :, :,
    #                                                          5, :] * x[:, :, :,
    #                                                                    2, :]
    # m4 = m[:, :, :,
    #        4, :] + c[:, :, :,
    #                  6, :] * x[:, :, :,
    #                            0, :] + c[:, :, :,
    #                                      7, :] * x[:, :, :,
    #                                                1, :] + c[:, :, :,
    #                                                          8, :] * x[:, :, :,
    #                                                                    2, :] + c[:, :, :,
    #                                                                              9, :] * x[:, :, :,
    #                                                                                        3, :]

    # Road conditional p(B|G,R,r)P(G|R,r)p(R|r)p(i|r)p(r)
    # m0 = m[:, :, :, 0, :]
    # m1 = m[:, :, :, 1, :] + c[:, :, :, 0, :] * x[:, :, :, 0, :]
    # m2 = m[:, :, :, 2, :] + c[:, :, :, 1, :] * x[:, :, :, 0, :]
    # m3 = m[:, :, :,
    #        3, :] + c[:, :, :, 2, :] * x[:, :, :, 0, :] + c[:, :, :,
    #                                                        3, :] * x[:, :, :,
    #                                                                  2, :]
    # m4 = m[:, :, :,
    #        4, :] + c[:, :, :,
    #                  4, :] * x[:, :, :,
    #                            0, :] + c[:, :, :,
    #                                      5, :] * x[:, :, :,
    #                                                2, :] + c[:, :, :,
    #                                                          6, :] * x[:, :, :,
    #                                                                    3, :]

    # Road only conditional p(B|r)P(G|r)p(R|r)p(i|r)p(r)
    m0 = m[:, :, :, 0, :]
    m1 = m[:, :, :, 1, :] + c[:, :, :, 0, :] * x[:, :, :, 0, :]
    m2 = m[:, :, :, 2, :] + c[:, :, :, 1, :] * x[:, :, :, 0, :]
    m3 = m[:, :, :, 3, :] + c[:, :, :, 2, :] * x[:, :, :, 0, :]
    m4 = m[:, :, :, 4, :] + c[:, :, :, 3, :] * x[:, :, :, 0, :]

    m = torch.stack((m0, m1, m2, m3, m4), dim=-2)  # (B,H,W,C,#mix)
    return m


def conditional_distr_inference_5ch(x, c):
    '''
    NOTE: Remember the clamping operation to (-1, 1)

    Args:
        m: Mean paramters (B,H,W,C)
        c: Conditional linar depedence model paramters (B,H,#coeff)

    Fully conditional p(B|G,R,i,r)p(G|R,i,r)p(R|i,r)p(i|r)p(r)
    x0 = x[:, :, :, 0]
    x1 = x[:, :, :, 1] + coeffs[:, :, :, 0] * x0
    x2 = x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:,:,:, 2] * x1
    x3 = x[:, :, :, 3] + coeffs[:, :, :, 3] * x0 + coeffs[:,:,:, 4] * x1 + coeffs[:,:,:, 5] * x2
    x4 = x[:, :, :, 4] + coeffs[:, :, :, 6] * x0 + coeffs[:,:,:, 7] * x1 + coeffs[:,:,:, 8] * x2 + coeffs[:,:,:, 9] * x3

    Road conditional p(B|G,R,r)P(G|R,r)p(R|r)p(i|r)p(r)
    x0 = x[:, :, :, 0]
    x1 = x[:, :, :, 1] + coeffs[:, :, :, 0] * x0
    x2 = x[:, :, :, 2] + coeffs[:, :, :, 1] * x0
    x3 = x[:, :, :, 3] + coeffs[:, :, :, 2] * x0 + coeffs[:,:,:, 3] * x2
    x4 = x[:, :, :, 4] + coeffs[:, :, :, 4] * x0 + coeffs[:,:,:, 5] * x2 + coeffs[:,:,:, 6] * x3

    Road only conditional p(B|r)P(G|r)p(R|r)p(i|r)p(r)
    x0 = x[:, :, :, 0]
    x1 = x[:, :, :, 1] + coeffs[:, :, :, 0] * x0
    x2 = x[:, :, :, 2] + coeffs[:, :, :, 1] * x0
    x3 = x[:, :, :, 3] + coeffs[:, :, :, 2] * x0
    x4 = x[:, :, :, 4] + coeffs[:, :, :, 3] * x0

    '''
    # Fully conditional p(B|G,R,i,r)p(G|R,i,r)p(R|i,r)p(i|r)p(r)
    # x0 = const_min(const_max(x[:, :, :, 0], -1), 1)
    # x1 = const_min(const_max(x[:, :, :, 1] + c[:, :, :, 0] * x0, -1), 1)
    # x2 = const_min(
    #     const_max(x[:, :, :, 2] + c[:, :, :, 1] * x0 + c[:, :, :, 2] * x1, -1),
    #     1)
    # x3 = const_min(
    #     const_max(
    #         x[:, :, :, 3] + c[:, :, :, 3] * x0 + c[:, :, :, 4] * x1 +
    #         c[:, :, :, 5] * x2, -1), 1)
    # x4 = const_min(
    #     const_max(
    #         x[:, :, :, 4] + c[:, :, :, 6] * x0 + c[:, :, :, 7] * x1 +
    #         c[:, :, :, 8] * x2 + c[:, :, :, 9] * x3, -1), 1)

    # Road conditional p(B|G,R,r)P(G|R,r)p(R|r)p(i|r)p(r)
    # x0 = const_min(const_max(x[:, :, :, 0], -1), 1)
    # x1 = const_min(const_max(x[:, :, :, 1] + c[:, :, :, 0] * x0, -1), 1)
    # x2 = const_min(const_max(x[:, :, :, 2] + c[:, :, :, 1] * x0, -1), 1)
    # x3 = const_min(
    #     const_max(x[:, :, :, 3] + c[:, :, :, 2] * x0 + c[:, :, :, 3] * x2, -1),
    #     1)
    # x4 = const_min(
    #     const_max(
    #         x[:, :, :, 4] + c[:, :, :, 4] * x0 + c[:, :, :, 5] * x2 +
    #         c[:, :, :, 6] * x3, -1), 1)

    # Road only conditional p(B|r)P(G|r)p(R|r)p(i|r)p(r)
    x0 = const_min(const_max(x[:, :, :, 0], -1), 1)
    x1 = const_min(const_max(x[:, :, :, 1] + c[:, :, :, 0] * x0, -1), 1)
    x2 = const_min(const_max(x[:, :, :, 2] + c[:, :, :, 1] * x0, -1), 1)
    x3 = const_min(const_max(x[:, :, :, 3] + c[:, :, :, 2] * x0, -1), 1)
    x4 = const_min(const_max(x[:, :, :, 4] + c[:, :, :, 3] * x0, -1), 1)

    x = torch.stack((x0, x1, x2, x3, x4), dim=-1)  # (B,H,W,C)
    return x


def discretized_mix_logistic_loss(x, l, mask, low_bit=False):
    """
    Log-likelihood for mixture of discretized logistics, assumes the data has
    been rescaled to [-1,1] interval.

    Ref: Adapted from https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py

    Structure of prediction tensor 'l':
        l[0:1*#mix] --> Mixture probabilities 'pi'
        Remove 'pi' ==> l = l[#mix:]
        Restruct (B,H,W,200) --> (B,H,W,C,)
        l[1*#mix:2*#mix] --> means
        l[2*#mix:3*#mix] --> log_scales
        l[3*#mix:] --> conditional distr. coefficients

    Args:
        x: Target torch.Tensor() (B,H,W,C) in interval (-1, 1).
        l: Prediction torch.Tensor() (B,H,W,100).
    """
    ##################################
    #  Unpack prediction tensor 'l'
    ##################################
    # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    xs = [s for s in x.shape]
    # predicted distribution, e.g. (B,32,32,100)
    ls = [s for s in l.shape]
    ch = xs[-1]
    # num_params_per_distr = 3 * ch + 1  # [mu, s, w]*C + 1
    num_params_per_distr, num_coeffs = get_num_mix_distr_params(ch)
    nr_mix = int(ls[-1] / num_params_per_distr)
    # here and below: unpacking the params of the mixture of logistics
    # Mixture probabilities 'pi': (B,H,W,#mixtures)

    logit_probs, coeffs, means, log_scales = unpack_pred_params(
        l, ch, nr_mix, num_coeffs)
    log_scales = const_max(log_scales, -7.)

    # logit_probs = l[:, :, :, :nr_mix]
    # Remove 'pi' paramters --> (B,H,W, M' = N-#mixtures)
    # l = l[:, :, :, nr_mix:]
    # Coefficients 'c': (B,H,W, #coeff*#mixtures)
    # coeffs = l[:, :, :, :nr_mix * num_coeffs]
    # coeffs = torch.reshape(coeffs, [xs[0], xs[1], xs[2], num_coeffs, nr_mix])
    # coeffs = torch.tanh(coeffs)
    # Remove 'coefficient' paramters --> (B,H,W, M = M'-#mixtures*10)
    # l = l[:, :, :, nr_mix * num_coeffs:]
    # Reshape paramters channel-wise --> (B,W,C,10+10)
    # l = torch.reshape(l, xs + [nr_mix * DISTR_PARAM_NUM])
    # l = torch.reshape(l, xs + [-1])
    # Extract channel-wise paramters
    # means = l[:, :, :, :, :nr_mix]  # (B,H,W,C,#mix)
    # log_scales = const_max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    # coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    # coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix:])

    #######################################################################
    #  Conditional distributions
    #
    #  Original text
    #    - here and below: getting the means and adjusting them based on
    #      preceding sub-pixels
    #
    #  NOTE: Adds 'target value' as 'r' etc.
    #######################################################################
    x = torch.reshape(x, xs + [1]) + torch.zeros(xs + [nr_mix]).to(
        x.device)  # Expand last dim to each mixture distr?
    means = conditional_distr_train_5ch(means, coeffs, x)

    # ms = []
    # m_1 = torch.reshape(means[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    # ms.append(m_1)
    # for idx in range(1, means.shape[3]):
    #     m_idx = torch.reshape(
    #         means[:, :, :, idx, :] +
    #         coeffs[:, :, :, idx - 1, :] * x[:, :, :, 0, :],
    #         [xs[0], xs[1], xs[2], 1, nr_mix])
    #     ms.append(m_idx)
    # m2 = torch.reshape(
    #     means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :],
    #     [xs[0], xs[1], xs[2], 1, nr_mix])
    # m3 = torch.reshape(
    #     means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] +
    #     coeffs[:, :, :, 2, :] * x[:, :, :, 1, :],
    #     [xs[0], xs[1], xs[2], 1, nr_mix])
    # means = torch.cat(ms, dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    if low_bit:
        plus_in = inv_stdv * (centered_x + 1. / 31.)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1. / 31.)
    else:
        plus_in = inv_stdv * (centered_x + 1. / NUM_CLASSES)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_x - 1. / NUM_CLASSES)
    cdf_min = torch.sigmoid(min_in)
    log_cdf_plus = plus_in - F.softplus(
        plus_in)  # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -F.softplus(
        min_in)  # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(
        mid_in
    )  # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    if low_bit:
        log_probs = torch.where(
            x < -0.999, log_cdf_plus,
            torch.where(
                x > 0.999, log_one_minus_cdf_min,
                torch.where(cdf_delta > 1e-5,
                            torch.log(const_max(cdf_delta, 1e-12)),
                            log_pdf_mid - np.log(15.5))))
    else:
        log_probs = torch.where(
            x < -0.999, log_cdf_plus,
            torch.where(
                x > 0.999, log_one_minus_cdf_min,
                torch.where(cdf_delta > 1e-5,
                            torch.log(const_max(cdf_delta, 1e-12)),
                            log_pdf_mid - np.log(0.5 * NUM_CLASSES))))
    log_probs = log_probs.sum(dim=3) + log_prob_from_logits(logit_probs)
    mixture_probs = torch.logsumexp(log_probs, -1)

    mixture_probs = mixture_probs * mask
    mixture_probs = mixture_probs.sum(dim=[1, 2])
    return -1. * torch.div(mixture_probs, mask.sum(dim=[1, 2]))
    # return -1. * mixture_probs.sum(dim=[1, 2]) / np.prod(xs[1:])


def sample_from_discretized_mix_logistic(l, nr_mix, ch=3):
    ls = [s for s in l.shape]
    xs = ls[:-1] + [ch]
    num_params_per_distr, num_coeffs = get_num_mix_distr_params(ch)
    nr_mix = int(ls[-1] / num_params_per_distr)

    # unpack parameters
    logit_probs, coeffs, means, log_scales = unpack_pred_params(
        l, ch, nr_mix, num_coeffs)

    # logit_probs = l[:, :, :, :nr_mix]
    # l = torch.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * DISTR_PARAM_NUM])
    # sample mixture indicator from softmax
    eps = torch.empty(logit_probs.shape,
                      device=l.device).uniform_(1e-5, 1. - 1e-5)
    amax = torch.argmax(logit_probs - torch.log(-torch.log(eps)), dim=3)
    sel = F.one_hot(amax, num_classes=nr_mix).float()
    sel = torch.reshape(sel, xs[:-1] + [1, nr_mix])  # (B,H,W,1,#mix)
    # select logistic parameters
    # means = (l[:, :, :, :, :nr_mix] * sel).sum(dim=4)
    means = (means * sel).sum(dim=4)
    log_scales = const_max((log_scales * sel).sum(dim=4), -7.)
    coeffs = (coeffs * sel).sum(dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.empty(means.shape, device=means.device).uniform_(1e-5, 1. - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))

    x_out = conditional_distr_inference_5ch(x, coeffs)

    #    # New
    #    if ch == 1:
    #        x0 = const_min(const_max(x[:, :, :, 0], -1.), 1.)
    #        x_out = torch.reshape(x0, xs[:-1] + [1])
    #    if ch == 2:
    #        x0 = const_min(const_max(x[:, :, :, 0], -1.), 1.)
    #        x1 = const_min(const_max(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.),
    #                       1.)
    #        x_out = torch.cat([
    #            torch.reshape(x0, xs[:-1] + [1]),
    #            torch.reshape(x1, xs[:-1] + [1])
    #        ],
    #                          dim=3)
    #    elif ch == 5:
    #        # Condition all other outputs on 'road' (i.e. pred structure) output
    #        # Road
    #        x0 = const_min(const_max(x[:, :, :, 0], -1.), 1.)
    #        # Intensity
    #        # x1 = const_min(const_max(x[:, :, :, 1], -1.), 1.)
    #        x1 = const_min(const_max(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.),
    #                       1.)
    #        # RGB
    #        x2 = const_min(const_max(x[:, :, :, 2], -1.), 1.)
    #        # x2 = const_min(const_max(x[:, :, :, 2] + coeffs[:, :, :, 0] * x0, -1.),
    #        #                1.)
    #        x3 = const_min(const_max(x[:, :, :, 3], -1.), 1.)
    #        # x3 = const_min(const_max(x[:, :, :, 3] + coeffs[:, :, :, 0] * x0, -1.),
    #        #                1.)
    #        x4 = const_min(const_max(x[:, :, :, 4], -1.), 1.)
    #        # x4 = const_min(const_max(x[:, :, :, 4] + coeffs[:, :, :, 0] * x0, -1.),
    #        #                1.)
    #        x_out = torch.cat([
    #            torch.reshape(x0, xs[:-1] + [1]),
    #            torch.reshape(x1, xs[:-1] + [1]),
    #            torch.reshape(x2, xs[:-1] + [1]),
    #            torch.reshape(x3, xs[:-1] + [1]),
    #            torch.reshape(x4, xs[:-1] + [1]),
    #        ],
    #                          dim=3)
    #    # Old
    #    elif ch == 3:
    #        x0 = const_min(const_max(x[:, :, :, 0], -1.), 1.)
    #        x1 = const_min(const_max(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.),
    #                       1.)
    #        x2 = const_min(
    #            const_max(
    #                x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 +
    #                coeffs[:, :, :, 2] * x1, -1.), 1.)
    #        x_out = torch.cat([
    #            torch.reshape(x0, xs[:-1] + [1]),
    #            torch.reshape(x1, xs[:-1] + [1]),
    #            torch.reshape(x2, xs[:-1] + [1])
    #        ],
    #                          dim=3)
    #    else:
    #        raise NotImplementedError()

    return x_out


class HModule(nn.Module):

    def __init__(self, H):
        super().__init__()
        self.H = H
        self.build()


class DmolNet(nn.Module):

    def __init__(self, H):
        super().__init__()
        self.H = H
        self.width = H.width
        self.ch = H.image_channels
        self.rec_objective = H.rec_objective
        # num_mix_distr_params = 3 * self.ch + 1  # [mu, s, w]*C + 1
        num_mix_distr_params, _ = get_num_mix_distr_params(self.ch)
        self.out_conv = get_conv(
            H.width,
            H.num_mixtures * num_mix_distr_params,  # 2,
            kernel_size=1,
            stride=1,
            padding=0)

    def nll(self, px_z, x, mask):
        # x_hat = self.forward(px_z)

        # x_hat_road = x_hat[:, :, :, 0:1]
        # x_hat_int = x_hat[:, :, :, 1:2]
        # x_road = x[:, :, :, 0:1]
        # x_int = x[:, :, :, 1:2]

        # if self.rec_objective == 'ce':
        #     recon_road = -1. * binary_cross_entropy(x_hat_road, x_road, mask)
        #     recon_int = -1. * binary_cross_entropy(x_hat_int, x_int, mask)
        # elif self.rec_objective == 'mse':
        #     recon_road = mse(x_hat_road, x_road, mask)
        #     recon_int = mse(x_hat_int, x_int, mask)
        # else:
        #     raise Exception(
        #         f'Undefined reconstruction objective ({self.rec_objective})')

        # recon_loss = recon_road + recon_int

        # return recon_loss
        return discretized_mix_logistic_loss(x=x,
                                             l=self.forward(px_z),
                                             mask=mask[:, :, :, 0],
                                             low_bit=self.H.dataset
                                             in ['ffhq_256'])

    def forward(self, px_z):
        x_hat = self.out_conv(px_z)
        # x_hat = torch.sigmoid(x_hat)
        return x_hat.permute(0, 2, 3, 1)

    def sample(self, px_z):
        im = sample_from_discretized_mix_logistic(self.forward(px_z),
                                                  self.H.num_mixtures, self.ch)
        x_hat = (im + 1.0) * 127.5
        # x_hat = self.forward(px_z)
        # x_hat = x_hat * 255.
        x_hat = x_hat.detach().cpu().numpy()
        x_hat = np.minimum(np.maximum(0.0, x_hat), 255.0).astype(np.uint8)
        return x_hat
