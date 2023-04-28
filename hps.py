import numpy as np

HPARAMS_REGISTRY = {}


class Hyperparams(dict):

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            return None

    def __setattr__(self, attr, value):
        self[attr] = value


cifar10 = Hyperparams()
cifar10.width = 384
cifar10.lr = 0.0002
cifar10.zdim = 16
cifar10.wd = 0.01
cifar10.dec_blocks = "1x1,4m1,4x2,8m4,8x5,16m8,16x10,32m16,32x21"
cifar10.enc_blocks = "32x11,32d2,16x6,16d2,8x6,8d2,4x3,4d4,1x3"
cifar10.warmup_iters = 100
cifar10.dataset = 'cifar10'
cifar10.n_batch = 16
cifar10.ema_rate = 0.9999
HPARAMS_REGISTRY['cifar10'] = cifar10

i32 = Hyperparams()
i32.update(cifar10)
i32.dataset = 'imagenet32'
i32.ema_rate = 0.999
i32.dec_blocks = "1x2,4m1,4x4,8m4,8x9,16m8,16x19,32m16,32x40"
i32.enc_blocks = "32x15,32d2,16x9,16d2,8x8,8d2,4x6,4d4,1x6"
i32.width = 512
i32.n_batch = 8
i32.lr = 0.00015
i32.grad_clip = 200.
i32.skip_threshold = 300.
HPARAMS_REGISTRY['imagenet32'] = i32

i64 = Hyperparams()
i64.update(i32)
i64.n_batch = 4
i64.grad_clip = 220.0
i64.skip_threshold = 380.0
i64.dataset = 'imagenet64'
i64.dec_blocks = "1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12"
i64.enc_blocks = "64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5"
HPARAMS_REGISTRY['imagenet64'] = i64

ffhq_256 = Hyperparams()
ffhq_256.update(i64)
ffhq_256.n_batch = 1
ffhq_256.lr = 0.00015
ffhq_256.dataset = 'ffhq_256'
ffhq_256.num_images_visualize = 2
ffhq_256.num_variables_visualize = 3
ffhq_256.num_temperatures_visualize = 1
ffhq_256.dec_blocks = "1x2,4m1,4x3,8m4,8x4,16m8,16x9,32m16,32x21,64m32,64x13,128m64,128x7,256m128"
ffhq_256.enc_blocks = "256x3,256d2,128x8,128d2,64x12,64d2,32x17,32d2,16x7,16d2,8x5,8d2,4x5,4d4,1x4"
ffhq_256.no_bias_above = 64
ffhq_256.grad_clip = 130.
ffhq_256.skip_threshold = 180.
HPARAMS_REGISTRY['ffhq256'] = ffhq_256

ffhq1024 = Hyperparams()
ffhq1024.update(ffhq_256)
ffhq1024.dataset = 'ffhq_1024'
ffhq1024.data_root = './ffhq_images1024x1024'
ffhq1024.num_images_visualize = 1
ffhq1024.iters_per_images = 25000
ffhq1024.num_variables_visualize = 0
ffhq1024.num_temperatures_visualize = 4
ffhq1024.grad_clip = 360.
ffhq1024.skip_threshold = 500.
ffhq1024.num_mixtures = 2
ffhq1024.width = 16
ffhq1024.lr = 0.00007
ffhq1024.dec_blocks = "1x2,4m1,4x3,8m4,8x4,16m8,16x9,32m16,32x20,64m32,64x14,128m64,128x7,256m128,256x2,512m256,1024m512"
ffhq1024.enc_blocks = "1024x1,1024d2,512x3,512d2,256x5,256d2,128x7,128d2,64x10,64d2,32x14,32d2,16x7,16d2,8x5,8d2,4x5,4d4,1x4"
ffhq1024.custom_width_str = "512:32,256:64,128:512,64:512,32:512,16:512,8:512,4:512,1:512"
HPARAMS_REGISTRY['ffhq1024'] = ffhq1024

bev64 = Hyperparams()
bev64.update(i32)
bev64.n_batch = 4 * 4  # def. BS * additional BS
bev64.width = 64  # Default width (match largest custom_width?)
bev64.lr = 0.00015  # * (4 / 32) * 4 * 2  # num_nodes * additional BS
bev64.grad_clip = 220.0
bev64.skip_threshold = 380.0
bev64.dataset = 'bev64'
bev64.data_root = 'bevs_64px_single'
bev64.dec_blocks = "1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12"
bev64.enc_blocks = "64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5"
bev64.custom_width_str = "64:64,32:64,16:256,8:256,4:512,1:512"  # res:width
HPARAMS_REGISTRY['bev64'] = bev64

bev128 = Hyperparams()
bev128.update(i32)
bev128.n_batch = 2  # 4 * 4  # def. BS * additional BS
bev128.width = 64  # Default width (match largest custom_width?)
bev128.lr = 0.00015  # * (4 / 32) * 4 * 2  # num_nodes * additional BS
bev128.grad_clip = 175.0
bev128.skip_threshold = 280
bev128.dataset = 'bev128'
bev128.data_root = './bevs_128px_single'
# bev128.dec_blocks = "1x1,4m1,4x1,8m1,8x1,16m8,16x1,32m1,32x1,64m32,64x1,128m64,128x1"  # Dummy single filters
# bev128.enc_blocks = "128x1,128d2,64x1,64d2,32x1,32d2,16x1,16d2,8x1,8d2,4x1,4d4,1x1"
bev128.dec_blocks = "1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12,128m64,128x6"
bev128.enc_blocks = "128x5,128d2,64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5"
bev128.custom_width_str = "128:64,64:64,32:64,16:256,8:256,4:512,1:512"  # res:width
bev128.no_bias_above = 128
HPARAMS_REGISTRY['bev128'] = bev128

bev256 = Hyperparams()
bev256.update(bev64)
bev256.n_batch = 2  # * 3  # def. BS * additional BS
bev256.width = 256
bev256.lr = 0.00015  # * (4 / 32) * 4 * 1  # num_nodes * additional BS
bev256.dataset = 'bev256'
bev256.data_train_root = './c_bevs_single'
bev256.data_val_root = './c_bevs_single'
bev256.epochs_per_eval = 1
bev256.epochs_per_eval_save = 1
bev256.dec_blocks = "1x2,4m1,4x3,8m4,8x4,16m8,16x9,32m16,32x21,64m32,64x13,128m64,128x7,256m128"
bev256.enc_blocks = "256x3,256d2,128x8,128d2,64x12,64d2,32x17,32d2,16x7,16d2,8x5,8d2,4x5,4d4,1x4"
# bev256.custom_width_str = "256:64,128:64,64:64,32:64,16:256,8:256,4:512,1:512"  # res:width
# bev256.custom_width_str = "256:128,128:128,64:128,32:256,16:256,8:512,4:512,1:512"  # res:width
bev256.custom_width_str = "256:256,128:256,64:256,32:256,16:256,8:256,4:512,1:512"  # res:width

bev256.no_bias_above = 256
bev256.grad_clip = 130.
bev256.skip_threshold = 180.
HPARAMS_REGISTRY['bev256'] = bev256


def parse_args_and_update_hparams(H, parser, s=None):
    args = parser.parse_args(s)
    valid_args = set(args.__dict__.keys())
    hparam_sets = [x for x in args.hparam_sets.split(',') if x]
    for hp_set in hparam_sets:
        hps = HPARAMS_REGISTRY[hp_set]
        for k in hps:
            if k not in valid_args:
                raise ValueError(f"{k} not in default args")
        parser.set_defaults(**hps)
    H.update(parser.parse_args(s).__dict__)


def add_vae_arguments(parser):
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--port', type=int, default=29500)
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--data_root', type=str, default='./')
    parser.add_argument('--data_train_root', type=str, default='./')
    parser.add_argument('--data_val_root', type=str, default='./')

    parser.add_argument('--desc', type=str, default='test')
    parser.add_argument('--hparam_sets', '--hps', type=str)
    parser.add_argument('--restore_path', type=str, default=None)
    parser.add_argument('--restore_ema_path', type=str, default=None)
    parser.add_argument('--restore_log_path', type=str, default=None)
    parser.add_argument('--restore_optimizer_path', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--dataloader_workers', type=int, default=0)

    parser.add_argument('--ema_rate', type=float, default=0.999)

    parser.add_argument('--enc_blocks', type=str, default=None)
    parser.add_argument('--dec_blocks', type=str, default=None)
    parser.add_argument('--zdim', type=int, default=16)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--custom_width_str', type=str, default='')
    parser.add_argument('--bottleneck_multiple', type=float, default=0.25)

    parser.add_argument('--no_bias_above', type=int, default=64)
    parser.add_argument('--scale_encblock', action="store_true")

    parser.add_argument('--test_eval', action="store_true")
    parser.add_argument('--warmup_iters', type=float, default=0)

    parser.add_argument('--num_mixtures', type=int, default=10)
    parser.add_argument('--grad_clip', type=float, default=200.0)
    parser.add_argument('--skip_threshold', type=float, default=400.0)
    parser.add_argument('--lr', type=float, default=0.00015)
    parser.add_argument('--lr_prior', type=float, default=0.00015)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--wd_prior', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=10000)
    parser.add_argument('--n_batch', type=int, default=32)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.9)

    parser.add_argument('--temperature', type=float, default=1.0)

    parser.add_argument('--iters_per_ckpt', type=int, default=10000)
    parser.add_argument('--iters_per_print', type=int, default=100)
    parser.add_argument('--iters_per_save', type=int, default=10000)
    parser.add_argument('--iters_per_images', type=int, default=10000)
    parser.add_argument('--epochs_per_eval', type=int, default=10)
    parser.add_argument('--epochs_per_probe', type=int, default=None)
    parser.add_argument('--epochs_per_eval_save', type=int, default=20)
    parser.add_argument('--num_images_visualize', type=int, default=8)
    parser.add_argument('--num_variables_visualize', type=int, default=6)
    parser.add_argument('--num_temperatures_visualize', type=int, default=3)
    parser.add_argument('--viz_temps', type=list, default=[0.1, 0.4, 1.0])

    # BEV
    parser.add_argument('--rnd_noise_ratio', type=float, default=0.)
    parser.add_argument('--rotate_samples', action="store_true")
    parser.add_argument('--w_kl_oracle', type=float, default=1.)
    parser.add_argument('--fully_observable', action="store_true")
    parser.add_argument('--fully_observable_pred', action="store_true")
    parser.add_argument('--do_grad_smoothening', action="store_true")
    parser.add_argument('--grad_smoothening_beta',
                        type=float,
                        default=np.log(2))
    parser.add_argument('--regularize_prior', type=float, default=0)
    parser.add_argument('--do_extrapolation', action="store_true")
    parser.add_argument('--do_masking', action="store_true")
    parser.add_argument('--rec_objective', type=str, default="ce")

    return parser
