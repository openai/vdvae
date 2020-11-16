# we provide a copy of ffhq-256 for convenience, downsampled using the same function as NVAE (https://github.com/NVlabs/NVAE) (personal communication with author)

# Resizing function is this one, with the default second argument and size=256
# https://pytorch.org/docs/stable/torchvision/transforms.html#torchvision.transforms.Resize

# 5-bit precision is calculated using the following lines, with num_bits=5 and for an x in [0, 1]
# x = torch.floor(x * 255 / 2 ** (8 - num_bits))
# x /= (2 ** num_bits - 1)

# the DMOL loss should also be adjusted to have 32 buckets instead of 256 (this code, or NVAE, can be used as reference)

wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets/ffhq-256.npy