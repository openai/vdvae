import gzip
import pickle

import torch

from data import set_up_data
from train_helpers import load_vaes, set_up_hyperparams


def read_compressed_pickle(path):
    try:
        with gzip.open(path, "rb") as f:
            pkl_obj = f.read()
            obj = pickle.loads(pkl_obj)
            return obj
    except IOError as error:
        print(error)


class vdvaeInferenceModule():

    def __init__(self):
        '''
        Initializes a model using the VDVAE 'hps.py' file and command line
        arguments.
        '''
        H, logprint = set_up_hyperparams()
        H, data_train, data_valid_or_test, preprocess_fn = set_up_data(H)
        _, ema_vae = load_vaes(H, logprint)
        self.vae = ema_vae
        # Setting eval mode reduces prediction variance?
        self.vae.eval()

    def forward(self, x: torch.Tensor, temp: float = 1.) -> torch.Tensor:
        '''
        Args:
            x: Partially observed 'road' and 'intensity' representation w. dim
               (B,H,W,C).
               Value intervals
               'road' (-1,1)
               'int'  (0, 1)

        Returns:
            x_hat: Predicted 'road' and 'intensity representation w. dim
                   (B,H,W,C).
        '''
        # (B,C,H,W) --> (B,H,W,C)
        x = torch.permute(x, (0, 2, 3, 1))
        x_hat = self.vae.inference(x, temp)
        x_hat = torch.tensor(x_hat, dtype=torch.float)
        # Range (0, 255) --> (0, 1)
        x_hat /= 255.
        # (B,H,W,C) --> (B,C,H,W)
        x_hat = torch.permute(x_hat, (0, 3, 1, 2))

        return x_hat


if __name__ == '__main__':

    vdvae = vdvaeInferenceModule()

    x = read_compressed_pickle('x_post_match.pkl.gz')

    x_hat = vdvae.forward(x)
