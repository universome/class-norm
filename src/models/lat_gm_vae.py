import numpy as np

from .lat_gm import LatGM
from .feat_vae import FeatVAE
from .classifier import ZSClassifier, ResnetClassifier


class LatGMVAE(LatGM):
    def __init__(self, config, attrs: np.ndarray=None):
        super(LatGMVAE, self).__init__(config, attrs)

    def init_gm(self):
        if self.config.use_attrs_in_vae:
            self.vae = FeatVAE(self.config, self.attrs.cpu().numpy())
        else:
            self.vae = FeatVAE(self.config)
