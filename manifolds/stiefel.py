import torch
import numpy as np


class StiefelManifold:
    eps = 1e-3
    norm_clip = 1
    max_norm = 1e3

    @classmethod
    def normalize(cls, w):
        return w

    @classmethod
    def symmetric(cls, A):
        return 0.5 * (A + A.t())

    @classmethod
    def rgrad(cls, A, B):
        out = B - A.mm(cls.symmetric(A.transpose(0, 1).mm(B)))
        return out

    @classmethod
    def exp_map_x(cls, A, ref):
        data = A + ref
        Q, R = data.qr()
        # To avoid (any possible) negative values in the output matrix, we multiply the negative
        # values by -1
        sign = (R.diag().sign() + 0.5).sign().diag()
        out = Q.mm(sign)
        return out
