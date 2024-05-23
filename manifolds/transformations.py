import torch

from manifolds import LorentzManifold


class GeometricTransformations:
    """
    Provides isometric and isomorphic bijections between Lorentz (l), Klein (k) and Poincaré (p)
    hyperbolic models.
    """

    @staticmethod
    def lorentz_to_poincare(x):
        """
        Transform Lorentz coordinates to Poincaré model coordinates.

        :param x: (batch_size, d+1)
        """
        return x.narrow(-1, 1, x.size(-1)-1) / (x.narrow(-1, 0, 1) + 1)

    @staticmethod
    def poincare_to_lorentz(b):
        """
        Transform Poincaré coordinates to Lorentz model coordinates.

        :param b: (batch_size, d)
        """
        b_norm_square = b.pow(2).sum(-1, keepdim=True)
        x = torch.cat((1 + b_norm_square, 2 * b), dim=1) \
            / (1 - b_norm_square + LorentzManifold.eps)
        return x

    @staticmethod
    def lorentz_to_klein(x):
        """
        Transform Lorentz coordinates to Klein model coordinates.

        :param x: (batch_size, d+1)
        """
        return x.narrow(-1, 1, x.size(-1)-1) / (x.narrow(-1, 0, 1))

    @staticmethod
    def klein_to_lorentz(k, *, device='cpu'):
        """
        Transform Klein coordinates to Lorentz model coordinates.

        :param k: (batch_size, d)
        :param device
        """
        k_norm_square = k.pow(2).sum(-1, keepdim=True)
        k_norm_square = torch.clamp(k_norm_square, max=0.9)
        ones = torch.ones((k.size(0), 1)).cuda().to(device)
        tmp1 = torch.cat((ones, x), dim=1)
        tmp2 = 1.0 / torch.sqrt(1.0 - k_norm_square)
        x = (tmp1 * tmp2)
        return x

    @staticmethod
    def poincare_to_klein(x):
        """
        Transform Poincaré coordinates to Klein model coordinates.
        """
        raise NotImplementedError("Not used in H2H-GCN architecture")

    @staticmethod
    def klein_to_poincare(x):
        """
        Transform Klein coordinates to Poincaré model coordinates.
        """
        raise NotImplementedError("Not used in H2H-GCN architecture")
