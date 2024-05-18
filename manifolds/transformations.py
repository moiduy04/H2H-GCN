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
        d = x.size(-1) - 1
        return x.narrow(-1, 1, d) / (x.narrow(-1, 0, 1) + 1)

    @staticmethod
    def poincare_to_lorentz(x):
        """
        Transform Poincaré coordinates to Lorentz model coordinates.

        :param x: (batch_size, d)
        """
        x_norm_square = x.pow(2).sum(-1, keepdim=True)
        return torch.cat((1 + x_norm_square, 2 * x), dim=1) \
            / (1 - x_norm_square + LorentzManifold.eps)

    @staticmethod
    def lorentz_to_klein(x):
        """
        Transform Lorentz coordinates to Klein model coordinates.

        :param x: (batch_size, d+1)
        """
        d = x.size(-1) - 1
        return x.narrow(-1, 1, d) / (x.narrow(-1, 0, 1) + 1)

    @staticmethod
    def klein_to_lorentz(x, *, device='cpu'):
        """
        Transform Klein coordinates to Lorentz model coordinates.

        :param x: (batch_size, d)
        :param device
        """
        x_norm_square = x.pow(2).sum(-1, keepdim=True)
        x_norm_square = torch.clamp(x_norm_square, max=0.9)
        tmp = torch.ones((x.size(0), 1)).cuda().to(device)
        tmp1 = torch.cat((tmp, x), dim=1)
        tmp2 = 1.0 / torch.sqrt(1.0 - x_norm_square)
        tmp3 = (tmp1 * tmp2)
        return tmp3

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
