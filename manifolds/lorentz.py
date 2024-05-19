import torch
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function, Variable
import torch
from utils import *
from utils.pre_utils import *
from manifolds import *
from utils.math_utils import arcosh, cosh, sinh


class LorentzManifold:
    _eps = 1e-10

    eps = 1e-3
    norm_clip = 1
    max_norm = 1e3

    @staticmethod
    def minkowski_dot(x, y, keepdim=True):
        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
        if keepdim:
            res = res.view(res.shape + (1,))
        return res

    @classmethod
    def sqdist(cls, x, y, c):
        K = 1. / c
        prod = cls.minkowski_dot(x, y)
        eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        theta = torch.clamp(-prod / K, min=1.0 + eps[x.dtype])
        sqdist = K * arcosh(theta) ** 2
        return torch.clamp(sqdist, max=50.0)

    @staticmethod
    def lorentz_factor(x, *, dim=-1, keepdim=False):
        """
            Calculate Lorentz factors
        """
        x_norm = x.pow(2).sum(dim=dim, keepdim=keepdim)
        # TODO: test the effects of the clipping function.
        #   Appears to have no discernible effect? (further testing needed)
        x_norm = torch.clamp(x_norm, 0, 0.9)
        tmp = 1 / torch.sqrt(1 - x_norm)
        return tmp

    @staticmethod
    def ldot(u, v, keepdim=False):
        """
        Lorentzian Scalar Product
        Args:
            u: [batch_size, d + 1]
            v: [batch_size, d + 1]
        Return:
            keepdim: False [batch_size]
            keepdim: True  [batch_size, 1]
        """
        d = u.size(1) - 1
        uv = u * v
        uv = torch.cat((-uv.narrow(1, 0, 1), uv.narrow(1, 1, d)), dim=1)
        return torch.sum(uv, dim=1, keepdim=keepdim)

    @classmethod
    def distance(cls, u, v):
        d = -LorentzDot.apply(u, v)
        dis = Acosh.apply(d, cls.eps)
        return dis

    @classmethod
    def normalize(cls, w):
        """
        Normalize vector such that it is located on the Lorentz
        Args:
            w: [batch_size, d + 1]
        """
        d = w.size(-1) - 1
        narrowed = w.narrow(-1, 1, d)
        if cls.max_norm:
            narrowed = torch.renorm(narrowed.view(-1, d), 2, 0, cls.max_norm)
        first = 1 + torch.sum(torch.pow(narrowed, 2), dim=-1, keepdim=True)
        first = torch.sqrt(first)
        tmp = torch.cat((first, narrowed), dim=1)
        return tmp

    @classmethod
    def exp_map_zero(cls, v):
        zeros = torch.zeros_like(v)
        zeros[:, 0] = 1
        return cls.exp_map_x(zeros, v)

    @classmethod
    def exp_map_x(cls, p, d_p, d_p_normalize=True, p_normalize=True):
        if d_p_normalize:
            d_p = cls.normalize_tan(p, d_p)

        ldv = cls.ldot(d_p, d_p, keepdim=True)
        nd_p = torch.sqrt(torch.clamp(ldv + cls.eps, _eps))

        t = torch.clamp(nd_p, max=cls.norm_clip)
        newp = (torch.cosh(t) * p) + (torch.sinh(t) * d_p / nd_p)

        if p_normalize:
            newp = cls.normalize(newp)
        return newp

    @staticmethod
    def normalize_tan(x_all, v_all):
        d = v_all.size(1) - 1
        x = x_all.narrow(1, 1, d)
        xv = torch.sum(x * v_all.narrow(1, 1, d), dim=1, keepdim=True)
        tmp = 1 + torch.sum(torch.pow(x_all.narrow(1, 1, d), 2), dim=1, keepdim=True)
        tmp = torch.sqrt(tmp)
        return torch.cat((xv / tmp, v_all.narrow(1, 1, d)), dim=1)

    @classmethod
    def log_map_zero(cls, y):
        zeros = torch.zeros_like(y)
        zeros[:, 0] = 1
        return cls.log_map_x(zeros, y)

    @classmethod
    def log_map_x(cls, x, y, normalize=False):
        """Logarithmic map on the Lorentz Manifold"""
        xy = cls.ldot(x, y).unsqueeze(-1)
        tmp = torch.sqrt(torch.clamp(xy * xy - 1 + cls.eps, _eps))
        v = Acosh.apply(-xy, cls.eps) / (
            tmp
        ) * torch.addcmul(y, xy, x)
        if normalize:
            result = cls.normalize_tan(x, v)
        else:
            result = v
        return result

    @classmethod
    def parallel_transport(cls, x, y, v):
        """Parallel transport for Lorentz"""
        v_ = v
        x_ = x
        y_ = y

        xy = cls.ldot(x_, y_, keepdim=True).expand_as(x_)
        vy = cls.ldot(v_, y_, keepdim=True).expand_as(x_)
        vnew = v_ + vy / (1 - xy) * (x_ + y_)
        return vnew


class LorentzDot(Function):
    @staticmethod
    def forward(ctx, u, v):
        ctx.save_for_backward(u, v)
        return LorentzManifold.ldot(u, v)

    @staticmethod
    def backward(ctx, g):
        u, v = ctx.saved_tensors
        g = g.unsqueeze(-1).expand_as(u).clone()
        g.narrow(-1, 0, 1).mul_(-1)
        return g * v, g * u


class Acosh(Function):
    @staticmethod
    def forward(ctx, x, eps):
        z = torch.sqrt(torch.clamp(x * x - 1 + eps, _eps))
        ctx.save_for_backward(z)
        ctx.eps = eps
        xz = x + z
        tmp = torch.log(xz)
        return tmp

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        z = torch.clamp(z, min=ctx.eps)
        z = g / z
        return z, None
