"""Graph decoders."""
import manifolds
import torch.nn as nn
import torch.nn.functional as F
from layers.layers import Linear


class BaseNCDecoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """
    def __init__(self, c):
        super(BaseNCDecoder, self).__init__()
        self.c = c

    def decode(self, x, adj):
        if self.decode_adj:
            input = (x, adj)
            probs, _ = self.cls.forward(input)
        else:
            probs = self.cls.forward(x)
        return probs


class NCDecoder(BaseNCDecoder):
    """
    Decoder abstract class for node classification tasks.
    """
    def __init__(self, c, args):
        super(NCDecoder, self).__init__(c)
        self.input_dim = args.num_centroid
        self.output_dim = args.n_classes
        act = lambda x: x
        self.cls = Linear(args, self.input_dim, self.output_dim, 0.0, act, args.bias)
        self.decode_adj = False

    def decode(self, x, adj):
        h = x
        return super(NCDecoder, self).decode(h, adj)
