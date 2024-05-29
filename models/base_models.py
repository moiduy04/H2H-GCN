from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score
# Refer to base_encoder to recover original code
from models.base_encoder import H2HGCN
from models.decoder import NCDecoder
from layers import FermiDiracDecoder, CentroidDistance
from manifolds import LorentzManifold
from utils.eval_utils import acc_f1


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.args = args
        self.encoder = H2HGCN(args)
        self.c = torch.Tensor([1.]).cuda().to(args.device)

    def encode(self, x, adj_list, adj_mask):
        return self.encoder.encode(x, adj_list, adj_mask)

    def get_loss_and_metric(self, embeddings: torch.Tensor, data: dict,
                            data_split: Literal['train', 'test', 'val']) \
            -> dict:
        """
        Computes the loss and performance metric for the given embeddings and data.

        :returns:
            A dictionary containing the performance metric
            and the computed loss for the embeddings.
        """
        raise NotImplementedError

    def init_metric_dict(self) -> dict:
        """
        :returns:
            A dictionary with keys representing the performance metrics and initial values
            set to indicate 'better' performance for any trained model.
        """
        raise NotImplementedError

    def has_improved(self, m1: dict, m2: dict) -> bool:
        """
        Compares two performance metrics.

        Each dictionary contains all valid performance metric keys.
        The function determines if `m2` represents 'better' performance compared to `m1`.

        :returns:
            `True` if `m2` is considered 'better' than `m1`,
            `False` otherwise.
        """
        raise NotImplementedError


class NCModel(BaseModel):
    """
    Model for node classification
    """
    def __init__(self, args):
        super(NCModel, self).__init__(args)
        assert args.n_classes > 0

        self.decoder = NCDecoder(self.c, args)

        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'

        self.weights = torch.Tensor([1.] * args.n_classes)

        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

        self.distance = CentroidDistance(args, 1, LorentzManifold)
    
    def encode(self, x, adj_list, adj_mask):
        node_repr = super().encode(x, adj_list, adj_mask)
        mask = torch.ones((node_repr.size(0),1)).cuda().to(self.args.device)
        _, node_centroid_sim = self.distance(node_repr, mask) 
        return node_centroid_sim.squeeze()

    def decode(self, h, adj_list, idx):
        output = self.decoder.decode(h, adj_list)
        return F.log_softmax(output[idx], dim=1)

    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = F.nll_loss(output, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]


class LPModel(BaseModel):
    """
    Model for link prediction
    """
    def __init__(self, args):
        super(LPModel, self).__init__(args)
        assert args.n_false_edges > 0
        assert args.n_edges > 0

        self.n_false_edges = args.n_false_edges
        self.n_edges = args.n_edges

        self.decoder = FermiDiracDecoder(r=args.r, t=args.t)

    def decode(self, h, idx, split):
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = LorentzManifold.sqdist(emb_in, emb_out, self.c)
        probs = self.decoder.forward(sqdist, split)
        return probs

    def compute_metrics(self, embeddings, data, split):
        if split == 'train':
            edges_false = data[f'{split}_edges_false'][
                np.random.randint(0, self.n_false_edges, self.n_edges)]
        else:
            edges_false = data[f'{split}_edges_false']
        pos_scores = self.decode(embeddings, data[f'{split}_edges'], split)
        neg_scores = self.decode(embeddings, edges_false, split)
        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])
