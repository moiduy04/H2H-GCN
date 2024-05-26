import numpy as np
import torch
import torch.nn as nn

from manifolds import LorentzManifold, StiefelManifold
from manifolds.transformations import GeometricTransformations as P


class H2HGCN(nn.Module):
    """
    A Hyperbolic-to-Hyperbolic Graph Convolutional network.

    IMPORTANT: read README.md for notes on encoder limitations compared to original implementation.
    """

    def __init__(self, args) -> None:
        super(H2HGCN, self).__init__()
        self.args = args

        self.activation = nn.SELU()

        # Linear projection from feat_dim to dim
        self.linear = nn.Linear(args.feat_dim, args.dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.uniform_(self.linear.bias, -1e-4, 1e-4)

        # Initialize message weights (equiv to W_hat)
        self.msg_weight = nn.Parameter(
            torch.zeros([args.dim - 1, args.dim - 1], requires_grad=True)
        )
        nn.init.orthogonal_(self.msg_weight)
        # Actual message weights will be updated every time encode() is called
        self.W = None

        # Add params to args for optimizer access.
        args.eucl_vars.append(self.linear)
        args.stie_vars.append(self.msg_weight)

    def _get_msg_weight(self):
        """
        :return: updated message weight
        """
        # self.msg_weight is W_hat
        _col = torch.zeros((self.args.dim - 1, 1)).cuda().to(self.args.device)
        W = torch.cat((_col, self.msg_weight), dim=1)
        _row = torch.zeros((1, self.args.dim)).cuda().to(self.args.device)
        _row[0, 0] = 1
        W = torch.cat((_row, W), dim=0)
        return W

    def aggregate(self, node_repr, n_nodes, max_neighbours, mask):
        """
        Aggregates all neighbours' messages.
        :param node_repr: (n_nodes * max_neighbours, d) - node_representations of
        all neighbours of every node in the graph.
        :param n_nodes: (int) - number of nodes in the graph
        :param max_neighbours: (int) - maximum amount of neighbours in the graph
        :param mask: (n_nodes, max_neighbours) - binary mask of `adj_list`
        :return: (n_nodes, d) - aggregated node representations
        """
        # Project to klein
        node_repr = P.lorentz_to_klein(node_repr)

        # Get lorentz factor of each node - (n_nodes * max_neighbours,1)
        lorentz_factor = LorentzManifold.lorentz_factor(node_repr, keepdim=True)
        # Mask away nodes that aren't actually neighbours
        lorentz_factor = lorentz_factor * mask.view(-1, 1)
        # Reshape back to (n_nodes, max_neighbours, d-1 or 1)
        node_repr = node_repr.view(n_nodes, max_neighbours, -1)
        lorentz_factor = lorentz_factor.view(n_nodes, max_neighbours, -1)
        # Get the Einstein mid-point
        node_repr = torch.sum(lorentz_factor * node_repr, dim=1, keepdim=True) \
                    / torch.sum(lorentz_factor, dim=1, keepdim=True)
        # squeeze (n_nodes, 1, d-1) to (n_nodes, d-1)
        node_repr = node_repr.squeeze()

        node_repr = P.klein_to_lorentz(node_repr, device=self.args.device)
        return node_repr

    def apply_activation(self, node_repr):
        return P.poincare_to_lorentz(
            self.activation(
                P.lorentz_to_poincare(node_repr)
            )
        )

    def skip_connection(self, node_repr1, node_repr2, *, n_nodes):
        """
        Averages 2 node represenations.
        """
        node_repr = torch.stack((node_repr1, node_repr2), dim=1)
        node_repr = node_repr.view(n_nodes*2, -1)
        # Project to klein
        node_repr = P.lorentz_to_klein(node_repr)

        # Get lorentz factor of each node - (n_nodes * 2,1)
        lorentz_factor = LorentzManifold.lorentz_factor(node_repr, keepdim=True)
        # Reshape back to (n_nodes, 2, d-1 or 1)
        node_repr = node_repr.view(n_nodes, 2, -1)
        lorentz_factor = lorentz_factor.view(n_nodes, 2, -1)
        # Get the Einstein mid-point
        node_repr = torch.sum(lorentz_factor * node_repr, dim=1, keepdim=True) \
                    / torch.sum(lorentz_factor, dim=1, keepdim=True)
        # squeeze (n_nodes, 1, d-1) to (n_nodes, d-1)
        node_repr = node_repr.squeeze()

        node_repr = P.klein_to_lorentz(node_repr, device=self.args.device)
        return node_repr
        pass

    def encode(self, node_repr, adj_list, adj_mask):
        """
        Generates hyperbolic node embeddings. 
        :param node_repr: (n_nodes, feature_dim) - original node representations
        :param adj_list: (n_nodes, max_neighbours) - padded adjacency list
        :param adj_mask: (n_nodes, max_neighbours) - binary mask for `adj_list`
        :return: (n_nodes, d) - Hyperbolic node embeddings
        """
        # Project to Lorentz manifold
        node_repr = self.activation(self.linear(node_repr))
        node_repr = LorentzManifold.exp_map_zero(node_repr)

        # Get message weight
        self.W = self._get_msg_weight()

        for layer in range(self.args.num_layers):
            # TODO: add residual connections?
            if self.args.skip_connections:
                old_node_repr = node_repr

            node_repr = node_repr @ self.W
            # Select neighbours' node representations and aggregate them
            neighbours_node_repr = torch.index_select(node_repr, dim=0, index=adj_list.view(-1))
            node_repr = self.aggregate(neighbours_node_repr,
                                       n_nodes=adj_list.size(0),
                                       max_neighbours=adj_list.size(1),
                                       mask=adj_mask)
            node_repr = self.apply_activation(node_repr)
            if self.args.skip_connections:
                node_repr = self.skip_connection(old_node_repr, node_repr, n_nodes=adj_list.size(0))
            node_repr = LorentzManifold.normalize(node_repr)
        return node_repr