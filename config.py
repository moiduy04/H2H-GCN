import argparse

from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'stie_lr': (0.1, 'learning rate for Stiefel parameters'),
        'eucl_lr': (0.01, 'learning rate for Euclidean parameters'),
        'cuda': (0, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (5000, 'maximum number of epochs to train for'),
        'weight-decay': (0., 'l2 regularization strength'),
        'momentum': (0.999, 'momentum in Euclidean optimizer'),
        'patience': (100, 'patience for early stopping'),

        'seed': (1234, 'seed for creating training seed'),
        'num-runs': (1, 'number of training seeds to create and run'),

        'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
        'log-to-stdout': (False, 'should the logger output immediately to stdout'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'test-freq': (1, 'how often to compute test metrics (in epochs)'),
        'save': (0, '1 to save model and logs and 0 otherwise'),
        'save-dir': (
            None, 'path to save training logs and model weights (defaults to logs/task/date/run/)'),
        'gamma': (0.5, 'gamma for lr scheduler'),
        'print-epoch': (True, ''),
        'grad-clip': (None, 'max norm for gradient clipping, or None for no gradient clipping'),
        'min-epochs': (100, 'do not early stop before min-epochs'),
        # lr_scheduler is always StepLR
        'step_lr_gamma': (0.1, 'gamma for StepLR scheduler'),
        'step_lr_reduce_freq': (500, 'step size for StepLR scheduler'),

        'weight_decay': (0.0, 'weight decay'),
        'proj_init': ('xavier', 'the way to initialize parameters'),
        'num_centroid': (200, 'number of centroids'),
        'feat_dim': (1, 'input feature dimensionality',),
        'pre_trained': (False, 'whether use pre-train model'),
    },
    'model_config': {
        'task': ('lp', 'which tasks to train on, can be any of [lp, nc]'),
        'dim': (128, 'embedding dimension'),
        'c': (1.0, 'hyperbolic radius, set to None for trainable curvature'),
        'r': (2., 'fermi-dirac decoder parameter for lp'),
        't': (1., 'fermi-dirac decoder parameter for lp'),
        'pretrained-embeddings': (None, 'path to pretrained embeddings'),
        'num-layers': (2, 'number of GNN layers'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'double-precision': (False, 'whether to use double precision')
    },
    'data_config': {
        'dataset': ('disease_lp', 'which dataset to use'),
        'val-prop': (0.05, 'proportion of validation edges for link prediction'),
        'test-prop': (0.1, 'proportion of test edges for link prediction'),
        'use-feats': (1, 'whether to use node features or not'),
        'normalize-feats': (1, 'whether to normalize input node features'),
        'normalize-adj': (1, 'whether to row-normalize the adjacency matrix'),
    },
    'Vars added to get args.{param} for better utility': {
        'stie_vars': ([], 'Stiefel parameters'),
        'eucl_vars': ([], 'Euclidean parameters'),
        'split-seed': (0, 'seed for data splits (train/test/val), '
                             'automatically set by args.seed'),
        'n_classes': (0, 'automatically set based on task and dataset'),
        'n_edges': (0, 'automatically set based on task and dataset'),
        'n_false_edges': (0, 'automatically set based on task and dataset'),
        'n_nodes': (-1, 'number of nodes in data (used for debug)')
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
