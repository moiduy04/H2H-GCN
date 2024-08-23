# H2H-GCN
### *(reimplementation)*

The accompanying report can be found [here](https://drive.google.com/file/d/1pyjPereq_i3DwwFBo7wUfhWXk47TVRLO/view?usp=sharing)

### Changes:
- Always encodes embeddings in Lorentz manifold.
- Always trains weight parameters via SGD on the Stiefel manifold.
- Assumes graph only has 1 message type.
- Always ties all layer weights.
- Implements hyperbolic skip connections

### Usage:
Set up environment according to `requirement.txt`.
#### For link prediction, run
```
python train.py \
  --task lp \
  --dataset disease_lp \
  --normalize-feats 0 \
  --seed 1234 \
  --num_runs 10 \
  --epochs 1000 \
  --step_lr_reduce_freq 5000 \
  --patience 1000  \
  --eucl_lr 0.001 \
  --stie_lr 0.001 \
  --dim 256 \
  --num-layers 2 \
  --log-freq 20 \
```
#### For node classification, run
```
python train.py \
  --task nc \
  --dataset disease_nc \
  --seed 1234 \
  --num_runs 10 \
  --epochs 1000 \
  --step_lr_reduce_freq 5000 \
  --eucl_lr 0.01 \
  --stie_lr 0.01 \
  --dim 64 \
  --num-layers 5 \
  --num_centroid 200 \
  --log-freq 20 \
```

#### Optional arguments:  
    --task                  which tasks to train on, 'lp' or 'nc'  
    --dataset               which dataset to use, 'disease_lp' or 'disease_nc'
    --seed                  seed to produce running seeds
    --num_runs              number of runs to test
    --eucl_lr               learning rate for Euclidean parameters  
    --stie_lr               learning rate for the Stiefel parameters  
    --normalize-feats       whether to normalize input node features  
    --epochs                maximum number of epochs  
    --step_lr_reduce_freq   step_size for StepLR scheduler    
    --dim                   embedding dimension  
    --num-layers            number of layers  
    --patience              patience for early stopping (epochs)  
    --num_centroid          number of centroids used in node classification task

#### Directory list: 
       data                     datasets files, including the "disease_lp" and "disease_nc"  
       layers                   include a centroid-based classification and layers used in H2H-GCN
       manifolds                include the Lorentz manifold and the Stiefel manifold
       models                   encoder for graph embedding and decoder for post-processing  
       optimizers               optimizers for orthogonal parameters  
       utils                    utility modules and functions  
       config.py                config file
       train.py                 run this file to start the training  
       requirements.txt         requirements file  
       README.md                README file  


### List of file changes:
- `/manifolds` package:
  - Lorentz and Stiefel manifold's functions are now `classmethod`, the classes themselves no longer take `args` as a param.
  - Added `GeometricTransformation` class handling projections between manifolds. 
- `optimizers.rsgd.RiemannianSGD` takes the manifold directly from `__init__` instead of through `args`.
- `/models` package:
  - `encoder.py` is reimplemented from scratch.
  - Other files in `/models` received minor changes.
- `/utils/visualization.py` package added
- `config.py` has some unused flags removed and some config names changed for clarity.
- `train.py` is reimplemented and modified
