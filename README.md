# H2H-GCN
## *(reimplementation)*


### Changes:
- Always encodes embeddings in Lorentz manifold.
- Always trains weight parameters via SGD on the Stiefel manifold.
- Assumes graph only has 1 message type.
- Always ties all layer weights.

### Usage:
#### For link prediction, run
```
python train.py \
  --task lp \
  --dataset disease_lp \
  --normalize-feats 0 \
  --num_runs 1 \
  --epochs 1000 \
  --step_lr_reduce_freq 5000 \
  --patience 1000  \
  --eucl_lr 0.001 \
  --stie_lr 0.001 \
  --dim 256 \
  --num-layers 2 \
  --log-freq 20 \
  --log-to-stdout False
```

#### For node classification, run
```
!python train.py \
  --task nc \
  --dataset disease_nc \
  --num_runs 1 \
  --epochs 1000 \
  --step_lr_reduce_freq 5000 \
  --euch_lr 0.01 \
  --stie_lt 0.01 \
  --dim 64 \
  --num-layers 5 \
  --num_centroid 200 \
  --log-freq 20 \
  --log-to-stdout False
```


### File changes:
- `/models` package:
  - `encoder.py` is reimplemented from scratch.
  - Other files in `/models` received minor changes.
- `train.py` is reimplemented and modified
- `config.py` has some unused flags removed and some config names changed for clarity. 
- `/manifolds` package:
  - Lorentz and Stiefel manifold's functions are now `classmethod`, the classes themselves no longer take `args` as a param.
  - Added `GeometricTransformation` class handling projections between manifolds. 
- `optimizers.rsgd.RiemannianSGD` takes the manifold directly from `__init__` instead of through `args`.
