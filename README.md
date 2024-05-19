# H2H-GCN
## *(reimplementation)*


# Plan:

### Notes on reimplementation:
- Encoder
  - Always encodes embeddings in Lorentz manifold.
  - Always trains weight parameters via SGD on the Stiefel manifold.
  - Assumes graph only has 1 message type.
  - Always ties all layer weights.
- Model
  - Only supports Link Precdiction and Node classification tasks

### File changes:
- `/models` package is reimplemented from scratch.
- `train.py` is reimplemented and modified
- `config.py` has unused flags removed and some config names changed for clarity. 
- `/manifolds` package:
  - Lorentz and Stiefel manifold's functions are now `classmethod`, the classes themselves no longer take `args` as a param.
  - Added `GeometricTransformation` class handling projections between manifolds. 
- `optimizers.rsgd.RiemannianSGD` takes the manifold directly from `__init__` instead of through `args`.
