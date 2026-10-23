# K-FUS-002 reference submission

`kernels/`: one file per problem. Problems 7, 69 and 100 keep the library convolution and run their elementwise tail in one Triton pass (`src/ew.py`); the other 17 are the problem's own PyTorch code with `Model` renamed `ModelNew` (a correct floor, not a contender). Design report: `report/index.html`; references: `references.yaml`.
