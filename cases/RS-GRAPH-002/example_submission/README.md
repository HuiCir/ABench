# RS-GRAPH-002 reference submission

`train.py` trains implicit-ALS factors in NumPy (`--dev` for the self-check split); `model.py` scores all items and returns the top 20 per user. The checkpoint is built by `train.py`, not shipped. Design report: `report/index.html`; references: `references.yaml`.
