# X-SFT-005 reference submission

`model.py`, `src/lora.py`: SmolLM2-360M with rank-16 LoRA on every projection (about 8.7M trained numbers), merged into the base weights at load time. `train.py` reproduces `checkpoint/` (one epoch over 50,000 public conversations, loss on assistant turns). Design report: `report/index.html`; references: `references.yaml`.
