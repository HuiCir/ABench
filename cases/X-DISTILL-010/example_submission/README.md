# X-DISTILL-010 reference submission

`model.py`: loads the distilled SmolLM2-135M weights from `checkpoint/student.safetensors`. `train.py`: 30M tokens of continued training with half cross-entropy, half forward KL to SmolLM2-1.7B. Design report: `report/index.html`; references: `references.yaml`.
