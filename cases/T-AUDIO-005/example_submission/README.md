# T-AUDIO-005 reference submission

`model.py`: frozen wav2vec 2.0 base, a learned softmax-weighted sum of its hidden states and one linear CTC layer (22,314 trained parameters), greedy CTC decoding. `train.py` reproduces `checkpoint/states.pt`. Design report: `report/index.html`; references: `references.yaml`.
