# T-AUDIO-005: Speech recognition from a frozen wav2vec 2.0 encoder with a light trained head: word error rate on clean, harder and noisy unseen speakers

Anchor: Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli, *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations* (arXiv:2006.11477). Baseline used for comparison: a learned softmax-weighted sum of the frozen encoder's 13 hidden states and one linear CTC layer, greedy decoding (the reference submission; its code is shown under `interface/baseline/`).

## Task

Design a **light trainable head on top of the frozen pre-trained wav2vec 2.0 base encoder** — layer weighting, adapters between frozen layers, a small recurrent or attention head, CTC or another alignment-free objective, decoding — that **transcribes speech of unseen speakers** with the **lowest word error rate**, averaged over clean speech, harder speech and speech with background talkers, after training on **10 hours** of transcribed audio.

- Submit the trained tensors (`checkpoint/states.pt`, at most 20 million numbers), the code that uses them with the frozen encoder, and the training program. The evaluator loads the encoder itself, gives your `load()` the tensors, and calls `transcribe()` on held-out utterances; it never retrains.
- The encoder's weights may not be changed: anything trained must be in `states.pt`, which also counts as the trained-parameter cost.
- No language model or text corpus other than the training transcripts may be used.
- You may build on any public work (SUPERB-style probing heads, adapters, CTC and its variants, noise augmentation, self-training on the training audio), but the design report must state what you borrowed and what you changed.

### Workflow

1. **Research (throughout).** Study the anchor and related work, and come back to the literature whenever your experiments raise a question (an unexpected failure, a bottleneck, a result you cannot explain). The sandbox is online for papers, code and documentation, and the offline paper search `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"` and `... show <arXiv id>` answers quickly from arXiv titles and abstracts. Searches and connections are logged with your run.
2. **Synthesise.** Work out what existing methods do, where they fall short for this task and which gap you will address, and keep this up to date as you learn. It becomes Section 2 (Related Work) of the design report.
3. **Design.** Propose an improvement over the anchor and the baseline: at least one change you can state precisely and point to in your code. List every change, with its code location, in the report's "Changes relative to the anchor" list (`<ol id="changes">`); a report without it fails the format check.
4. **Develop and verify.** Implement it, write a `train.py` that produces your checkpoint from the provided data, train in the sandbox, check the checkpoint on the dev split with `selfcheck.py`, and iterate as your budget allows. The reference code in `interface/baseline/` (model code and `train.py`) runs as is and is a starting point. The checkpoint you submit is exactly what the evaluator scores; it does not retrain.
5. **Submit** the deliverables below. Submission is one-shot.

## Data

`$ABENCH_DATA_DIR/train_10h.npz` (read-only): 10 hours of LibriSpeech train-clean-100 (Panayotov et al. 2015, CC BY 4.0; parquet export `openslr/librispeech_asr` @71cacbfb), whole speakers in shard order (about 80); `dev_clean.npz`: LibriSpeech dev-clean (5.4 hours, 40 speakers). Each holds `audio` (int16, 16 kHz, all utterances concatenated), `audio_offsets` (utterance i is `audio[offsets[i]:offsets[i+1]]`), packed transcripts `text_utf8`/`text_offsets` (upper-case letters, apostrophes, spaces; `interface/textpack.py`) and `speaker`. `$ABENCH_MODELS_DIR/wav2vec2-base/`: the pre-trained (not fine-tuned) encoder `facebook/wav2vec2-base` @0b5b8e86 (Apache-2.0).

The held-out set is not in the sandbox: LibriSpeech test-clean (2,620 utterances), test-other (2,939, harder recordings and speakers) and test-clean mixed with three-speaker babble from train-other-500 at 5–15 dB signal-to-noise ratio (mixtures drawn with a secret seed). Every held-out speaker is absent from the training data.

## Interface

Implement `load(states, models_dir, device)` and `transcribe(state, audios) -> list[str]` in `model.py`: `states` is the dict of tensors from `checkpoint/states.pt`, `audios` a list of int16 NumPy arrays at 16 kHz; return one transcript per utterance. Write a `train.py` that produces `checkpoint/states.pt`.

The full reference is `interface/interface.md`. `abench_stage.py`, `stages.py` and `metrics.py` in `interface/` are the evaluator's own pipeline and metric code; `selfcheck.py` runs them on a dev split.

## Metrics

| Metric | Definition | Requirement |
|---|---|---|
| **Mean WER** (primary) | Word error rate (word-level Levenshtein distance / reference words, after mapping transcripts to upper-case letters, apostrophes and single spaces) on test-clean, test-other and noisy test-clean, averaged over the three; lower is better | — |
| WER and CER per set | The three word and character error rates | reported |
| Real-time factor | Time of all `transcribe()` calls / audio duration, batches of at most 160 s of audio sorted by length, on the reference GPU | reported |
| Trained parameters | Numbers in `states.pt` | Threshold: ≤ 20,000,000 |
| Peak memory | Peak GPU memory during transcription | reported |

**Measurement and decision**

- One stage in a new process on the reference GPU; loading is killed at 300 s and transcription of all 8,179 utterances at 1,800 s.
- Valid submissions are ranked by mean WER; differences below 0.3 points are treated as ties.
- The final ABench score is driven mainly by mean WER, and also considers how different the method is from existing ones and the quality of the sources cited in the design report; the exact weights are not published.

## Evaluation

1. When your session ends, the sandbox collects only the allowed files from `ABENCH_SUBMISSION_DIR`; the workspace is deleted.
2. The evaluator runs outside the sandbox: it checks the submission format, then loads your submitted checkpoint with your model code and runs inference through the stages in `interface/stages.py`, once, on the evaluation hardware; it never retrains your model to evaluate it (`train.py` is only re-run in the audit below).
3. Held-out inputs are copied into the run directory only after the timed phase has ended, and never with their labels or reference values; those stay in the evaluator process, which computes the metrics with `interface/metrics.py`.
4. Audit: the operator re-runs your `train.py` from scratch for the top-ranked submissions and for a random sample of the others. If the re-trained checkpoint's primary metric falls outside your submitted checkpoint's 95% interval, widened by the run-to-run spread the operator measures for the reference `train.py`, the submission is invalid.
5. You receive no evaluation results during your session.

## Sandbox Guide

The full guide is `interface/sandbox_guide.md`. Key points:

- One 24GB reference GPU (RTX 4090 class); the evaluation uses the same GPU model. Available software: Python 3, PyTorch (CUDA), torchaudio, transformers, NumPy; your submission may depend only on it (packages you install while developing are not available to the evaluator).
- Network: online for research: papers, code, documentation and PyPI are reachable through the sandbox proxy (already set in `HTTPS_PROXY`) at any point of your work; dataset and model hubs and this case's own data sources are blocked, and every connection is logged with your run.
- The session lasts at most 8 hours; the deadline is in `$ABENCH_DEADLINE`. Compute during development is capped at 6 GPU-hours; the runner meters it (GPU-busy time, CPU time of your processes) and a run that exceeds it is not scored.
- Data is read-only under `$ABENCH_DATA_DIR`, frozen pretrained files under `$ABENCH_MODELS_DIR`. Write code and run experiments in `$ABENCH_WORKSPACE`; put the final files in `$ABENCH_SUBMISSION_DIR`.
- Offline paper search (arXiv titles and abstracts, fast, logged): `python3 "$ABENCH_VIEW_DIR/tools/abench_papers.py" search "<query>"`.
- Self-check: `python3 "$ABENCH_VIEW_DIR/interface/selfcheck.py" --quick` (interface, small slice), then without `--quick` on the full dev split with your checkpoint.

## Deliverables

```
$ABENCH_SUBMISSION_DIR/
├── model.py            # required: load() and transcribe()
├── train.py            # required: reproduces checkpoint/states.pt
├── src/                # optional: other .py modules
├── checkpoint/
│   └── states.pt       # required: {name: tensor}, at most 20 million numbers
├── README.md           # optional
├── report/index.html   # required: design report
└── references.yaml     # required: reference register
```

- The code area may only contain `.py`, `.yaml`, `.yml`, `.md` and `.txt` files, at most 200,000 bytes in total.
- Write the design report following the structure of `templates/report_template.html`: related work and the gap you address, the changes relative to the anchor (`<ol id="changes">`, each with its code location), method definition, algorithm, complexity and design rationale. The report explains the design only; it does not describe experiments or numeric results.
- Fill in `references.yaml` following `templates/references_template.yaml`. Each entry states what was borrowed and where it is in the code; ids match the report's references one to one.

## Rules

- The encoder is loaded from `models_dir` and must not be modified; only tensors in `states.pt` may carry what was trained.
- Train only on the public training audio and transcripts; no other audio, text, language model or pretrained model.
- Do not read or write files outside the submission directory, the run directory and the paths the stage gives you; do not access the network; do not touch the timing pipe or the protocol bytes.
- Do not use extra processes or background threads to move work outside the timed phase, and do not leave processes running after a stage returns.
- Submission is one-shot. A submission that breaks the rules is invalid.
