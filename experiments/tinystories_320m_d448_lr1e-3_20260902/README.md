# TinyStories 320M-token training

This run trained the project Transformer LM for exactly 9,766 optimizer steps on an NVIDIA B200.
With a batch size of 128 and a context length of 256, the model consumed 320,012,288 training tokens.

## Result

| Metric | Value |
|---|---:|
| Parameters | 18,712,512 |
| Initial validation loss | 9.2685 |
| Best validation loss | 1.4112 at step 9,700 |
| Final validation loss | 1.4113 |
| Wall-clock time | 650.5 s (10.8 min) |
| End-to-end training throughput | 491,975 tokens/s |
| Peak allocated GPU memory | 13,073 MiB |

The final validation value is marginally above the best value by 0.00009, which is normal validation
noise rather than meaningful degradation.

## Configuration

- Model: 4 layers, `d_model=448`, 14 heads, `d_ff=1216`, context length 256, vocabulary size 10,000.
- Optimizer: AdamW, betas `(0.9, 0.95)`, weight decay 0.01, gradient clipping at 1.0.
- Schedule: 100-step linear warmup to `1e-3`, then cosine decay to `1e-4`.
- Validation: the same 20 deterministically sampled batches every 100 steps, plus the initial and final evaluations.
- Precision: float32 parameters with TF32 matrix multiplication enabled.

## Artifacts

- `final_model.pt`: compact final model checkpoint, 74,866,658 bytes (71.40 MiB).
  SHA-256: `5dacf4a4bcb74a87a1f9eecad6061cadec3e4e58889d8ff4306858636248f98c`.
- `loss_curves.svg`: directly viewable train/validation loss figure.
- `metrics.csv`: all 9,766 per-step training losses and 99 validation measurements, together with learning
  rate, gradient norm, throughput, elapsed time, tokens seen, and peak GPU memory.
- `tensorboard/`: TensorBoard event data for the same metrics.
- `train.log`: timestamped validation and checkpoint log.
- `sample.txt`: text sampled from the final model.
- `config.json` and `summary.json`: machine-readable configuration and result summary.

The optimizer-inclusive resumable checkpoint is retained on the Modal Volume at:

```text
/training_runs/tinystories_320m_d448_lr1e-3_20260902/checkpoint_final.pt
```

It is 214.2 MiB because AdamW keeps two float32 moment tensors for every parameter. Download it only
when resuming training:

```bash
modal volume get cs336-assignment1-data \
  /training_runs/tinystories_320m_d448_lr1e-3_20260902/checkpoint_final.pt \
  experiments/tinystories_320m_d448_lr1e-3_20260902/checkpoint_final.pt
```

## View TensorBoard

From the repository root:

```bash
uv run --with tensorboard tensorboard \
  --logdir experiments/tinystories_320m_d448_lr1e-3_20260902/tensorboard
```

Then open the local URL printed by TensorBoard, normally `http://localhost:6006`.

The reproducible Modal entrypoint is `modal_full_train.py`. It also performs full-corpus streaming
tokenization and verifies that the fast tokenizer produces exactly the same IDs as the assignment
tokenizer before training.
