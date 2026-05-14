# Reproduction artifacts — ELF-B on OpenWebText

This directory holds the artifacts shipped with the published reproduction of ELF-B
trained from scratch on OpenWebText using the PyTorch Lightning implementation in
[`../../pytorch_lightning/`](../../pytorch_lightning/). See the top-level
[`README.md`](../../README.md) for the full reproduction report and download
instructions for the trained checkpoint.

## Contents

| Path | Role |
| --- | --- |
| `config.yml` | Resolved training-config snapshot from the actual run (the original `epochs: 8` was stopped after epoch 6; the published training YAML uses `epochs: 6` to match the deliverable). |
| `eval1000/all_generated.jsonl` | 1000 generated samples from the final EMA checkpoint at the headline schedule (32-step SDE, γ=1.5, SC-CFG=3). |
| `eval1000/metrics.jsonl` | Gen. PPL + sample entropy for the 1000-sample eval. |
| `per_epoch/epoch_001.jsonl` … `epoch_006.jsonl` | 256-sample sanity generations after each training epoch, same schedule. |
| `per_epoch/metrics.jsonl` | Per-epoch Gen. PPL + entropy trajectory. |

## Headline numbers (32-step SDE, γ=1.5, SC-CFG=3, ELF-B, OpenWebText)

| Metric | Paper (TPU v5p-64) | This reproduction (8× B200 DDP, Lightning) |
| --- | --- | --- |
| Gen. PPL ↓ | 24.1 | **25.61** |
| Entropy ↑ | 5.15 | **5.20** |

## Reproducing these numbers

```bash
# 1. Download the EMA checkpoint (1.4 GB) from https://huggingface.co/Ugness/elf-torch
huggingface-cli download Ugness/elf-torch last.ckpt --local-dir .

# 2. Run the same 1000-sample eval that produced eval1000/metrics.jsonl.
cd ../../pytorch_lightning/
torchrun --nproc_per_node=8 --master_port=29510 eval_lightning.py \
    --config configs/training_configs/train_owt_ELF-B.yml \
    --checkpoint_path ../reproduction/elf_b-owt/last.ckpt \
    --num_samples 1000
```

The per-epoch checkpoints (`checkpoint_epoch00_*.ckpt` … `checkpoint_epoch05_*.ckpt`)
are also on the HF repo under `checkpoints/` if you want to reproduce the per-epoch
trajectory in `per_epoch/metrics.jsonl`.
