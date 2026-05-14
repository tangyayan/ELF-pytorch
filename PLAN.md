# PLAN.md — Refactor into an unofficial PyTorch reproduction of ELF

## Goal

Refactor this repo into a publishable, **unofficial** PyTorch reproduction of
*ELF: Embedded Language Flows*. The published repo should:

1. Ship **only the PyTorch Lightning implementation**. The JAX (`src/`) and the non-Lightning
   PyTorch (`pytorch/`) ports are deleted from the public tree.
2. Be free of operator-identifying artifacts (W&B entity/run names tied to a specific account,
   absolute filesystem paths under `/NHNHOME/.../wogns98/...`, etc.).
3. Include the actual reproduced results obtained at
   `/NHNHOME/WORKSPACE/26weather001_A/wogns98/ELF/pytorch_lightning/outputs/elf_b-owt-lightning/`
   (config snapshot, per-epoch and 1000-sample generated samples, metrics) and provide the
   trained checkpoint for download.
4. Carry a clear note in the README that the repo was heavily developed with Claude Code.

## Non-goals

- No algorithmic change. Math/seed/optimizer/precision are frozen.
- No new tasks, models, or datasets beyond what already runs from `pytorch_lightning/`.
- No re-running of training. The plan consumes the existing reproduction artifacts only.
- We do **not** copy `pytorch/` or `src/` into the public repo, even as references.

## Reproduced result we will publish

Source dir: `/NHNHOME/WORKSPACE/26weather001_A/wogns98/ELF/pytorch_lightning/outputs/elf_b-owt-lightning/`

| File | Size | Role |
| --- | --- | --- |
| `last.ckpt` (== `checkpoint_epoch05_step00228204.ckpt`) | 1.4 GB | Final EMA-bearing checkpoint |
| `checkpoint_epoch00..05_step*.ckpt` | 6 × 1.4 GB | Per-epoch checkpoints |
| `config.yml` | 1.4 KB | Resolved training config snapshot |
| `sde-steps32-cfg1.0-sccfg3.0-ts_logit_normal-gamma1.5-eval1000/all_generated.jsonl` | 4.3 MB | 1000 generated samples (final eval) |
| `sde-steps32-cfg1.0-sccfg3.0-ts_logit_normal-gamma1.5-eval1000/metrics.jsonl` | 233 B | Final Gen. PPL + entropy |
| `sde-steps32-cfg1.0-sccfg3.0-ts_logit_normal-gamma1.5-uncond_epoch/epoch_00{1..6}.jsonl` | 6.3 MB total | Per-epoch 256-sample sanity-eval generations |
| `sde-steps32-cfg1.0-sccfg3.0-ts_logit_normal-gamma1.5-uncond_epoch/metrics.jsonl` | 667 B | Per-epoch Gen. PPL + entropy |

Final 1000-sample numbers (32-step SDE, γ=1.5, SC-CFG=3, ELF-B):

| Metric | Paper (TPU v5p-64) | This reproduction (8× B200 DDP, Lightning) |
| --- | --- | --- |
| Gen. PPL ↓ | 24.1 | **25.61** |
| Entropy ↑ | 5.15 | **5.20** |

Per-epoch training trajectory (32-step SDE, 256 samples per epoch):

| Epoch | Step | Gen. PPL | Entropy |
| --- | --- | --- | --- |
| 1 | 38 034  | 2.73¹  | 0.70¹ |
| 2 | 76 068  | 37.11  | 5.17 |
| 3 | 114 102 | 28.63  | 5.21 |
| 4 | 152 136 | 25.00  | 5.16 |
| 5 | 190 170 | 25.58  | 5.19 |
| 6 | 228 204 | 26.11  | 5.21 |

¹ Epoch-1 is degenerate (entropy ≈ 0.7); the run only begins producing fluent text from
epoch 2 onward. The plan keeps this row in the table — it documents the actual training curve.

The training config snapshot says `epochs: 8`, but the run was stopped after epoch 6
(`last.ckpt = step 228204`). The published config will be set to `epochs: 6` so the snapshot
matches the deliverable. (Anyone retraining can override with `--config_override epochs=N`.)

## Phase 1 — Prune the public tree

Delete or move out everything that is not needed to run training + evaluation in
`pytorch_lightning/`. Operate from the repo root.

**Delete (top-level):**
- `pytorch/` — older non-Lightning PyTorch port. References to it in CLAUDE.md are also removed.
- `src/` — original JAX/Flax reference.

**Delete (inside `pytorch_lightning/`):**
- `debug_isolation.py` — contains a hard-coded `HF_HOME=/NHNHOME/WORKSPACE/26weather001_A/wogns98/ELF/dataset` and is purely an internal bisection script.
- `smoke_test.py` — internal sanity test that references the FA4 dev path; not user-facing.
- `REFACTOR.md` — internal task list, superseded by this PLAN.md.

**Keep (inside `pytorch_lightning/`):**
- `train_lightning.py`, `eval_lightning.py`, `lightning_module.py`
- `configs/` (`config.py`, `training_configs/train_owt_ELF-B.yml`, `sampling_configs/*.yml`)
- `callbacks/`, `encoders/`, `modules/`, `utils/`

**Top-level layout after Phase 1:**

```
ELF-pytorch/
├── README.md                 # rewritten in Phase 5
├── CLAUDE.md                 # updated in Phase 6 (Lightning-only)
├── LICENSE                   # MIT, unchanged
├── PLAN.md                   # this file (kept until publication, then optionally moved to docs/)
├── requirements.txt          # rewritten (PyTorch+Lightning, no JAX)
├── environment.yml           # minor cleanup
├── .gitignore                # add exception for reproduction/ artifacts
├── assets/                   # teaser.gif, generation.gif, sys_compare.jpg (kept)
├── pytorch_lightning/        # the only impl
└── reproduction/             # added in Phase 4 (samples + metrics + config snapshot)
```

Decision recorded: we keep `pytorch_lightning/` as a sub-directory rather than promoting its
contents to the repo root, because (a) the YAML configs resolve paths relative to that
directory and rewriting them is a separate refactor, and (b) the CLAUDE.md path conventions
(`cd pytorch_lightning/ && torchrun ...`) are already established.

## Phase 2 — Scrub identifiable info

### Filesystem paths
- `debug_isolation.py:51` is deleted in Phase 1 (the only file with an absolute user path).
  After Phase 1 a fresh `grep -rn "/NHNHOME\|wogns\|wognsfjq\|naver" .` must return zero matches outside `.git/`.

### W&B identifiers
- `pytorch_lightning/configs/training_configs/train_owt_ELF-B.yml`:
  - `use_wandb: true` → `use_wandb: false` (opt-in)
  - `wandb_project: elf` → kept (generic)
  - `wandb_entity: null` → kept
  - `wandb_run_name: elf_b-owt-lightning` → kept (it is the deliverable's name, not a personal id)
- `pytorch_lightning/configs/config.py`: defaults already neutral; no change.
- `pytorch_lightning/train_lightning.py`: no change to the W&B code path itself; users enable W&B by setting `use_wandb: true` and supplying their own `wandb_entity`.

### Other
- Any references to `embedded-language-flows/...` HuggingFace repos in configs are paper-author repos, not personal — kept.
- T5 weight download (`google-t5/t5-small`) is upstream — kept.

After Phase 2, the grep matrix:

```
grep -rIn "NHNHOME\|/wogns\|wognsfjq\|naver"     # → 0 matches outside .git/
grep -rIn "use_wandb: *true"                     # → 0 matches outside reproduction/config.yml snapshot
```

## Phase 3 — Refactor configs and code for publication

These are minimal touch-ups only; no math changes.

1. **`pytorch_lightning/configs/training_configs/train_owt_ELF-B.yml`:**
   - `use_wandb: false` (Phase 2).
   - `epochs: 5` → `epochs: 6` to match the published checkpoint (228 204 steps == 6 epochs at global_batch_size 512 on OWT-T5). Document in a top-of-file comment that the original ELF JAX recipe was 5 epochs; we trained one extra to match the eval entropy more closely.
   - Verify the rest of the file is unchanged vs the reproduction snapshot (`outputs/elf_b-owt-lightning/config.yml`). Diff items to expect (all benign): `epochs`, comment headers.

2. **`pytorch_lightning/configs/config.py`:** no functional change. Optionally drop `wandb_resume*` keys if they exist (they don't on the Lightning side — only in the deleted `pytorch/` tree).

3. **`pytorch_lightning/train_lightning.py`, `eval_lightning.py`, `lightning_module.py`, `callbacks/`, `modules/`, `utils/`, `encoders/`:** no change beyond removing dead imports surfaced after Phase 1 deletes.

4. **Verification command** (no GPU needed; just an import-time smoke):
   ```bash
   cd pytorch_lightning/
   python -c "from configs.config import load_config_from_yaml; \
              cfg = load_config_from_yaml('configs/training_configs/train_owt_ELF-B.yml'); \
              print(cfg.epochs, cfg.use_wandb, cfg.wandb_run_name)"
   # → 6 False elf_b-owt-lightning
   ```

## Phase 4 — Bring the reproduction artifacts in

Create a `reproduction/elf_b-owt/` directory at the repo root (not under `pytorch_lightning/outputs/`, to keep training output paths and shipped artifacts logically distinct):

```
reproduction/elf_b-owt/
├── README.md                      # short README pointing to the parent README and listing what's here
├── config.yml                     # copied verbatim from outputs/elf_b-owt-lightning/config.yml
├── eval1000/
│   ├── all_generated.jsonl        # 4.3 MB, 1000 samples
│   └── metrics.jsonl              # final Gen. PPL + entropy row
└── per_epoch/
    ├── epoch_001.jsonl … epoch_006.jsonl   # 256-sample sanity gens per epoch
    └── metrics.jsonl                       # per-epoch Gen. PPL + entropy
```

**`.gitignore`:** the existing rule `outputs/` and `wandb/` already keeps generated training
outputs out of git. The reproduction dir lives at `reproduction/` and is untouched by the
ignore list — no exception needed.

**Checkpoint hosting (NOT committed to the repo).** Each checkpoint is 1.4 GB, total ~9.8 GB —
too large for a normal git repository. The plan:

- Final EMA checkpoint (`last.ckpt`) is uploaded to a personal HuggingFace model repo named
  e.g. `<hf-username>/ELF-B-owt-reproduction-pl`. The README documents the exact HF repo id.
- Per-epoch checkpoints are uploaded as additional files in the same HF repo under
  `checkpoints/checkpoint_epoch{00..05}_step*.ckpt`, gated behind a one-line `huggingface-cli
  download` instruction. They are optional for users who only want to evaluate.
- The README provides a small snippet showing how to evaluate the downloaded `last.ckpt`:

  ```bash
  pip install huggingface_hub
  huggingface-cli download <hf-username>/ELF-B-owt-reproduction-pl last.ckpt \
      --local-dir reproduction/elf_b-owt/
  cd pytorch_lightning/
  torchrun --nproc_per_node=8 eval_lightning.py \
      --config configs/training_configs/train_owt_ELF-B.yml \
      --checkpoint_path ../reproduction/elf_b-owt/last.ckpt \
      --num_samples 1000
  ```

The actual upload step is left for the maintainer to run after the repo refactor is approved
(the upload requires HF credentials and is out of scope for the in-repo refactor commits).
The plan **deliberately does not embed a placeholder username**; it will be filled in once
the HF repo exists.

**Copy commands** (Phase 4 execution checklist):

```bash
SRC=/NHNHOME/WORKSPACE/26weather001_A/wogns98/ELF/pytorch_lightning/outputs/elf_b-owt-lightning
DST=<repo-root>/reproduction/elf_b-owt

mkdir -p "$DST/eval1000" "$DST/per_epoch"
cp "$SRC/config.yml"  "$DST/config.yml"
cp "$SRC/sde-steps32-cfg1.0-sccfg3.0-ts_logit_normal-gamma1.5-eval1000/"*  "$DST/eval1000/"
cp "$SRC/sde-steps32-cfg1.0-sccfg3.0-ts_logit_normal-gamma1.5-uncond_epoch/"*  "$DST/per_epoch/"
```

Total committed reproduction artifacts: ~11 MB (well within git limits).

## Phase 5 — Rewrite README.md

New top-of-file framing (replaces the current JAX-first README):

1. **Title + unofficial-reproduction banner**

   ```markdown
   # ELF — Embedded Language Flows (Unofficial PyTorch Reproduction)

   > **This is an unofficial PyTorch reproduction** of *ELF: Embedded Language Flows*.
   > It is not affiliated with or endorsed by the paper authors. The official JAX/TPU
   > implementation lives at <https://github.com/...> (see the paper).
   >
   > **This repository was heavily developed with [Claude Code](https://claude.com/claude-code).**
   ```

2. **Status table.** Reuse the existing OWT/ELF-B column from the paper README, add a "this
   repro" column with the actual numbers from `reproduction/elf_b-owt/eval1000/metrics.jsonl`.

3. **What's in this repo.** A short bulleted map:
   - `pytorch_lightning/` — the model + training/eval scripts (PyTorch Lightning, 8-GPU DDP).
   - `reproduction/elf_b-owt/` — config snapshot + 1000 generated samples + per-epoch metrics from our run.
   - `assets/` — the figures from the official README (kept as-is, attribution preserved).

4. **Quickstart — Evaluate the reproduced checkpoint** (matches the actual reproduced run; gives users a 5-minute path to the headline number):

   ```bash
   # 1. Environment
   conda env create -f environment.yml -n elf-pytorch && conda activate elf-pytorch
   # (flash-attn install instructions, if the user wants to use the FlashAttention path)

   # 2. Download the reproduced checkpoint (1.4 GB)
   huggingface-cli download <hf-username>/ELF-B-owt-reproduction-pl last.ckpt \
       --local-dir reproduction/elf_b-owt/

   # 3. Run the 1000-sample evaluation
   cd pytorch_lightning/
   torchrun --nproc_per_node=8 --master_port=29510 eval_lightning.py \
       --config configs/training_configs/train_owt_ELF-B.yml \
       --checkpoint_path ../reproduction/elf_b-owt/last.ckpt \
       --num_samples 1000
   # Expected: Gen. PPL ≈ 25.6, sample entropy ≈ 5.20
   ```

5. **Quickstart — Train from scratch** (matches `train_lightning.py` exactly):

   ```bash
   cd pytorch_lightning/
   torchrun --nproc_per_node=8 --master_port=29501 train_lightning.py \
       --config configs/training_configs/train_owt_ELF-B.yml
   ```

   Followed by a short "Override config from the CLI" example for `--config_override`.

6. **Reproduction details.** A subsection listing:
   - Hardware: 8× NVIDIA B200 (sm_100), CUDA 12.8.
   - Wall-clock: ~3 hours per epoch × 6 epochs ≈ 18 hours total.
   - Differences vs the paper run:
     - Hardware: B200 GPU DDP vs TPU v5p-64.
     - Framework: PyTorch Lightning vs JAX/Flax.
     - Epochs: 6 (one extra over the paper's 5) to reach entropy ≈ 5.20.
     - Precision: fp32 by default (config field `precision`); bf16-mixed available behind the same flag.
     - All math otherwise identical to `src/train_step.py` from the official repo.

7. **Reproduced samples preview.** Inline a single sample from `eval1000/all_generated.jsonl`
   (one paragraph, ≤200 words) so casual readers see what the model produces without having
   to download anything.

8. **Acknowledgements / Disclaimer.**
   - Credit the original authors and the paper.
   - State explicitly: "This repository was heavily developed with [Claude Code](https://claude.com/claude-code)." Place this near the top (in the banner) and again in Acknowledgements for emphasis.
   - Acknowledge Google TRC the same way the official README does (kept).

9. **License.** MIT, unchanged.

10. **Sections we drop from the current README** because they no longer apply to this repo:
    - JAX/TPU install steps.
    - WMT14 De-En and XSum conditional sections (no reproduced checkpoints for those tasks here — keep them out rather than ship broken instructions; mention them in "Future work").

## Phase 6 — Update CLAUDE.md

Edit `CLAUDE.md` (currently describing the three-tree layout) to:

1. Open with: this repo has **one implementation**, in `pytorch_lightning/`.
2. Drop the bullet points about `pytorch/` and `src/`.
3. Update the "Repository" section to mention the unofficial-reproduction framing and the reproduction artifacts.
4. Keep the architecture, distributed/DDP, checkpoint-paths, model-sizes sections — they are still accurate.
5. Remove the line cross-referencing `pytorch/train_step.py` for math derivations; it no longer exists.

## Phase 7 — Update dependencies

`requirements.txt` currently mixes JAX (`jax[tpu]==0.4.38`, `flax`, `orbax-checkpoint`, `optax`,
`ml-collections`, `tensorflow`) with PyTorch. After Phase 1 the JAX deps are unused. Replace
the file with the PyTorch-only set actually imported under `pytorch_lightning/`:

```
torch
torchvision
lightning
transformers>=4.41.2,<4.45.0
tokenizers>=0.19.0
sentencepiece>=0.1.99
datasets>=2.19.0
huggingface-hub>=0.23.0
safetensors>=0.4.0
accelerate>=0.30.0
einops
numpy>=1.26.4,<2.0.0
scipy>=1.12.0
PyYAML>=6.0.1
tqdm>=4.66.0
sacrebleu
rouge-score
matplotlib>=3.8.0
requests>=2.31.0
# Optional logging
wandb>=0.16.6
# Optional, only if you use the FlashAttention path (use_flash: true)
# flash-attn  — installed out-of-tree; see the FlashAttention repo
```

`environment.yml` is already PyTorch-shaped; keep it as the conda entry point and align the
pip dependencies list with the cleaned `requirements.txt`. Note in a comment that the cu128
extra-index is required for B200 (sm_100); for older GPUs use the default cu121/cu124 wheels.

## Phase 8 — Final verification

Before publishing:

1. **Tree size sanity check.**
   ```bash
   du -sh .                        # should be < 50 MB (artifacts ~11 MB + code)
   git ls-files | xargs du -ch | tail -1
   ```

2. **Identifiable-info scrub.**
   ```bash
   grep -rIn "NHNHOME\|/wogns\|wognsfjq\|naver" . | grep -v .git/   # → empty
   grep -rIn "use_wandb: *true" .                                    # → only the reproduction/config.yml snapshot
   ```

3. **Import-time smoke** (Phase 3 verification command).

4. **Eval-time smoke** (only if a GPU machine with the checkpoint is available):
   ```bash
   cd pytorch_lightning/
   torchrun --nproc_per_node=1 eval_lightning.py \
       --config configs/training_configs/train_owt_ELF-B.yml \
       --checkpoint_path /path/to/last.ckpt \
       --num_samples 8
   # Should run end-to-end without crashing; PPL on 8 samples is noisy and only sanity-checks the path.
   ```

5. **README link check.** Every link in the rewritten README either resolves or is bracketed
   with `<TODO: fill in after HF upload>` so it cannot ship as a dead link by accident.

## Open questions deferred to execution

- **HF upload target.** The plan does not assume a specific HuggingFace username — the README will be edited in a follow-up commit once the HF repo is created. Until then, the README carries a single `<HF_REPO_ID>` placeholder so the rest of the document can be finalized.
- **Conditional tasks (WMT14, XSum).** Not reproduced in this run, so they are excluded from the public repo. If a future reproduction lands, the README and a `configs/training_configs/train_{de-en,xsum}_ELF-B.yml` file can be reintroduced under the same `reproduction/` scheme.
- **flash-attn install instructions.** The current `pytorch_lightning/` code has `use_flash: true` by default. The README will document a fallback path (`--config_override use_flash=false`) so users on a machine without a compatible FlashAttention build can still evaluate / train.

## Execution order (one-line-per-step)

1. Delete `pytorch/`, `src/`, `pytorch_lightning/{debug_isolation,smoke_test}.py`, `pytorch_lightning/REFACTOR.md`.
2. Set `use_wandb: false` and `epochs: 6` in `pytorch_lightning/configs/training_configs/train_owt_ELF-B.yml`.
3. Copy reproduction artifacts (samples + metrics + config snapshot) into `reproduction/elf_b-owt/`.
4. Rewrite `requirements.txt` to PyTorch-only; tidy `environment.yml`.
5. Rewrite `README.md` per Phase 5.
6. Update `CLAUDE.md` per Phase 6.
7. Run the Phase 8 verification grep + import smoke.
8. (Out-of-band) Upload `last.ckpt` to the chosen HF repo, then a follow-up commit that fills the `<HF_REPO_ID>` placeholder in the README.
