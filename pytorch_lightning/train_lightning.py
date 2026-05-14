#!/usr/bin/env python
"""Lightning entry point for the ELF PyTorch port.

Usage (8 B200 DDP):
  cd pytorch_lightning/
  torchrun --nproc_per_node=8 --master_port=29501 train_lightning.py \
      --config configs/training_configs/train_owt_ELF-B.yml
"""

import argparse
import logging
import math
import os
import sys

import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path: sys.path.insert(0, REPO_ROOT)

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from transformers import AutoTokenizer

from callbacks import PerEpochGenEvalCallback
from configs.config import SamplingConfig, apply_config_overrides, load_config_from_yaml, load_sampling_configs
from lightning_module import ELFDataModule, ELFLitModule


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--config_override", action="append", default=[])
    return p.parse_args()


def _resolve_precision(cfg_precision: str) -> str:
    return {
        "fp32": "32",
        "32": "32",
        "bf16": "bf16-mixed",
        "bf16-mixed": "bf16-mixed",
        "fp16": "16-mixed",
        "16-mixed": "16-mixed",
    }.get(cfg_precision, "32")


def main():
    args = parse_args()
    logging.basicConfig(
        format="%(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)], level=logging.INFO, force=True,
    )

    cfg = load_config_from_yaml(args.config)
    if args.config_override:
        cfg = apply_config_overrides(cfg, args.config_override)
    if cfg.sampling_configs_path:
        cfg.sampling_configs = load_sampling_configs(cfg.sampling_configs_path)

    # B200 TF32 + cudnn benchmark (same as baseline).
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    L.seed_everything(cfg.seed, workers=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name or cfg.encoder_model_name)
    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = tokenizer.vocab_size

    # --- modules --------------------------------------------------------
    model = ELFLitModule(cfg, vocab_size=vocab_size)
    datamodule = ELFDataModule(cfg, tokenizer=tokenizer)

    # --- logger ----------------------------------------------------------
    logger = False
    if cfg.use_wandb:
        logger = WandbLogger(
            project=cfg.wandb_project, entity=cfg.wandb_entity,
            name=cfg.wandb_run_name, save_dir="/tmp",
            tags=cfg.wandb_tag.split(",") if cfg.wandb_tag else None,
        )

    # --- callbacks -------------------------------------------------------
    callbacks = []
    callbacks.append(ModelCheckpoint(
        dirpath=cfg.output_dir, filename="checkpoint_epoch{epoch:02d}_step{step:08d}",
        every_n_epochs=int(cfg.save_freq) if cfg.save_freq >= 1 else 1,
        save_top_k=-1,  # keep all
        save_last=True,
        auto_insert_metric_name=False,
    ))
    if cfg.online_eval and cfg.eval_freq >= 1:
        callbacks.append(PerEpochGenEvalCallback(
            tokenizer=tokenizer, output_dir=cfg.output_dir,
            num_samples=cfg.epoch_eval_num_samples,
            num_sampling_steps=cfg.epoch_eval_num_sampling_steps,
            sde_gamma=cfg.epoch_eval_sde_gamma,
            self_cond_cfg_scale=cfg.epoch_eval_self_cond_cfg_scale,
            eval_ppl_model=cfg.eval_ppl_model,
            eval_ppl_batch_size=cfg.eval_ppl_batch_size,
            eval_ppl_max_length=cfg.eval_ppl_max_length,
        ))

    # --- trainer ---------------------------------------------------------
    # broadcast_buffers=False: RoPE cos/sin buffers are deterministic per-rank;
    # DDP's pre-forward `copy_` bumps their autograd version, breaking backward
    # of the real denoiser forward (we issue up to 4 model() calls per step).
    strategy = DDPStrategy(find_unused_parameters=True, broadcast_buffers=False)

    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator="gpu", devices=-1, strategy=strategy,
        precision=_resolve_precision(cfg.precision),
        accumulate_grad_batches=cfg.grad_accum_steps,
        logger=logger, callbacks=callbacks,
        log_every_n_steps=cfg.log_freq,
        default_root_dir=cfg.output_dir,
        use_distributed_sampler=True,
    )

    ckpt_path = cfg.resume
    if ckpt_path is None:
        last = os.path.join(cfg.output_dir, "last.ckpt")
        if os.path.exists(last):
            ckpt_path = last

    # Snapshot resolved config alongside checkpoints so eval can replay.
    if trainer.is_global_zero:
        os.makedirs(cfg.output_dir, exist_ok=True)
        snap = {k: ([vars(sc) for sc in v]
                    if isinstance(v, list) and v and isinstance(v[0], SamplingConfig) else v)
                for k, v in vars(cfg).items()}
        with open(os.path.join(cfg.output_dir, "config.yml"), "w") as f:
            yaml.dump(snap, f, default_flow_style=False, sort_keys=False)

    # Stash a few hints on the module so its LR schedule can read them.
    steps_per_epoch_est = max(1, math.ceil(9_737_184 / cfg.global_batch_size))
    model._steps_per_epoch = steps_per_epoch_est
    model._num_optimizer_steps = steps_per_epoch_est * cfg.epochs // max(1, cfg.grad_accum_steps)

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
