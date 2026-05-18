#!/usr/bin/env python
"""Canonical 1000-sample eval against a Lightning checkpoint.

Usage:
  cd pytorch_lightning/
  torchrun --nproc_per_node=8 --master_port=29510 eval_lightning.py \
      --config configs/training_configs/train_owt_ELF-B.yml \
      --checkpoint_path ../reproduction/elf_b-owt/last.ckpt \
      --num_samples 1000
torchrun --nproc_per_node=8 --master_port=29510 eval_lightning.py \
    --config configs/training_configs/train_owt_ELF-B.yml \
    --checkpoint_path ../reproduction/elf_b-owt/last.ckpt \
    --task eval_ppl \
    --all_generated_path outputs/elf_b-owt-lightning/sde-steps32-cfg1.0-sccfg3.0-ts_logit_normal-gamma1.5-eval50/all_generated.jsonl
    --num_samples 1000

Mirrors `PerEpochGenEvalCallback` (32-step SDE γ=1.5 SC-CFG=3 by default) but
takes `num_samples` as a CLI arg and runs standalone (no Trainer loop). Writes
the generated samples + metrics into outputs/<output_dir>/eval_<run_name>/.
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path: sys.path.insert(0, REPO_ROOT)

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer

from configs.config import SamplingConfig, apply_config_overrides, load_config_from_yaml
from encoders import build_encoder
from lightning_module import EMA
from modules.model import ELF_models
from utils.generation_utils import (
    build_run_name, dlm_decode_batch, generate_samples, mask_after_eos,
)
from utils.metrics_utils import Metrics as PPLMetrics
from utils.sampling_utils import get_sampling_steps
from eval_ppl import eval_ppl_mode
from generation import test_generation_uncond, test_generation_cond


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--num_sampling_steps", type=int, default=32)
    p.add_argument("--sde_gamma", type=float, default=1.5)
    p.add_argument("--self_cond_cfg_scale", type=float, default=3.0)
    p.add_argument("--cfg_scale", type=float, default=1.0)
    p.add_argument("--config_override", action="append", default=[])
    p.add_argument("--task", type=str, default="uncond", choices=["uncond", "cond", "eval_ppl"],
                   help="Task type: 'uncond' for unconditional or 'cond' for conditional generation")
    p.add_argument("--eval_dataset_path", type=str, default=None,
                   help="Path to evaluation dataset (required for conditional generation)")
    p.add_argument("--all_generated_path", type=str, default=None,
                   help="Optional path to save all generated samples in a single JSONL file")
    return p.parse_args()



def main():
    args = parse_args()

    # Distributed init
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    cfg = load_config_from_yaml(args.config)
    if args.config_override:
        cfg = apply_config_overrides(cfg, args.config_override)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if args.task == "eval_ppl":
        eval_ppl_mode(args, cfg, device, rank=rank)
        if world_size > 1 and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name or cfg.encoder_model_name)
    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = tokenizer.vocab_size

    encoder = build_encoder(cfg.encoder_model_name, dtype=torch.float32).to(device)
    encoder.eval()
    encoder_dim = encoder.d_model

    model = ELF_models[cfg.model](
        text_encoder_dim=encoder_dim, max_length=cfg.max_length,
        attn_drop=cfg.attn_dropout, proj_drop=cfg.proj_dropout,
        num_time_tokens=cfg.num_time_tokens,
        num_self_cond_cfg_tokens=cfg.num_self_cond_cfg_tokens,
        vocab_size=vocab_size,
        num_model_mode_tokens=cfg.num_model_mode_tokens,
        bottleneck_dim=cfg.bottleneck_dim,
        self_cond_input=(cfg.self_cond_prob > 0),
        use_flash=cfg.use_flash,
    ).to(device)

    # Load Lightning checkpoint
    if rank == 0:
        print(f"Loading checkpoint: {args.checkpoint_path}")
    ckpt = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    model_state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if rank == 0:
        if missing:    print(f"WARNING missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected: print(f"WARNING unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    # EMA
    ema = EMA(model, decay=cfg.ema_decay1)
    if "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"], device=device)
        ema.swap_in(model)
        if rank == 0:
            print(f"EMA weights swapped in (decay={ema.decay}).")
    elif rank == 0:
        print("WARNING: no EMA in checkpoint; using raw weights.")

    model.eval()

    # ===== 选择生成任务 =====
    if args.task == "uncond":
        if rank == 0:
            print("\n" + "="*70)
            print("UNCONDITIONAL GENERATION")
            print("="*70 + "\n")
        
        generated_texts = test_generation_uncond(
            model, encoder_dim, tokenizer, device, cfg, args, rank, world_size,
        )
    
    elif args.task == "cond":
        if args.eval_dataset_path is None:
            raise ValueError("--eval_dataset_path is required for conditional generation")
        
        if rank == 0:
            print("\n" + "="*70)
            print("CONDITIONAL GENERATION")
            print("="*70 + "\n")
        
        # Load dataset
        from datasets import load_dataset
        if args.eval_dataset_path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=args.eval_dataset_path)["train"]
        else:
            dataset = load_dataset(args.eval_dataset_path)["validation"]
        
        generated_texts, context_texts, target_texts = test_generation_cond(
            model, encoder, encoder_dim, tokenizer, device, cfg, args, rank, world_size,
            dataset,
        )
    
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    # if rank == 0:
    #     run_name = build_run_name(
    #         "sde", args.num_sampling_steps, args.cfg_scale, args.self_cond_cfg_scale,
    #         "logit_normal", args.sde_gamma, suffix=f"eval{args.num_samples}_uncond",
    #     )
    #     out_dir = os.path.join(cfg.output_dir, run_name)
    #     os.makedirs(out_dir, exist_ok=True)
    #     out_path = os.path.join(out_dir, f"intermediates.jsonl")
    #     with open(out_path, "w", encoding="utf-8") as f:
    #         for entry in all_step_results:
    #             f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ===== Rank 0 处理结果 =====
    if rank == 0:
        if args.task == "uncond":
            generated_texts = generated_texts[:args.num_samples]
            
            run_name = build_run_name(
                "sde", args.num_sampling_steps, args.cfg_scale, args.self_cond_cfg_scale,
                "logit_normal", args.sde_gamma, suffix=f"eval{args.num_samples}_uncond",
            )
        else:
            generated_texts = generated_texts[:args.num_samples]
            context_texts = context_texts[:args.num_samples]
            target_texts = target_texts[:args.num_samples]
            
            run_name = build_run_name(
                "sde", args.num_sampling_steps, args.cfg_scale, args.self_cond_cfg_scale,
                "logit_normal", args.sde_gamma, suffix=f"eval{args.num_samples}_cond",
            )
        
        out_dir = os.path.join(cfg.output_dir, run_name)
        os.makedirs(out_dir, exist_ok=True)
        
        # 写入 JSONL
        with open(os.path.join(out_dir, "all_generated.jsonl"), "w", encoding="utf-8") as f:
            for i, t in enumerate(generated_texts):
                if args.task == "uncond":
                    f.write(json.dumps({"id": i, "generated": t}, ensure_ascii=False) + "\n")
                else:
                    f.write(json.dumps({
                        "id": i, "context": context_texts[i], "target": target_texts[i],
                        "generated": t
                    }, ensure_ascii=False) + "\n")
        
        nonempty = [s for s in generated_texts if isinstance(s, str) and s.strip()]
        print(f"Generated {len(generated_texts)} samples; {len(nonempty)} non-empty.")

        # PPL 评估
        metrics_path = os.path.join(out_dir, "metrics.jsonl")
        if not nonempty:
            with open(metrics_path, "w") as f:
                f.write(json.dumps({"ppl": None, "mean_entropy": None,
                                    "checkpoint": args.checkpoint_path,
                                    "num_samples": args.num_samples,
                                    "task": args.task,
                                    "note": "all generations empty"}) + "\n")
        else:
            ppl_eval = PPLMetrics(
                gen_ppl_eval_model_name_or_path=cfg.eval_ppl_model,
                eval_ppl_batch_size=cfg.eval_ppl_batch_size,
                eval_context_size=cfg.eval_ppl_max_length,
                device=str(device),
            )
            res = ppl_eval.record_generative_perplexity(
                text_samples=nonempty, max_length=cfg.eval_ppl_max_length, retokenize=True,
            )
            row = {
                "ppl": float(res["ppl"]), "mean_entropy": float(res["mean_entropy"]),
                "checkpoint": args.checkpoint_path,
                "num_samples": args.num_samples,
                "num_nonempty": len(nonempty),
                "num_sampling_steps": args.num_sampling_steps,
                "sde_gamma": args.sde_gamma,
                "self_cond_cfg_scale": args.self_cond_cfg_scale,
                "cfg_scale": args.cfg_scale,
                "task": args.task,
            }
            with open(metrics_path, "w") as f:
                f.write(json.dumps(row) + "\n")
            print(f"==> gen_ppl={row['ppl']:.4f}  sample_entropy={row['mean_entropy']:.4f}")
            print(f"==> wrote {metrics_path}")

    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()