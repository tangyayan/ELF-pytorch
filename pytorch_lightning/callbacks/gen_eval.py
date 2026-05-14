"""Per-epoch gen-PPL + unigram-entropy callback.

At `on_train_epoch_end`: swap EMA weights in → generate `epoch_eval_num_samples`
with 32-step SDE (γ=1.5, SC-CFG=3) → decode via DLM head → retokenize with
gpt2-large → log `eval/gen_ppl_32step` and `eval/sample_entropy_32step` →
restore weights and resume training.
"""

import json
import os

import lightning as L
import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import Callback
from tqdm import tqdm

from configs.config import Config, SamplingConfig
from utils.generation_utils import (
    build_run_name, dlm_decode_batch, generate_samples, mask_after_eos,
)
from utils.metrics_utils import Metrics as PPLMetrics
from utils.sampling_utils import get_sampling_steps


class PerEpochGenEvalCallback(Callback):
    """Run 32-step generation + gen_ppl + sample_entropy at every epoch end."""

    def __init__(self, *, tokenizer, output_dir: str,
                  num_samples: int, num_sampling_steps: int,
                  sde_gamma: float, self_cond_cfg_scale: float,
                  eval_ppl_model: str, eval_ppl_batch_size: int,
                  eval_ppl_max_length: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.num_sampling_steps = num_sampling_steps
        self.sde_gamma = sde_gamma
        self.self_cond_cfg_scale = self_cond_cfg_scale
        self.eval_ppl_model = eval_ppl_model
        self.eval_ppl_batch_size = eval_ppl_batch_size
        self.eval_ppl_max_length = eval_ppl_max_length
        self._ppl_metrics = None  # lazy-initialized on rank 0

    @torch.no_grad()
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        cfg: Config = pl_module.cfg
        device = pl_module.device
        rank = trainer.global_rank
        world_size = trainer.world_size

        # Swap EMA into the model for evaluation.
        backup = pl_module._ema.swap_in(pl_module.model)
        try:
            pl_module.model.eval()

            pad_token_id = (self.tokenizer.eos_token_id if cfg.pad_token == "eos"
                            else self.tokenizer.pad_token_id)
            eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 1

            text_encoder_dim = pl_module.model.text_encoder_dim
            samples_per_rank_total = (self.num_samples + world_size - 1) // world_size

            # Mini sampling-config object so we reuse `generate_samples`.
            sc = SamplingConfig(
                sampling_method="sde",
                num_sampling_steps=[self.num_sampling_steps],
                cfgs=[1],
                self_cond_cfg_scales=[self.self_cond_cfg_scale],
                sde_gamma=self.sde_gamma,
                time_schedule="logit_normal",
            )

            generated_texts = []
            per_rank_batch = cfg.global_batch_size // max(1, world_size)
            num_batches = (samples_per_rank_total + per_rank_batch - 1) // per_rank_batch

            for batch_idx in tqdm(range(num_batches), desc=f"[ep{trainer.current_epoch+1}] gen32",
                                   disable=(rank != 0)):
                cur = min(per_rank_batch, samples_per_rank_total - batch_idx * per_rank_batch)
                if cur <= 0: break
                seed = cfg.seed * 1000003 + trainer.current_epoch * 991 + batch_idx * 97 + rank
                gen = torch.Generator(device=device).manual_seed(seed)

                t_steps = get_sampling_steps(
                    gen, n_steps=self.num_sampling_steps, device=device,
                    time_schedule="logit_normal",
                    P_mean=cfg.denoiser_p_mean, P_std=cfg.denoiser_p_std,
                )
                z = torch.randn(
                    (cur, cfg.max_length, text_encoder_dim),
                    generator=gen, device=device,
                ) * cfg.denoiser_noise_scale

                latent = generate_samples(
                    pl_module.model, z, t_steps,
                    cond_seq=None, cond_seq_mask=None,
                    config=cfg, sampling_config=sc,
                    cfg_scale=1.0, self_cond_cfg_scale=self.self_cond_cfg_scale,
                    generator=gen,
                )
                predicted_ids = dlm_decode_batch(
                    pl_module.model, latent, config=cfg,
                    self_cond_cfg_scale=self.self_cond_cfg_scale,
                    t_final_val=float(t_steps[-1].item()),
                )
                predicted_ids = mask_after_eos(predicted_ids, eos_token_id=eos_token_id,
                                                pad_token_id=pad_token_id)

                # Gather across ranks (only rank 0 keeps the consolidated list).
                gathered = self._all_gather_ids(predicted_ids, world_size)
                for row in gathered:
                    text = self.tokenizer.decode(row.cpu().tolist(), skip_special_tokens=True)
                    generated_texts.append(text)

            # Rank-0 writes JSONL + runs PPL.
            if rank == 0:
                run_name = build_run_name(
                    "sde", self.num_sampling_steps, 1.0, self.self_cond_cfg_scale,
                    "logit_normal", self.sde_gamma, suffix="uncond_epoch",
                )
                out_dir = os.path.join(self.output_dir, run_name)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"epoch_{trainer.current_epoch+1:03d}.jsonl")
                with open(out_path, "w", encoding="utf-8") as f:
                    for i, t in enumerate(generated_texts):
                        f.write(json.dumps({"id": i, "generated": t},
                                            ensure_ascii=False) + "\n")

                nonempty = [s for s in generated_texts if isinstance(s, str) and s.strip()]
                if not nonempty:
                    pl_module.log("eval/gen_ppl_32step", float("nan"), rank_zero_only=True)
                    pl_module.log("eval/sample_entropy_32step", float("nan"), rank_zero_only=True)
                else:
                    if self._ppl_metrics is None:
                        self._ppl_metrics = PPLMetrics(
                            gen_ppl_eval_model_name_or_path=self.eval_ppl_model,
                            eval_ppl_batch_size=self.eval_ppl_batch_size,
                            eval_context_size=self.eval_ppl_max_length,
                            device=str(device),
                        )
                    res = self._ppl_metrics.record_generative_perplexity(
                        text_samples=nonempty,
                        max_length=self.eval_ppl_max_length,
                        retokenize=True,
                    )
                    pl_module.log("eval/gen_ppl_32step", float(res["ppl"]), rank_zero_only=True)
                    pl_module.log("eval/sample_entropy_32step",
                                  float(res["mean_entropy"]), rank_zero_only=True)
                    # Persist alongside the JSONL for offline analysis.
                    with open(os.path.join(out_dir, "metrics.jsonl"), "a",
                              encoding="utf-8") as f:
                        f.write(json.dumps({
                            "epoch": trainer.current_epoch + 1,
                            "step": trainer.global_step,
                            "gen_ppl_32step": float(res["ppl"]),
                            "sample_entropy_32step": float(res["mean_entropy"]),
                        }, ensure_ascii=False) + "\n")

            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        finally:
            pl_module._ema.restore(pl_module.model, backup)
            pl_module.model.train()

    @staticmethod
    def _all_gather_ids(local: torch.Tensor, world_size: int) -> torch.Tensor:
        if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
            return local
        sizes = [torch.zeros_like(local) for _ in range(world_size)]
        dist.all_gather(sizes, local)
        return torch.cat(sizes, dim=0)
