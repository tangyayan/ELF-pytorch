"""ELF as a `pl.LightningModule`.

Training-step math is byte-for-byte identical to the baseline `pytorch/train_step.py`:
two-objective (Bernoulli(decoder_prob)) branch select, denoiser L2 with optional
self-cond CFG guidance, decoder-branch CE on token ids, EMA updated only on real
optimizer steps. DDP / grad accumulation / AMP / checkpointing are delegated to
`pl.Trainer` — see `train_lightning.py`.
"""

import math
from typing import Any, Dict, List

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT

from configs.config import Config
from encoders import build_encoder
from modules.model import ELF_models
from utils.data_utils import get_pad_token_id, load_dataset, make_dataloader
from utils.encoder_utils import encode_text
from utils.muon import Muon, build_muon_param_groups
from utils.sampling_utils import (
    add_noise, net_out_to_v_x, restore_cond, sample_cfg_scale, sample_timesteps,
)


# EMA — updated only on real optimizer steps (matches JAX `is_optimizer_step`).
class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def _aligned(self, name: str, p: torch.Tensor) -> torch.Tensor:
        """Return the shadow tensor for `name` on `p`'s device, migrating lazily.
        Shadow is built before Lightning moves the model; first call after the
        move copies each entry across."""
        buf = self.shadow[name]
        if buf.device != p.device:
            buf = buf.to(p.device)
            self.shadow[name] = buf
        return buf

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self._aligned(n, p).mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap_in(self, model: torch.nn.Module) -> dict:
        backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                backup[n] = p.detach().clone()
                p.data.copy_(self._aligned(n, p))
        return backup

    @torch.no_grad()
    def restore(self, model: torch.nn.Module, backup: dict):
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])

    def state_dict(self) -> dict:
        return {"decay": self.decay,
                "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: dict, device):
        self.decay = state.get("decay", self.decay)
        self.shadow = {k: v.to(device) for k, v in state["shadow"].items()}


# -----------------------------------------------------------------------------
# LightningModule
# -----------------------------------------------------------------------------
class ELFLitModule(L.LightningModule):
    def __init__(self, config: Config, vocab_size: int):
        super().__init__()
        self.cfg = config
        self.vocab_size = vocab_size
        self.automatic_optimization = False  # we manage Muon + AdamW + EMA manually
        self._train_generator = None
        self._ema = None
        self._loss_running: Dict[str, List[float]] = {"loss": [], "l2": [], "ce": []}

        # Frozen pretrained encoder (Task 1: behind an interface).
        self.encoder = build_encoder(config.encoder_model_name, dtype=torch.float32)
        encoder_dim = self.encoder.d_model

        # ELF transformer.
        model_fn = ELF_models[config.model]
        self.model = model_fn(
            text_encoder_dim=encoder_dim, max_length=config.max_length,
            attn_drop=config.attn_dropout, proj_drop=config.proj_dropout,
            num_time_tokens=config.num_time_tokens,
            num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
            vocab_size=vocab_size,
            num_model_mode_tokens=config.num_model_mode_tokens,
            bottleneck_dim=config.bottleneck_dim,
            self_cond_input=(config.self_cond_prob > 0),
            use_flash=config.use_flash,
        )

    # --- lifecycle hooks --------------------------------------------------
    def setup(self, stage=None):
        if self._ema is None:
            self._ema = EMA(self.model, decay=self.cfg.ema_decay1)
        if self._train_generator is None:
            seed = self.cfg.seed * 100003 + self.global_rank
            self._train_generator = torch.Generator(device=self.device).manual_seed(seed)
        # Self-maintained step counter: manual-optim mode's `self.global_step`
        # semantics with multiple optimizers (Muon + AdamW) are opaque enough
        # that gating on it can silently miss. We increment this every is_opt_step
        # and use it for the LR schedule and log throttling.
        if not hasattr(self, "_my_opt_step"):
            self._my_opt_step = 0

    # --- optimizer & schedule --------------------------------------------
    def configure_optimizers(self):
        cfg = self.cfg
        if cfg.optimizer == "muon":
            muon_groups, adamw_groups = build_muon_param_groups(
                self.model,
                adamw_betas=(cfg.adam_b1, cfg.adam_b2), weight_decay=cfg.weight_decay,
            )
            muon = Muon(muon_groups, lr=1e-12)
            adamw = torch.optim.AdamW(adamw_groups, lr=1e-12,
                                       betas=(cfg.adam_b1, cfg.adam_b2),
                                       weight_decay=cfg.weight_decay)
            return [muon, adamw]
        elif cfg.optimizer == "adamw":
            return [torch.optim.AdamW(self.model.parameters(), lr=1e-12,
                                       betas=(cfg.adam_b1, cfg.adam_b2),
                                       weight_decay=cfg.weight_decay)]
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    # --- training step ----------------------------------------------------
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        cfg = self.cfg
        gen = self._train_generator
        device = batch["input_ids"].device

        # Use our own opt-step counter: with manual optimization + multiple
        # optimizers (Muon + AdamW), Lightning's `global_step` semantics are
        # opaque enough that gating on it can silently miss.
        is_opt_step = (batch_idx + 1) % cfg.grad_accum_steps == 0
        lr = self._lr_at_step(self._my_opt_step)
        for opt in self.optimizers():
            for g in opt.param_groups:
                g["lr"] = lr

        encoder_attention_mask = batch["encoder_attention_mask"]
        if cfg.label_drop_prob > 0:
            B = encoder_attention_mask.size(0)
            drop = (torch.rand(B, generator=gen, device=device)
                    < cfg.label_drop_prob).float().view(B, 1, 1)
            cm = batch["cond_seq_mask"]
            block_mask = (1 - cm).unsqueeze(2) * cm.unsqueeze(1)
            encoder_attention_mask = encoder_attention_mask * (1 - drop * block_mask)
            label_drop_mask = drop.squeeze(-1).squeeze(-1).bool()
        else:
            label_drop_mask = torch.zeros(encoder_attention_mask.size(0),
                                          dtype=torch.bool, device=device)

        with torch.no_grad():
            x0 = encode_text(batch["input_ids"], encoder_attention_mask,
                             encoder=self.encoder, latent_mean=cfg.latent_mean,
                             latent_std=cfg.latent_std)
        B, S, _ = x0.shape

        t = sample_timesteps(gen, B, device=device,
                             P_mean=cfg.denoiser_p_mean, P_std=cfg.denoiser_p_std,
                             time_schedule=cfg.time_schedule)
        noise = torch.randn(x0.shape, generator=gen, device=device, dtype=x0.dtype)

        cond_seq_mask = batch["cond_seq_mask"].unsqueeze(-1)
        loss_mask = (batch["attention_mask"] if cfg.pad_token == "pad"
                     else torch.ones_like(batch["attention_mask"]))
        loss_mask = loss_mask * (1 - batch["cond_seq_mask"])

        denoiser_z = add_noise(x0, noise, t, cfg, cond_seq_mask=cond_seq_mask)
        if cfg.label_drop_prob > 0:
            drop_zero = ((label_drop_mask.view(-1, 1, 1).float() * (cond_seq_mask > 0).float()) > 0)
            zero = torch.zeros_like(denoiser_z)
            denoiser_z = torch.where(drop_zero, zero, denoiser_z)
            x0 = torch.where(drop_zero, zero, x0)

        decoder_z_vals = (torch.randn(B * S, generator=gen, device=device)
                          * cfg.decoder_p_std + cfg.decoder_p_mean)
        decoder_lambda_t = torch.sigmoid(decoder_z_vals).view(B, S, 1).to(x0.dtype)
        decoder_noise = (torch.randn(x0.shape, generator=gen, device=device, dtype=x0.dtype)
                         * cfg.decoder_noise_scale)
        decoder_z = decoder_lambda_t * x0 + (1.0 - decoder_lambda_t) * decoder_noise

        t_eps = cfg.t_eps
        v_target = (x0 - denoiser_z) / torch.clamp(1.0 - t.view(-1, 1, 1), min=t_eps)

        use_self_cond_mask = None
        if cfg.self_cond_prob > 0:
            use_self_cond_mask = ((torch.rand(B, generator=gen, device=device) < cfg.self_cond_prob)
                                  .view(-1, 1, 1).to(x0.dtype))
        sc_cfg_scale = None
        if cfg.num_self_cond_cfg_tokens > 0:
            sc_cfg_scale = sample_cfg_scale(gen, B, device=device,
                                            cfg_min=cfg.self_cond_cfg_min,
                                            cfg_max=cfg.self_cond_cfg_max).to(x0.dtype)

        decoder_step_active = bool(
            (torch.rand((), generator=gen, device=device) < cfg.decoder_prob).item()
        )

        if decoder_step_active:
            decoder_input = (torch.cat([decoder_z, torch.zeros_like(decoder_z)], dim=-1)
                             if cfg.self_cond_prob > 0 else decoder_z)
            _, decoder_logits = self.model(decoder_input, torch.ones_like(t),
                                           self_cond_cfg_scale=sc_cfg_scale,
                                           decoder_step_active=True)
            log_probs = F.log_softmax(decoder_logits.float(), dim=-1)
            ce = -log_probs.gather(-1, batch["input_ids"].unsqueeze(-1)).squeeze(-1)
            ce_loss = (ce * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
            loss = ce_loss
            l2_loss = torch.zeros((), device=device)
        else:
            if cfg.self_cond_prob > 0:
                with torch.no_grad():
                    z_uncond = restore_cond(torch.zeros_like(denoiser_z), x0, cond_seq_mask)
                    net_out_init = self.model(torch.cat([denoiser_z, z_uncond], dim=-1),
                                              t, self_cond_cfg_scale=sc_cfg_scale)
                    _, x_pred_init = net_out_to_v_x(net_out_init, denoiser_z, t, t_eps)
                    x_pred_init = restore_cond(x_pred_init, x0, cond_seq_mask)
                    x_pred_cond = x_pred_init * use_self_cond_mask.to(denoiser_z.dtype)
                    x_pred_cond = restore_cond(x_pred_cond, x0, cond_seq_mask)
                denoiser_input = torch.cat([denoiser_z, x_pred_cond], dim=-1)
            else:
                denoiser_input = denoiser_z

            net_out = self.model(denoiser_input, t, self_cond_cfg_scale=sc_cfg_scale,
                                 decoder_step_active=False)
            v_pred, _ = net_out_to_v_x(net_out, denoiser_z, t, t_eps)

            if cfg.num_self_cond_cfg_tokens > 0:
                with torch.no_grad():
                    z_uncond = restore_cond(torch.zeros_like(denoiser_z), x0, cond_seq_mask)
                    net_out_uncond = self.model(torch.cat([denoiser_z, z_uncond], dim=-1),
                                                t, self_cond_cfg_scale=sc_cfg_scale)
                    v_uncond, x_uncond = net_out_to_v_x(net_out_uncond, denoiser_z, t, t_eps)
                    x_uncond = restore_cond(x_uncond, x0, cond_seq_mask)
                    net_out_cond = self.model(torch.cat([denoiser_z, x_uncond], dim=-1),
                                              t, self_cond_cfg_scale=sc_cfg_scale)
                    v_cond, _ = net_out_to_v_x(net_out_cond, denoiser_z, t, t_eps)
                    sc_w = sc_cfg_scale.view(B, 1, 1).to(v_target.dtype)
                    sc_guidance = (1 - 1 / sc_w) * (v_cond - v_uncond)
                    if use_self_cond_mask is not None:
                        sc_guidance = torch.where(use_self_cond_mask > 0, sc_guidance,
                                                  torch.zeros_like(sc_guidance))
                    v_final_target = (v_target + sc_guidance).detach()
            else:
                v_final_target = v_target

            per_token_loss = ((v_pred - v_final_target) ** 2).mean(dim=-1)
            safe = torch.where(loss_mask > 0, per_token_loss, torch.zeros_like(per_token_loss))
            l2_loss = (safe * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
            loss = l2_loss
            ce_loss = torch.zeros((), device=device)

        self.manual_backward(loss / cfg.grad_accum_steps)
        self._loss_running["loss"].append(loss.detach())
        self._loss_running["l2"].append(l2_loss.detach())
        self._loss_running["ce"].append(ce_loss.detach())

        if is_opt_step:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            for opt in self.optimizers():
                opt.step()
                opt.zero_grad(set_to_none=True)
            self._ema.update(self.model)
            self._my_opt_step += 1
            if self._my_opt_step % cfg.log_freq == 0:
                self._log_running(lr)

        return loss.detach()

    # --- helpers ---------------------------------------------------------
    def _lr_at_step(self, opt_step: int) -> float:
        cfg = self.cfg
        base_lr = cfg.lr if cfg.lr else cfg.blr * cfg.global_batch_size * cfg.grad_accum_steps / 256
        if cfg.warmup_steps is not None and cfg.warmup_steps > 0:
            num_warmup = cfg.warmup_steps
        elif cfg.warmup_epochs is not None:
            num_warmup = int(cfg.warmup_epochs * getattr(self, "_steps_per_epoch", 1))
        else:
            num_warmup = 0
        if opt_step < num_warmup:
            return base_lr * (opt_step / max(1, num_warmup))
        if cfg.lr_schedule == "cosine":
            total = getattr(self, "_num_optimizer_steps", 1)
            t = (opt_step - num_warmup) / max(1, total - num_warmup)
            cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
            return cfg.min_lr + (base_lr - cfg.min_lr) * cos
        return base_lr

    def _log_running(self, lr: float):
        if not self._loss_running["loss"]: return
        dp = max(1e-8, self.cfg.decoder_prob)
        np_ = max(1e-8, 1.0 - self.cfg.decoder_prob)
        loss = torch.stack(self._loss_running["loss"]).float().mean()
        l2 = torch.stack(self._loss_running["l2"]).float().mean() / np_
        ce = torch.stack(self._loss_running["ce"]).float().mean() / dp
        # Lightning auto-syncs across DDP ranks when sync_dist=True. In manual
        # optimization mode, `self.log` defaults to on_epoch=True/on_step=False,
        # which queues metrics until epoch end — explicitly setting on_step=True
        # makes them flow to wandb every `log_every_n_steps` iterations.
        self.log("train/loss", loss, sync_dist=True, prog_bar=True,
                  on_step=True, on_epoch=False)
        self.log("train/l2_loss", l2, sync_dist=True,
                  on_step=True, on_epoch=False)
        self.log("train/ce_loss", ce, sync_dist=True,
                  on_step=True, on_epoch=False)
        self.log("train/lr", lr, sync_dist=False,
                  on_step=True, on_epoch=False)
        for k in self._loss_running: self._loss_running[k].clear()

    # --- checkpoint plumbing --------------------------------------------
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["ema"] = self._ema.state_dict()
        # Persist our own opt-step counter so the LR schedule (warmup + cosine)
        # is continuous across resumes. Without this, _my_opt_step would reset
        # to 0 in setup() and re-trigger warmup from base_lr=0.
        checkpoint["_my_opt_step"] = getattr(self, "_my_opt_step", 0)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self._ema is None:
            self._ema = EMA(self.model, decay=self.cfg.ema_decay1)
        if "ema" in checkpoint:
            self._ema.load_state_dict(checkpoint["ema"], device=self.device)
        if "_my_opt_step" in checkpoint:
            self._my_opt_step = int(checkpoint["_my_opt_step"])
        elif "global_step" in checkpoint:
            # Fallback for checkpoints saved before _my_opt_step was persisted.
            # In manual-optim mode, Lightning's `global_step` increments once
            # per `LightningOptimizer.step()`, so it counts opt steps × #
            # optimizers (Muon + AdamW → 2; AdamW alone → 1).
            num_opts = 2 if self.cfg.optimizer == "muon" else 1
            self._my_opt_step = int(checkpoint["global_step"]) // num_opts


# -----------------------------------------------------------------------------
# DataModule
# -----------------------------------------------------------------------------
class ELFDataModule(L.LightningDataModule):
    def __init__(self, config: Config, tokenizer):
        super().__init__()
        self.cfg = config
        self.tokenizer = tokenizer
        self._train_dataset = None
        self._eval_dataset = None

    def setup(self, stage=None):
        if self._train_dataset is None:
            self._train_dataset, self._eval_dataset = load_dataset(self.cfg)

    def train_dataloader(self):
        # Lightning attaches the DistributedSampler when strategy=ddp.
        return make_dataloader(
            self._train_dataset,
            batch_size=self.cfg.global_batch_size // self.trainer.world_size,
            shuffle=True,
            max_seq_length=self.cfg.max_length,
            pad_token_id=get_pad_token_id(self.tokenizer, self.cfg.pad_token),
            max_input_seq_length=self.cfg.max_input_length,
            num_workers=self.cfg.num_workers,
            prefetch_factor=self.cfg.prefetch_factor,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers,
            drop_last=True,
        )
