"""Sampling utilities (1:1 port of src/utils/sampling_utils.py).

All formulas match the JAX original:
- add_noise:        z = t*x0 + (1-t)*noise*denoiser_noise_scale  (cond preserved)
- sample_timesteps: sigmoid(N(P_mean, P_std)) or uniform
- sample_cfg_scale: log-uniform in [cfg_min, cfg_max]
- net_out_to_v_x:   v = (x - z) / max(1 - t, t_eps)
- ode_step:         z' = z + (t_next - t) * v
- sde_step:         z_back = alpha*z + (1-alpha)*eps (cond restored); ODE from t_back
"""

import math
from typing import Optional, Tuple

import torch


def add_noise(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, config,
              cond_seq_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    t_expanded = t.view(-1, 1, 1)
    z = t_expanded * x0 + (1 - t_expanded) * noise * config.denoiser_noise_scale
    if cond_seq_mask is not None:
        z = cond_seq_mask * x0 + (1 - cond_seq_mask) * z
    return z


def sample_timesteps(generator: Optional[torch.Generator], batch_size: int, device,
                     P_mean: float = -0.8, P_std: float = 0.8,
                     time_schedule: str = "logit_normal") -> torch.Tensor:
    if time_schedule == "logit_normal":
        z = torch.randn(batch_size, generator=generator, device=device) * P_std + P_mean
        return torch.sigmoid(z)
    if time_schedule == "uniform":
        return torch.rand(batch_size, generator=generator, device=device)
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


def get_sampling_steps(generator: Optional[torch.Generator], n_steps: int, device,
                       time_schedule: str = "logit_normal",
                       P_mean: float = -0.8, P_std: float = 0.8) -> torch.Tensor:
    if time_schedule == "uniform":
        return torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    if time_schedule == "logit_normal":
        if n_steps < 2:
            return torch.tensor([0.0, 1.0], device=device)
        steps = sample_timesteps(generator, n_steps - 1, device,
                                 P_mean=P_mean, P_std=P_std, time_schedule="logit_normal")
        steps, _ = torch.sort(steps)
        return torch.cat([torch.zeros(1, device=device), steps, torch.ones(1, device=device)])
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


def sample_cfg_scale(generator: Optional[torch.Generator], batch_size: int, device,
                     cfg_min: float = 0.0, cfg_max: float = 3.0) -> torch.Tensor:
    u = torch.rand(batch_size, generator=generator, device=device)
    a, b = 1.0 + cfg_min, 1.0 + cfg_max
    return a * torch.exp(u * math.log(b / a)) - 1.0


def restore_cond(z_updated: torch.Tensor, cond_seq: torch.Tensor,
                 cond_seq_mask: torch.Tensor) -> torch.Tensor:
    mask = cond_seq_mask
    target_ndim = max(z_updated.ndim, cond_seq.ndim)
    while mask.ndim < target_ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask > 0, cond_seq, z_updated)


def _restore_vx(v: torch.Tensor, x: torch.Tensor,
                cond_seq: Optional[torch.Tensor],
                cond_seq_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    if cond_seq is None:
        return v, x
    return restore_cond(v, torch.zeros_like(cond_seq), cond_seq_mask), restore_cond(x, cond_seq, cond_seq_mask)


def _zero_cond(z: torch.Tensor, cond_seq: Optional[torch.Tensor],
               cond_seq_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Build a zero tensor with cond positions restored from `cond_seq`."""
    if cond_seq is None:
        return torch.zeros_like(z)
    return restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)


def net_out_to_v_x(net_out, z: torch.Tensor, t: torch.Tensor,
                   t_eps: float = 5e-2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert x_pred network output to (v, x). Drops decoder_logits if present."""
    if isinstance(net_out, tuple):
        net_out = net_out[0]
    x = net_out
    v = (x - z) / torch.clamp(1.0 - t.view(-1, 1, 1), min=t_eps)
    return v, x


def _forward_sample_self_cond(
    model, z: torch.Tensor, t_batch: torch.Tensor, x_pred_prev: Optional[torch.Tensor],
    config, self_cond_cfg_scale: float,
    cond_seq: Optional[torch.Tensor], cond_seq_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    t_eps = config.t_eps

    if config.num_self_cond_cfg_tokens > 0:
        if x_pred_prev is None:
            x_pred_prev = _zero_cond(z, cond_seq, cond_seq_mask)
        sc_scale_batch = torch.full((z.size(0),), float(self_cond_cfg_scale),
                                    device=z.device, dtype=z.dtype)
        net_out_cond = model(torch.cat([z, x_pred_prev], dim=-1), t_batch,
                             self_cond_cfg_scale=sc_scale_batch)
        return _restore_vx(*net_out_to_v_x(net_out_cond, z, t_batch, t_eps),
                           cond_seq, cond_seq_mask)

    if config.self_cond_prob == 0:
        return _restore_vx(*net_out_to_v_x(model(z, t_batch), z, t_batch, t_eps),
                           cond_seq, cond_seq_mask)

    if self_cond_cfg_scale != 1 or x_pred_prev is None:
        net_out_uncond = model(torch.cat([z, _zero_cond(z, cond_seq, cond_seq_mask)], dim=-1), t_batch)
        v_uncond, x_uncond = _restore_vx(*net_out_to_v_x(net_out_uncond, z, t_batch, t_eps),
                                         cond_seq, cond_seq_mask)
        if self_cond_cfg_scale == 0.0 or x_pred_prev is None:
            return v_uncond, x_uncond

    net_out_cond = model(torch.cat([z, x_pred_prev], dim=-1), t_batch)
    v_cond, x_cond = _restore_vx(*net_out_to_v_x(net_out_cond, z, t_batch, t_eps),
                                 cond_seq, cond_seq_mask)
    if self_cond_cfg_scale == 1:
        return v_cond, x_cond

    v_out = v_uncond + self_cond_cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + self_cond_cfg_scale * (x_cond - x_uncond)
    return _restore_vx(v_out, x_out, cond_seq, cond_seq_mask)


def _forward_sample(
    model, z: torch.Tensor, t_batch: torch.Tensor, x_pred_prev: Optional[torch.Tensor],
    config, cfg_scale: float, self_cond_cfg_scale: float,
    cond_seq: Optional[torch.Tensor], cond_seq_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    v_cond, x_cond = _forward_sample_self_cond(
        model, z, t_batch, x_pred_prev, config,
        self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    if cfg_scale == 1.0:
        return v_cond, x_cond

    z_uncond = restore_cond(z, torch.zeros_like(z), cond_seq_mask)
    x_pred_prev_uncond = (None if x_pred_prev is None
                          else restore_cond(x_pred_prev, torch.zeros_like(x_pred_prev), cond_seq_mask))
    v_uncond, x_uncond = _forward_sample_self_cond(
        model, z_uncond, t_batch, x_pred_prev_uncond, config,
        self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=torch.zeros_like(cond_seq) if cond_seq is not None else None,
        cond_seq_mask=cond_seq_mask,
    )
    v_out = v_uncond + cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + cfg_scale * (x_cond - x_uncond)
    return _restore_vx(v_out, x_out, cond_seq, cond_seq_mask)


def ode_step(model, z, t, t_next, x_pred_prev, config,
             cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask):
    t_batch = torch.full((z.size(0),), float(t), device=z.device, dtype=z.dtype)
    v_pred, x_pred = _forward_sample(
        model, z, t_batch, x_pred_prev, config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    return z + (t_next - t) * v_pred, x_pred


def sde_step(model, z, t, t_next, x_pred_prev, config,
             cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask,
             gamma, generator):
    alpha = max(0.0, min(1.0, 1.0 - gamma * (t_next - t)))
    t_back = alpha * t
    eps = torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype) * config.denoiser_noise_scale
    z_back = alpha * z + (1.0 - alpha) * eps
    if cond_seq is not None:
        z_back = restore_cond(z_back, cond_seq, cond_seq_mask)
    t_batch = torch.full((z.size(0),), float(t_back), device=z.device, dtype=z.dtype)
    v_pred, x_pred = _forward_sample(
        model, z_back, t_batch, x_pred_prev, config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    return z_back + (t_next - t_back) * v_pred, x_pred
