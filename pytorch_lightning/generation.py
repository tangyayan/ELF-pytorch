import torch
import torch.distributed as dist
from tqdm import tqdm
from configs.config import SamplingConfig
from utils.generation_utils import (
    dlm_decode_batch, generate_samples, mask_after_eos,
)
from utils.sampling_utils import get_sampling_steps

def _all_gather_ids(local: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
        return local
    sizes = [torch.zeros_like(local) for _ in range(world_size)]
    dist.all_gather(sizes, local)
    return torch.cat(sizes, dim=0)

def _collate_fn(batch):
    """简单的 collate 函数"""
    keys = batch[0].keys()
    result = {}
    for key in keys:
        if key in ["input", "target", "input_text"]:
            result[key] = [item[key] for item in batch]
        else:
            result[key] = torch.stack([torch.tensor(item[key]) for item in batch])
    return result


def _all_gather_list(local_list: list, world_size: int) -> list:
    """Gather list across all ranks"""
    if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
        return local_list
    
    # Convert to tensor, gather, convert back
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_list)
    
    result = []
    for sublist in gathered:
        result.extend(sublist)
    return result

def test_generation_uncond(
    model, encoder_dim, tokenizer, device, cfg, args, rank, world_size,
):
    """Unconditional generation (无条件生成)"""
    print(f"[Rank {rank}] Starting unconditional generation...")
    
    sc = SamplingConfig(
        sampling_method="sde", num_sampling_steps=[args.num_sampling_steps],
        cfgs=[args.cfg_scale], self_cond_cfg_scales=[args.self_cond_cfg_scale],
        sde_gamma=args.sde_gamma, time_schedule="logit_normal",
    )
    pad_token_id = (tokenizer.eos_token_id if cfg.pad_token == "eos" else tokenizer.pad_token_id)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

    samples_per_rank = (args.num_samples + world_size - 1) // world_size
    per_rank_batch = cfg.global_batch_size // max(1, world_size)
    num_batches = (samples_per_rank + per_rank_batch - 1) // per_rank_batch

    generated_texts: list = []
    all_step_results: list = [] 
    
    for batch_idx in tqdm(range(num_batches), desc="generate_uncond", disable=(rank != 0)):
        cur = min(per_rank_batch, samples_per_rank - batch_idx * per_rank_batch)
        if cur <= 0:
            break
            
        # Stable per-rank seed
        seed = cfg.seed * 1000003 + 991 + batch_idx * 97 + rank
        gen = torch.Generator(device=device).manual_seed(seed)

        t_steps = get_sampling_steps(
            gen, n_steps=args.num_sampling_steps, device=device,
            time_schedule="logit_normal",
            P_mean=cfg.denoiser_p_mean, P_std=cfg.denoiser_p_std,
        )
        z = torch.randn(
            (cur, cfg.max_length, encoder_dim),
            generator=gen, device=device,
        ) * cfg.denoiser_noise_scale

        with torch.no_grad():
            latent = generate_samples(
                model, z, t_steps, cond_seq=None, cond_seq_mask=None,
                config=cfg, sampling_config=sc,
                cfg_scale=args.cfg_scale, self_cond_cfg_scale=args.self_cond_cfg_scale,
                generator=gen,
            )
            # for t_val, step_z, step_x_pred in intermediates:
            #     predicted_ids = dlm_decode_batch(
            #         model, step_x_pred, config=cfg,
            #         self_cond_cfg_scale=args.self_cond_cfg_scale,
            #         t_final_val=t_val,
            #     )
            #     predicted_ids = mask_after_eos(predicted_ids, eos_token_id=eos_token_id,
            #                                 pad_token_id=pad_token_id)
            #     gathered = _all_gather_ids(predicted_ids, world_size)
            #     texts = [tokenizer.decode(row.cpu().tolist(), skip_special_tokens=True)
            #             for row in gathered]
            #     all_step_results.append({
            #         "t": round(t_val, 4),
            #         "texts": texts,
            #     })
            # if rank == 0:
            #     print(f"t={t_val:.3f}: {texts[0]}")
            predicted_ids = dlm_decode_batch(
                model, latent, config=cfg,
                self_cond_cfg_scale=args.self_cond_cfg_scale,
                t_final_val=float(t_steps[-1].item()),
            )
        
        predicted_ids = mask_after_eos(predicted_ids, eos_token_id=eos_token_id,
                                       pad_token_id=pad_token_id)
        gathered = _all_gather_ids(predicted_ids, world_size)
        
        for row in gathered:
            text = tokenizer.decode(row.cpu().tolist(), skip_special_tokens=True)
            generated_texts.append(text)

    return generated_texts


def test_generation_cond(
    model, encoder, encoder_dim, tokenizer, device, cfg, args, rank, world_size,
    dataset, encoder_params=None,
):
    """Conditional generation (条件生成)"""
    print(f"[Rank {rank}] Starting conditional generation...")
    
    if dataset is None:
        raise ValueError("Dataset is required for conditional generation")
    
    from torch.utils.data import DataLoader, DistributedSampler
    
    sc = SamplingConfig(
        sampling_method="sde", num_sampling_steps=[args.num_sampling_steps],
        cfgs=[args.cfg_scale], self_cond_cfg_scales=[args.self_cond_cfg_scale],
        sde_gamma=args.sde_gamma, time_schedule="logit_normal",
    )
    pad_token_id = (tokenizer.eos_token_id if cfg.pad_token == "eos" else tokenizer.pad_token_id)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

    # 创建 dataloader
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank,
        shuffle=False, drop_last=False,
    )
    dataloader = DataLoader(
        dataset, batch_size=cfg.global_batch_size // world_size,
        sampler=sampler, num_workers=0, collate_fn=_collate_fn,
    )

    generated_texts: list = []
    context_texts: list = []
    target_texts: list = []
    
    samples_processed = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="generate_cond", disable=(rank != 0))):
        if samples_processed >= args.num_samples:
            break
        
        batch_size_current = batch["input_ids"].shape[0]
        
        # 编码条件文本
        input_ids = torch.tensor(batch["input_ids"], device=device)
        encoder_attention_mask = torch.tensor(batch["encoder_attention_mask"], device=device)
        cond_seq_mask = torch.tensor(batch["cond_seq_mask"], device=device)
        
        with torch.no_grad():
            # 编码条件
            cond_seq = encoder(input_ids, attention_mask=encoder_attention_mask).last_hidden_state
            # 归一化
            cond_seq = (cond_seq - cfg.latent_mean) / (cfg.latent_std + 1e-8)
        
        # 生成时间步
        seed = cfg.seed * 1000003 + 991 + batch_idx * 97 + rank
        gen = torch.Generator(device=device).manual_seed(seed)
        
        t_steps = get_sampling_steps(
            gen, n_steps=args.num_sampling_steps, device=device,
            time_schedule="logit_normal",
            P_mean=cfg.denoiser_p_mean, P_std=cfg.denoiser_p_std,
        )
        
        z = torch.randn(
            (batch_size_current, cfg.max_length, encoder_dim),
            generator=gen, device=device,
        ) * cfg.denoiser_noise_scale
        
        with torch.no_grad():
            latent = generate_samples(
                model, z, t_steps, cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
                config=cfg, sampling_config=sc,
                cfg_scale=args.cfg_scale, self_cond_cfg_scale=args.self_cond_cfg_scale,
                generator=gen,
            )
            predicted_ids = dlm_decode_batch(
                model, latent, config=cfg,
                self_cond_cfg_scale=args.self_cond_cfg_scale,
                t_final_val=float(t_steps[-1].item()),
            )
        
        predicted_ids = mask_after_eos(predicted_ids, eos_token_id=eos_token_id,
                                       pad_token_id=pad_token_id)
        
        # Gather across all ranks
        gathered_ids = _all_gather_ids(predicted_ids, world_size)
        gathered_context = _all_gather_list(batch.get("input", []), world_size)
        gathered_target = _all_gather_list(batch.get("target", []), world_size)
        
        for i, row in enumerate(gathered_ids):
            if samples_processed >= args.num_samples:
                break
            text = tokenizer.decode(row.cpu().tolist(), skip_special_tokens=True)
            generated_texts.append(text)
            context_texts.append(gathered_context[i] if i < len(gathered_context) else "")
            target_texts.append(gathered_target[i] if i < len(gathered_target) else "")
            samples_processed += 1
    
    return generated_texts, context_texts, target_texts