import json
import os
import torch
from utils.metrics_utils import Metrics as PPLMetrics

def compute_ppl_from_jsonl(
    jsonl_path: str,
    cfg,
    device: torch.device,
    rank: int = 0,
) -> dict:
    """从现有的 all_generated.jsonl 直接计算 PPL
    
    Args:
        jsonl_path: all_generated.jsonl 的路径
        cfg: 配置对象
        device: 计算设备
        rank: 进程 rank（只在 rank 0 打印）
    
    Returns:
        包含 PPL 和其他指标的字典
    """
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Computing PPL from: {jsonl_path}")
        print(f"{'='*70}\n")
    
    # 读取所有生成的文本
    generated_texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                # 支持多种字段名
                text = item.get("generated") or item.get("text") or ""
                if text:
                    generated_texts.append(text)
            except json.JSONDecodeError:
                continue
    
    if not generated_texts:
        if rank == 0:
            print(f"WARNING: No generated texts found in {jsonl_path}")
        return {
            "ppl": None,
            "mean_entropy": None,
            "num_samples": 0,
            "num_nonempty": 0,
            "note": "no valid texts found",
        }
    
    nonempty = [s for s in generated_texts if isinstance(s, str) and s.strip()]
    
    if rank == 0:
        print(f"Loaded {len(generated_texts)} samples, {len(nonempty)} non-empty")
    
    if not nonempty:
        return {
            "ppl": None,
            "mean_entropy": None,
            "num_samples": len(generated_texts),
            "num_nonempty": 0,
            "note": "all generations empty",
        }
    
    # 计算 PPL
    ppl_eval = PPLMetrics(
        gen_ppl_eval_model_name_or_path=cfg.eval_ppl_model,
        eval_ppl_batch_size=cfg.eval_ppl_batch_size,
        eval_context_size=cfg.eval_ppl_max_length,
        device=str(device),
    )
    
    if rank == 0:
        print(f"Computing PPL on {len(nonempty)} samples...")
    
    res = ppl_eval.record_generative_perplexity(
        text_samples=nonempty,
        max_length=cfg.eval_ppl_max_length,
        retokenize=True,
    )
    
    result = {
        "ppl": float(res["ppl"]),
        "mean_entropy": float(res["mean_entropy"]),
        "num_samples": len(generated_texts),
        "num_nonempty": len(nonempty),
    }
    
    return result


def eval_ppl_mode(
    args,
    cfg,
    device: torch.device,
    rank: int = 0,
) -> None:
    """PPL 评估模式：直接从 all_generated.jsonl 计算 PPL
    
    Args:
        args: 命令行参数
        cfg: 配置对象
        device: 计算设备
        rank: 进程 rank
    """
    if args.all_generated_path is None:
        raise ValueError("--all_generated_path is required for eval_ppl task")
    
    if not os.path.exists(args.all_generated_path):
        raise FileNotFoundError(f"File not found: {args.all_generated_path}")
    
    # 计算 PPL
    result = compute_ppl_from_jsonl(
        args.all_generated_path,
        cfg,
        device,
        rank=rank,
    )
    
    # 只在 rank 0 写入结果
    if rank == 0:
        # 确定输出目录
        if os.path.isdir(args.all_generated_path):
            out_dir = args.all_generated_path
        else:
            out_dir = os.path.dirname(args.all_generated_path)
        
        os.makedirs(out_dir, exist_ok=True)
        
        # 写入 metrics
        metrics_path = os.path.join(out_dir, "metrics_ppl.jsonl")
        
        row = {
            "ppl": result.get("ppl"),
            "mean_entropy": result.get("mean_entropy"),
            "num_samples": result.get("num_samples"),
            "num_nonempty": result.get("num_nonempty"),
            "all_generated_path": args.all_generated_path,
            "num_sampling_steps": args.num_sampling_steps,
            "sde_gamma": args.sde_gamma,
            "self_cond_cfg_scale": args.self_cond_cfg_scale,
            "cfg_scale": args.cfg_scale,
            "task": args.task,
        }
        
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
        
        print(f"\n{'='*70}")
        if result["ppl"] is not None:
            print(f"gen_ppl={result['ppl']:.4f}")
            print(f"mean_entropy={result['mean_entropy']:.4f}")
            print(f"num_samples={result['num_samples']}")
            print(f"num_nonempty={result['num_nonempty']}")
        else:
            print(f"PPL computation failed: {result.get('note', 'unknown error')}")
        print(f"Metrics saved to: {metrics_path}")
        print(f"{'='*70}\n")