"""Training utility functions (parameter logging, etc.)."""
import torch.nn as nn


def log_detailed_trainable(model: nn.Module):
    print("\n" + "="*60, flush=True)
    print(f"{'PARAM NAME':<50} {'SHAPE':<20} {'SIZE':>10}", flush=True)
    print("-" * 82, flush=True)
    
    total_trainable = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            count = param.numel()
            total_trainable += count
            shape_str = str(tuple(param.shape))
            print(f"{name:<50} {shape_str:<20} {count:>10,}", flush=True)
            
    print("-" * 82, flush=True)
    print(f"TOTAL TRAINABLE PARAMS: {total_trainable:,} ({total_trainable/1e6:.3f}M)", flush=True)
    print("="*60 + "\n", flush=True)