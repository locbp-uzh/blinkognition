#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with ChatGPT5
# __copyright_ = "Copyright 2025, UZH, Switzerland"

#Accelerator functions:

# --- Accelerator setup (drop-in replacement) ---
import os
import torch
from contextlib import nullcontext

def detect_accelerator():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)  # (major, minor)
        return {"type": "cuda", "device": dev, "name": name, "cap": cap}
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return {"type": "mps", "device": torch.device("mps"), "name": "Apple MPS", "cap": None}
    return {"type": "cpu", "device": torch.device("cpu"), "name": "CPU", "cap": None}

def _mps_autocast_supported():
    try:
        with torch.autocast(device_type="mps", dtype=torch.float16):
            pass
        return True
    except Exception:
        return False

class _NoopScaler:
    def scale(self, x): return x
    def step(self, opt): opt.step()
    def update(self): pass
    def __bool__(self): return False

def setup_precision_and_flags(accel, enable_amp_on_mps=False):
    """
    Return (amp_dtype, autocast_ctx, scaler) with new torch.amp API. Safe on MPS/CPU.

    Args:
        accel: Accelerator dict from detect_accelerator()
        enable_amp_on_mps: If True, enable fp16 autocast on MPS (default: False for stability)
                          MPS fp16 can cause NaN issues with deep models (10+ blocks).
                          Only enable if you have a shallow model or need the speed.
    """
    atype = accel["type"]

    # defaults
    amp_dtype = None
    autocast_ctx = nullcontext
    scaler = _NoopScaler()

    if atype == "cuda":
        major, _ = accel["cap"]
        # TF32 + bf16 on Ampere+; fp16 on pre-Ampere
        if major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            amp_dtype = torch.bfloat16
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            amp_dtype = torch.float16

        autocast_ctx = lambda: torch.autocast(device_type="cuda", dtype=amp_dtype)
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
        torch.backends.cudnn.benchmark = True

    elif atype == "mps":
        # MPS autocast (fp16) is disabled by default for numerical stability
        # Deep models (10+ blocks) often get NaN losses with fp16 on MPS
        # Enable with force_amp=true in config if you need speed and have a stable model
        if enable_amp_on_mps and _mps_autocast_supported():
            amp_dtype = torch.float16
            autocast_ctx = lambda: torch.autocast(device_type="mps", dtype=amp_dtype)
        else:
            amp_dtype = None
            autocast_ctx = nullcontext

    else:
        # CPU: fp32 only
        amp_dtype = None
        autocast_ctx = nullcontext
        scaler = _NoopScaler()

    return amp_dtype, autocast_ctx, scaler
    
def _slurm_cpus():
    v = os.environ.get("SLURM_CPUS_PER_TASK")
    try:
        return int(v) if v else None
    except Exception:
        return None

def dataloader_kwargs_for(accel):
    """
    Clamp workers to allocated CPUs, keep prefetch low, and avoid pinning on non-CUDA.
    This prevents RAM spikes under Slurm.
    """
    alloc = _slurm_cpus()
    host = os.cpu_count() or 1
    # target cores we can actually use
    usable = alloc if alloc is not None else min(host, 8)

    # conservative workers for tuning
    if usable <= 2:
        nw = 0
    elif usable <= 4:
        nw = 2
    else:
        nw = min(4, usable - 2)

    pin = (accel["type"] == "cuda")
    base = dict(
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=False,  # safer for RAM during many short trials
    )
    if nw > 0:
        base["prefetch_factor"] = 1
    return base

def maybe_compile(model, accel, enabled=True):
    """Compile only when safe. Default: CUDA SM80+; never on MPS."""
    if not enabled:
        return model
    try:
        if hasattr(torch, "compile"):
            if accel["type"] == "cuda" and accel.get("cap", (0, 0))[0] >= 8:
                # A100/H100 etc.
                return torch.compile(model, mode="max-autotune")
            # MPS or older CUDA -> skip
    except Exception as e:
        print(f"torch.compile skipped: {e}")
    return model

def batch_size_hint(default_bs, accel):
    # Keep your estimate_batch_size() for CUDA. Give a safe floor elsewhere.
    if accel["type"] == "cuda":
        return None  # signal to use your estimate_batch_size()
    if accel["type"] == "mps":
        return max(128, default_bs // 2)  # Apple GPUs like larger batches
    return 256  # CPU fallback
