#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
ML/utils.train_model warmup_epochs (TRAINING_NOTES.md point 10), on the CPU with a small model:
1. warmup_epochs = 0 reproduces the train_model of commit 83efc7c (before the option) exactly:
   losses, AUCs and the saved checkpoint, so the paper's training is unchanged;
2. with a frozen model (lr 0, nothing ever improves), training stops at epoch 1 + patience
   without warm-up and at warmup + 1 + patience with it, the checkpoint is taken at epoch
   warmup + 1, and no checkpoint file exists before then;
3. warmup_epochs >= max_epochs is refused.

    python Revisions/DualColor/tests/test_warmup.py
"""

import contextlib
import importlib.util
import io
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "ML"))
from utils import train_model  # noqa: E402

BEFORE = "83efc7c"   # the last commit without warmup_epochs


def old_train_model():
    src = subprocess.run(["git", "-C", str(REPO), "show", f"{BEFORE}:ML/utils.py"], check=True,
                         capture_output=True, text=True).stdout
    path = Path(tempfile.mkdtemp()) / "utils_before.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("utils_before", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.train_model


def data(seed: int):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 2, (256,), generator=g)
    X = torch.randn(256, 1, 64, generator=g) + 0.4 * y[:, None, None]
    return (DataLoader(TensorDataset(X[:192], y[:192]), batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed)),
            DataLoader(TensorDataset(X[192:], y[192:]), batch_size=32))


def run(fn, lr: float, epochs: int, patience: int, out: Path, **kw):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv1d(1, 4, 5), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(4, 2))
    tr, va = data(0)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = fn(model=model, train_loader=tr, val_loader=va, criterion=nn.CrossEntropyLoss(),
                 optimizer=torch.optim.AdamW(model.parameters(), lr=lr), device="cpu", num_classes=2,
                 max_epochs=epochs, patience_limit=patience, model_save_path=str(out / "best.pth"),
                 checkpoint_path=str(out / "ckpt.pth"), threshold_metric="argmax", auc_min_delta=0.005,
                 loss_mode="relative", loss_tolerance=1.1, auc_tolerance=0.003, loss_min_delta=0.005, **kw)
    return res, buf.getvalue()


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    # 1. no warm-up = the code before the option, exactly
    (a := tmp / "old").mkdir(); (b := tmp / "new").mkdir()
    old, _ = run(old_train_model(), 1e-2, 12, 3, a)
    new, _ = run(train_model, 1e-2, 12, 3, b, warmup_epochs=0)
    for x, y, name in zip(old[:3], new[:3], ("train losses", "val losses", "val AUCs")):
        assert x == y, f"{name} differ: {x} vs {y}"
    so, sn = torch.load(a / "best.pth"), torch.load(b / "best.pth")
    assert all(torch.equal(so[k], sn[k]) for k in so), "saved checkpoints differ"
    print(f"1. warmup 0 = {BEFORE}: {len(new[0])} epochs, identical losses, AUCs and checkpoint")

    # 2. frozen model: stop epochs and checkpoint epoch
    for warm, patience in ((0, 3), (4, 3), (7, 2)):
        (d := tmp / f"w{warm}").mkdir()
        res, log = run(train_model, 0.0, 40, patience, d, warmup_epochs=warm)
        m = re.search(r"Early stopping at epoch (\d+) \(best AUC [0-9.]+ at epoch (\d+)\)", log)
        assert m, log
        stop, best = int(m.group(1)), int(m.group(2))
        assert (stop, best, len(res[0])) == (warm + 1 + patience, warm + 1, warm + 1 + patience), (warm, stop, best)
        print(f"2. warmup {warm}, patience {patience}: stop at epoch {stop}, checkpoint at epoch {best}")
    (d := tmp / "short").mkdir()
    run(train_model, 0.0, 3, 1, d, warmup_epochs=2)   # max_epochs 3: the only checkpoint is epoch 3
    assert (d / "best.pth").exists(), "no checkpoint at epoch warmup + 1"
    (d := tmp / "none").mkdir()
    try:
        run(train_model, 0.0, 3, 1, d, warmup_epochs=3)
    except ValueError as e:
        print(f"3. refused: {e}")
    else:
        raise AssertionError("warmup_epochs >= max_epochs was not refused")
    print("all passed")


if __name__ == "__main__":
    main()
