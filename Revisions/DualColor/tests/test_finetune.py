#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with Claude Code (Opus 5.5)
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
train_pure.py's finetune option (TRAINING_NOTES.md point 16), on the CPU with the real model (orig_conv_gru):
1. init_from loads the source run's weights and batch-norm statistics exactly;
2. train_layers [fc]: only fc's parameters require gradients, and after optimizer steps in training mode
   (AdamW with weight decay, as train_pure) every other parameter and every batch-norm running statistic
   is unchanged while fc has moved;
3. without train_layers every weight trains (a convolution and a batch-norm statistic move);
4. an unknown layer and a model that differs from the source run's are refused.
Without the finetune key train_pure.py builds the optimizer from model.parameters() as before the option
(by construction: ft is None, so params = model.parameters() and config_full.json gains no key).

    python Revisions/DualColor/tests/test_finetune.py
"""

import json
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "ML"))
from train import build_model_from_config  # noqa: E402
from train_pure import CLASSES, finetune_init  # noqa: E402

CPU = torch.device("cpu")
CFG = {"model": {"name": "orig_conv_gru"}}


def model(seed: int):
    torch.manual_seed(seed)
    return build_model_from_config(model_name="orig_conv_gru", in_channels=1, num_classes=len(CLASSES),
                                   dropout=None, model_kwargs={}, device=CPU)


def source_run(tmp: Path) -> tuple[Path, dict]:
    """A 'trained' run folder: weights from another seed, batch-norm statistics moved off their defaults."""
    m = model(7)
    m.train()
    with torch.no_grad():
        for _ in range(3):
            m(torch.rand(16, 1, 400) * 3)
    run = tmp / "2026-01-01_00-00-00_src"
    run.mkdir()
    torch.save(m.state_dict(), run / "best_model.pth")
    (run / "config_full.json").write_text(json.dumps({"config": CFG}))
    return run, {k: v.clone() for k, v in m.state_dict().items()}


def steps(m, params, n: int = 3) -> None:
    opt = torch.optim.AdamW(params, lr=1e-2, weight_decay=4e-4)
    m.train()
    g = torch.Generator().manual_seed(0)
    for _ in range(n):
        x, y = torch.rand(32, 1, 400, generator=g), torch.randint(0, 2, (32,), generator=g)
        opt.zero_grad()
        torch.nn.functional.cross_entropy(m(x), y).backward()
        opt.step()


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        run, ref = source_run(Path(d))

        m = model(1)
        finetune_init(m, {"init_from": str(run)}, CFG, CPU)
        assert all(torch.equal(m.state_dict()[k], ref[k]) for k in ref), "weights not loaded exactly"
        print("1. init_from loads the weights and batch-norm statistics exactly")

        m = model(1)
        info = finetune_init(m, {"init_from": str(run), "train_layers": ["fc"]}, CFG, CPU)
        grad = {n for n, p in m.named_parameters() if p.requires_grad}
        assert grad == {"fc.weight", "fc.bias"}, grad
        steps(m, [p for p in m.parameters() if p.requires_grad])
        after = m.state_dict()
        moved = sorted(k for k in ref if k != "fc.weight" and k != "fc.bias" and not k.endswith("num_batches_tracked")
                       and not torch.equal(after[k], ref[k]))
        assert not moved, f"frozen state moved: {moved}"
        assert not torch.equal(after["fc.weight"], ref["fc.weight"]), "fc did not train"
        print(f"2. train_layers [fc]: {info['trainable_parameters']} of {info['parameters']} parameters train; "
              "every other weight and batch-norm statistic unchanged after 3 steps")

        m = model(1)
        finetune_init(m, {"init_from": str(run)}, CFG, CPU)
        assert all(p.requires_grad for p in m.parameters())
        steps(m, [p for p in m.parameters() if p.requires_grad])
        after = m.state_dict()
        bn = [k for k in ref if k.endswith("running_mean")]
        assert not torch.equal(after["conv1.weight"], ref["conv1.weight"]), "conv1 did not train"
        assert any(not torch.equal(after[k], ref[k]) for k in bn), "batch-norm statistics did not update"
        print("3. without train_layers every weight trains and the batch-norm statistics update")

        for ft, cfg, what in (({"init_from": str(run), "train_layers": ["fc", "head"]}, CFG, "unknown layer"),
                              ({"init_from": str(run)}, {"model": {"name": "tcn"}}, "different model")):
            try:
                finetune_init(model(1), ft, cfg, CPU)
            except SystemExit as e:
                print(f"4. {what} refused: {e}")
            else:
                raise AssertionError(f"{what} accepted")
    print("All finetune checks passed.")


if __name__ == "__main__":
    main()
