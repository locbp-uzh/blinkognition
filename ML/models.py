#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with ChatGPT5 and Claude Code (Sonnet 4.6)
# based on code by Salome Püntener (EPFL/UZH), Andreas Biri (ETHZ) and Roman Briskine (UZH).
# __copyright_ = "Copyright 2026, UZH, Switzerland"

"""
models.py
Deep learning models for Blinkognition2:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

# --- Batch / Group / Instance normalization
def Norm1d(kind: str, num_channels: int, gn_groups: int = 8):
    kind = kind.lower()
    if kind == "bn":
        return nn.BatchNorm1d(num_channels)
    if kind == "gn":
        g = min(gn_groups, num_channels)
        while num_channels % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, num_channels)
    if kind == "in":
        return nn.InstanceNorm1d(num_channels, affine=True, track_running_stats=False)
    raise ValueError(f"Unknown norm kind: {kind}")

# --- Baseline model: pytorch clone of original blinkognition paper ---

class MCDropout(nn.Dropout):
    """Dropout that stays ON at eval for MC inference."""
    def forward(self, x):
        return F.dropout(x, p=self.p, training=True)

class LockedDropout(nn.Module):
    """Same dropout mask across time steps (B,T,C)."""
    def __init__(self, p: float = 0.0, mc_eval: bool = False):
        super().__init__()
        self.p = float(p)
        self.mc_eval = bool(mc_eval)

    def forward(self, x):
        if self.p <= 0.0:
            return x
        use_dropout = self.training or self.mc_eval
        if not use_dropout:
            return x
        if x.dim() == 3:  # (B,T,C)
            B, T, C = x.shape
            mask = x.new_ones(B, 1, C)
            mask = F.dropout(mask, p=self.p, training=True)  # broadcast over T
            return x * mask
        return F.dropout(x, p=self.p, training=True)

class OrigConvGRUClassifier(nn.Module):
    """
    Keras baseline in PyTorch:
    Conv1D(64,k=9,s=2)->BN->Dropout(0.1)
    Conv1D(64,k=3,s=2)->BN->Dropout(0.3)
    Conv1D(64,k=3,s=2)->BN
    GRU(128, return_sequences=True, input dropout=0.05)
    GRU(256)
    Dense(num_classes)
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        # conv stack
        conv_channels: int = 64,
        k1: int = 9, k2: int = 3, k3: int = 3,
        s1: int = 2, s2: int = 2, s3: int = 2,
        drop1: float = 0.10, drop2: float = 0.30,
        # GRUs
        gru_hidden_1: int = 128,
        gru_hidden_2: int = 256,
        gru_input_dropout: float = 0.05,   # matches Keras GRU dropout (input)
        # normalization
        norm: str = "bn", gn_groups: int = 8,  # baseline used BN; you can pass norm="gn"
        # MC settings
        mc_eval: bool = False,  # keep dropout ON at eval for MC predictions
    ):
        super().__init__()
        # Convs
        self.conv1 = nn.Conv1d(in_channels, conv_channels, kernel_size=k1, stride=s1, padding=k1//2)
        self.n1    = Norm1d(norm, conv_channels, gn_groups)
        self.do1   = MCDropout(drop1) if mc_eval else nn.Dropout(drop1)

        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=k2, stride=s2, padding=k2//2)
        self.n2    = Norm1d(norm, conv_channels, gn_groups)
        self.do2   = MCDropout(drop2) if mc_eval else nn.Dropout(drop2)

        self.conv3 = nn.Conv1d(conv_channels, conv_channels, kernel_size=k3, stride=s3, padding=k3//2)
        self.n3    = Norm1d(norm, conv_channels, gn_groups)

        # GRUs (baseline is unidirectional, batch_first=True)
        self.gru_in_drop1 = LockedDropout(gru_input_dropout, mc_eval=mc_eval)
        self.gru1 = nn.GRU(input_size=conv_channels, hidden_size=gru_hidden_1,
                           batch_first=True, bidirectional=False)

        self.gru_in_drop2 = LockedDropout(gru_input_dropout, mc_eval=mc_eval)
        self.gru2 = nn.GRU(input_size=gru_hidden_1, hidden_size=gru_hidden_2,
                           batch_first=True, bidirectional=False)

        # Classifier head (no softmax; use CrossEntropyLoss)
        self.fc = nn.Linear(gru_hidden_2, num_classes)
        self.relu = nn.ReLU()

    def get_embedding(self, x):
        """Extract embedding (representation before final classification layer)"""
        # Expect (B,C,T). If (B,T), add channel dim.
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B,1,T)

        # Conv stack
        x = self.relu(self.n1(self.conv1(x)))
        x = self.do1(x)
        x = self.relu(self.n2(self.conv2(x)))
        x = self.do2(x)
        x = self.relu(self.n3(self.conv3(x)))  # (B,C,T)

        # To (B,T,C) for GRU
        x = x.transpose(1, 2)

        # GRU 1 (return sequences)
        x = self.gru_in_drop1(x)
        x, _ = self.gru1(x)  # (B,T,H1)

        # GRU 2 (final state)
        x = self.gru_in_drop2(x)
        x, h2 = self.gru2(x)  # h2: (1,B,H2)
        feat = h2[-1]         # (B,H2)
        return feat

    def forward(self, x):
        feat = self.get_embedding(x)
        return self.fc(feat)

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1,
                 downsample=False, dropout=0.05, norm="gn", gn_groups=8):
        super().__init__()
        padding = kernel_size // 2
        self.downsample = downsample or (in_channels != out_channels)

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.n1 = Norm1d(norm, out_channels, gn_groups)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.n2 = Norm1d(norm, out_channels, gn_groups)

        self.dropout = nn.Dropout(p=dropout)

        if self.downsample:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                Norm1d(norm, out_channels, gn_groups),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.n1(self.conv1(x)))
        out = self.n2(self.conv2(out))
        out = self.dropout(out)
        out += identity
        return self.relu(out)

# --- ResNet1D ---

class ResNet1DClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, base_filters=64, num_blocks=3,
                 kernel_size=7, dropout=0.3, norm="gn", gn_groups=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=kernel_size, padding=kernel_size//2),
            Norm1d(norm, base_filters, gn_groups),
            nn.ReLU()
        )

        blocks = []
        filters = base_filters
        for _ in range(num_blocks):
            blocks.append(ResidualBlock1D(
                in_channels=filters,
                out_channels=filters * 2,
                kernel_size=kernel_size,
                stride=2,
                downsample=True,
                dropout=0.05,
                norm=norm, gn_groups=gn_groups
            ))
            filters *= 2

        self.resnet = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(filters, num_classes)

    def get_embedding(self, x):
        """Extract embedding (representation before final classification layer)"""
        x = self.stem(x)
        x = self.resnet(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        return x

    def forward(self, x):
        x = self.get_embedding(x)
        return self.classifier(x)

# --- TCN (Temporal Convolutional Network) ---

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation,
                 dropout, norm="gn", gn_groups=8):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        # conv1 -> norm -> relu -> dropout
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation)
        )
        self.n1 = Norm1d(norm, out_channels, gn_groups) if norm else nn.Identity()
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        # conv2 -> norm -> relu -> dropout
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=dilation)
        )
        self.n2 = Norm1d(norm, out_channels, gn_groups) if norm else nn.Identity()
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        # residual projection if channels change
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                Norm1d(norm, out_channels, gn_groups) if norm else nn.Identity(),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.n1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.n2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        # crop to match input length (due to padding/dilation)
        if out.size(-1) != x.size(-1):
            out = out[..., :x.size(-1)]

        return F.relu(out + self.downsample(x))


class TCN(nn.Module):
    def __init__(self, num_inputs, num_classes, num_channels,
                 kernel_size=3, dropout=0.05, norm="gn", gn_groups=8):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                              dilation=dilation, dropout=dropout,
                              norm=norm, gn_groups=gn_groups)
            )
        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def get_embedding(self, x):
        """Extract embedding (representation before final classification layer)"""
        # x: (B, C=num_inputs, T)
        x = self.network(x)
        x = self.global_pool(x).squeeze(-1)
        return x

    def forward(self, x):
        x = self.get_embedding(x)
        return self.fc(x)


# --- Model registry and factory ---

MODEL_REGISTRY = {
    "orig_conv_gru": OrigConvGRUClassifier,
    "resnet1d": ResNet1DClassifier,
    "tcn": TCN,
}

# Default parameters for each model
MODEL_DEFAULTS = {
    # Global defaults applied to all models
    "_global": {
        "norm": "gn",
        "gn_groups": 8,
        "dropout": 0.2,
    },

    # Model-specific defaults
    "orig_conv_gru": {
        "norm": "bn",
        "mc_eval": True,
        "gru_input_dropout": 0.05,
    },

    "tcn": {
        "num_channels": [64, 64, 128, 128],
        "kernel_size": 3,
    },

}

def build_model(name: str, **kwargs) -> nn.Module:
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")

    # Start with global defaults
    defaults = MODEL_DEFAULTS.get("_global", {}).copy()

    # Apply model-specific defaults
    model_defaults = MODEL_DEFAULTS.get(key, {})
    defaults.update(model_defaults)

    # User-provided kwargs override defaults
    defaults.update(kwargs)

    cls = MODEL_REGISTRY[key]
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in defaults.items() if k in allowed}
    return cls(**filtered)
