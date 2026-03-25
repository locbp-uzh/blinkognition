#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# __author__ = Pablo Rivera Fuentes pablo.riverafuentes@uzh.ch with ChatGPT5
# __copyright_ = "Copyright 2025, UZH, Switzerland"

"""
models.py
Deep learning models for Blinkognition2:
"""
import math
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

# --- Hybrid CNN GRU (inspired by original blinkgonition paper but augmented) ---
class ConvGRUClassifier(nn.Module):
    # Current numbers come from optuna optimization of 20250728
    def __init__(
        self, num_classes, in_channels=1,
        conv_dropout_1=0.37844750329926613, conv_dropout_2=0.11903520814735996,
        gru_dropout_1=0.11735026933359419, gru_dropout_2=0.01796955443412299,
        bidirectional_gru1=True, bidirectional_gru2=True,
        conv_channels_1=128, conv_channels_2=128, conv_channels_3=128,
        kernel_size_1=7, kernel_size_2=7, kernel_size_3=3,
        stride_1=2, stride_2=1, stride_3=1,
        gru_hidden_1=518, gru_hidden_2=16,
        norm="gn", gn_groups=8
    ):
        super().__init__()

        # --- Convolutions ---
        self.conv1 = nn.Conv1d(in_channels, conv_channels_1, kernel_size=kernel_size_1, stride=stride_1)
        self.n1 = Norm1d(norm, conv_channels_1, gn_groups)
        self.drop1 = nn.Dropout(conv_dropout_1)

        self.conv2 = nn.Conv1d(conv_channels_1, conv_channels_2, kernel_size=kernel_size_2, stride=stride_2)
        self.n2 = Norm1d(norm, conv_channels_2, gn_groups)
        self.drop2 = nn.Dropout(conv_dropout_2)

        self.conv3 = nn.Conv1d(conv_channels_2, conv_channels_3, kernel_size=kernel_size_3, stride=stride_3)
        self.n3 = Norm1d(norm, conv_channels_3, gn_groups)

        # --- GRUs ---
        self.gru1 = nn.GRU(
            input_size=conv_channels_3,
            hidden_size=gru_hidden_1,
            batch_first=True,
            bidirectional=bidirectional_gru1
        )
        self.drop_gru1 = nn.Dropout(gru_dropout_1)

        gru1_out = gru_hidden_1 * (2 if bidirectional_gru1 else 1)
        self.gru2 = nn.GRU(
            input_size=gru1_out,
            hidden_size=gru_hidden_2,
            batch_first=True,
            bidirectional=bidirectional_gru2
        )
        self.drop_gru2 = nn.Dropout(gru_dropout_2)

        gru2_out = gru_hidden_2 * (2 if bidirectional_gru2 else 1)
        self.attn_fc = nn.Linear(gru2_out, 1)
        self.fc = nn.Linear(gru2_out, num_classes)

    def get_embedding(self, x):
        """Extract embedding (representation before final classification layer)"""
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.drop1(F.relu(self.n1(self.conv1(x))))
        x = self.drop2(F.relu(self.n2(self.conv2(x))))
        x = F.relu(self.n3(self.conv3(x)))
        x = x.permute(0, 2, 1)

        x, _ = self.gru1(x)
        x = self.drop_gru1(x)
        x, _ = self.gru2(x)
        x = self.drop_gru2(x)

        attn_scores = self.attn_fc(x)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = (x * attn_weights).sum(dim=1)
        return context

    def forward(self, x):
        context = self.get_embedding(x)
        return self.fc(context)

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

class TemporalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, T) → (B, T, C)
        x = x.transpose(1, 2)
        attn_output, _ = self.attn(x, x, x)
        out = self.norm(x + attn_output)
        return out.transpose(1, 2)  # (B, T, C) → (B, C, T)

class SelfAttentionPooling1D(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: (B, C, T)
        x = x.permute(0, 2, 1)  # (B, T, C)
        weights = self.attention(x)  # (B, T, 1)
        weights = torch.softmax(weights, dim=1)
        pooled = (x * weights).sum(dim=1)  # (B, C)
        return pooled

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

# --- Self-attention variant ---

class ResNet1DClassifierSelfAttn(nn.Module):
    def __init__(self, in_channels, num_classes, base_filters=64, num_blocks=3,
                 kernel_size=7, dropout=0.3, attn_heads=4, norm="gn", gn_groups=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=kernel_size, padding=kernel_size // 2),
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
        self.attn = TemporalSelfAttention(embed_dim=filters, num_heads=attn_heads)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(filters, num_classes)

    def get_embedding(self, x):
        """Extract embedding (representation before final classification layer)"""
        x = self.stem(x)
        x = self.resnet(x)
        x = self.attn(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        return x

    def forward(self, x):
        x = self.get_embedding(x)
        return self.classifier(x)

# --- Attentive pooling variant ---

class ResNet1DClassifierAttnPooling(nn.Module):
    def __init__(self, in_channels, num_classes, base_filters=64, num_blocks=3,
                 kernel_size=7, dropout=0.3, norm="gn", gn_groups=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=kernel_size, padding=kernel_size // 2),
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
                dropout=0.1,
                norm=norm, gn_groups=gn_groups
            ))
            filters *= 2

        self.resnet = nn.Sequential(*blocks)
        self.attentive_pool = SelfAttentionPooling1D(filters)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(filters, num_classes)

    def get_embedding(self, x):
        """Extract embedding (representation before final classification layer)"""
        x = self.stem(x)
        x = self.resnet(x)
        x = self.attentive_pool(x)
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


# --- ResNet1D with Self-Attention + Attentive Pooling ---

class ConvBNAct1d(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=None, norm="gn", gn_groups=8, act="relu"):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv1d(c_in, c_out, k, stride=s, padding=p, bias=False)
        if norm == "bn":
            self.norm = nn.BatchNorm1d(c_out)
        elif norm == "gn":
            self.norm = nn.GroupNorm(num_groups=min(gn_groups, c_out), num_channels=c_out)
        else:
            self.norm = nn.Identity()
        self.act = nn.ReLU(inplace=True) if act == "relu" else nn.GELU()

    def forward(self, x):  # (B, C, T)
        return self.act(self.norm(self.conv(x)))

class ResBlock1d(nn.Module):
    def __init__(self, c, k=3, norm="gn", gn_groups=8, dropout=0.0):
        super().__init__()
        self.conv1 = ConvBNAct1d(c, c, k, norm=norm, gn_groups=gn_groups)
        self.conv2 = ConvBNAct1d(c, c, k, norm=norm, gn_groups=gn_groups, act="gelu")
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.drop(out)
        out = self.conv2(out)
        return x + out

class AttentivePool1d(nn.Module):
    """Additive attention over time: weights -> weighted sum."""
    def __init__(self, d, hidden=128, dropout=0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1, bias=False),
        )

    def forward(self, H):  # H: (B, T, D)
        scores = self.proj(H).squeeze(-1)        # (B, T)
        alpha = torch.softmax(scores, dim=1)     # (B, T)
        pooled = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)  # (B, D)
        return pooled, alpha

class ResNet1D_DualAttn(nn.Module):
    """
    ResNet1D backbone -> temporal downsample -> SelfAttention -> AttentivePooling.
    Fuse both pooled vectors for classification.
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        d_model: int = 128,
        n_blocks: int = 6,
        ksize: int = 5,
        attn_heads: int = 4,
        attn_dropout: float = 0.1,
        attn_stride: int = 4,         # temporal reduction before attention
        pool_hidden: int = 128,
        dropout: float = 0.1,
        norm: str = "gn",
        gn_groups: int = 8,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct1d(in_channels, d_model, k=7, s=2, norm=norm, gn_groups=gn_groups),
            ResBlock1d(d_model, k=ksize, norm=norm, gn_groups=gn_groups, dropout=dropout),
        )
        self.layers = nn.Sequential(*[
            ResBlock1d(d_model, k=ksize, norm=norm, gn_groups=gn_groups, dropout=dropout)
            for _ in range(max(0, n_blocks - 1))
        ])

        # temporal reduction to curb O(T^2) in attention
        self.reduce = nn.Conv1d(d_model, d_model, kernel_size=3, stride=max(1, attn_stride), padding=1, bias=False)

        # Self-attention (batch_first=True expects (B, T, D))
        self.self_attn = nn.MultiheadAttention(d_model, attn_heads, dropout=attn_dropout, batch_first=True)

        # Attentive pooling head
        self.attn_pool = AttentivePool1d(d_model, hidden=pool_hidden, dropout=dropout)

        # Fusion + classifier
        fused_dim = 2 * d_model
        self.head = nn.Sequential(
            nn.Linear(fused_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def get_embedding(self, x):
        """Extract embedding (representation before final classification layer)"""
        # x: (B, C, T)
        x = self.stem(x)           # (B, D, T/2)
        x = self.layers(x)         # (B, D, T/2)
        x = self.reduce(x)         # (B, D, T')
        H = x.transpose(1, 2)      # (B, T', D)

        # Self-attention refinement
        H_sa, _ = self.self_attn(H, H, H)  # (B, T', D)

        # Pool both views
        v_pool, _ = self.attn_pool(H)      # attentive pooling on raw H
        v_sa = H_sa.mean(dim=1)            # mean-pool self-attended sequence

        fused = torch.cat([v_sa, v_pool], dim=1)  # (B, 2D)
        return fused

    def forward(self, x):  # x: (B, C, T)
        fused = self.get_embedding(x)
        logits = self.head(fused)  # (B, C)
        return logits


# --- TransformerEncoder (META-SiM inspired) ---

class TransformerEncoder(nn.Module):
    """
    Transformer-based encoder for trace classification, inspired by META-SiM.
    Architecture:
      1. TCN tokenizer: converts raw traces to tokens
      2. Learnable positional embeddings
      3. Transformer encoder (multi-head self-attention + FFN)
      4. Pooling (CLS-style or mean pooling)
      5. Classification head
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        # Tokenizer (TCN-based)
        tcn_channels: list = None,      # e.g., [64, 128] for 2-layer TCN
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.05,
        # Transformer
        d_model: int = 96,               # embedding dimension
        num_layers: int = 4,             # transformer depth
        num_heads: int = 4,              # attention heads
        dim_feedforward: int = 384,      # 4 × d_model
        dropout: float = 0.1,
        activation: str = "gelu",
        # Positional embedding
        max_seq_len: int = 400,          # max number of tokens after TCN
        # Pooling strategy
        pooling: str = "cls",            # "cls" or "mean"
        # Normalization
        norm: str = "gn",
        gn_groups: int = 8,
        # Learning rate schedule (for future HPO)
        warmup_steps: int = 100,         # exposed for HPO
        max_lr: float = 1e-3,            # exposed for HPO
    ):
        super().__init__()

        if tcn_channels is None:
            tcn_channels = [64, 128]

        self.d_model = d_model
        self.pooling = pooling.lower()
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr

        # TCN tokenizer
        tcn_layers = []
        for i in range(len(tcn_channels)):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else tcn_channels[i - 1]
            out_ch = tcn_channels[i]
            tcn_layers.append(
                TemporalBlock(in_ch, out_ch, tcn_kernel_size, stride=1,
                              dilation=dilation, dropout=tcn_dropout,
                              norm=norm, gn_groups=gn_groups)
            )
        self.tcn = nn.Sequential(*tcn_layers)

        # Project TCN output to d_model
        self.proj = nn.Linear(tcn_channels[-1], d_model)

        # Learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        # CLS token (only if using cls pooling)
        if self.pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def get_embedding(self, x):
        """Extract embedding (representation before final classification layer)"""
        # x: (B, C, T)
        B = x.shape[0]

        # TCN tokenization
        tokens = self.tcn(x)  # (B, tcn_channels[-1], T')
        tokens = tokens.transpose(1, 2)  # (B, T', tcn_channels[-1])
        tokens = self.proj(tokens)  # (B, T', d_model)

        # Scale embeddings (standard transformer practice)
        tokens = tokens * math.sqrt(self.d_model)

        T = tokens.shape[1]

        # Add positional embeddings (handle sequences longer than max_seq_len)
        if T <= self.pos_embedding.shape[1]:
            tokens = tokens + self.pos_embedding[:, :T, :]
        else:
            # If sequence is longer, tile the positional embeddings
            n_repeats = (T // self.pos_embedding.shape[1]) + 1
            pos_emb_extended = self.pos_embedding.repeat(1, n_repeats, 1)
            tokens = tokens + pos_emb_extended[:, :T, :]

        # Prepend CLS token if using cls pooling
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
            tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, T'+1, d_model)

        # Transformer encoding
        encoded = self.transformer(tokens)  # (B, T'[+1], d_model)

        # Pooling
        if self.pooling == "cls":
            embedding = encoded[:, 0, :]  # (B, d_model) - take CLS token
        elif self.pooling == "mean":
            embedding = encoded.mean(dim=1)  # (B, d_model) - mean pooling
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        return embedding

    def forward(self, x):
        embedding = self.get_embedding(x)
        return self.head(embedding)



# --- Model registry and factory ---

MODEL_REGISTRY = {
    "conv_gru": ConvGRUClassifier,
    "resnet1d": ResNet1DClassifier,
    "resnet1d_selfattn": ResNet1DClassifierSelfAttn,
    "resnet1d_attnpool": ResNet1DClassifierAttnPooling,
    "tcn": TCN,
    "orig_conv_gru": OrigConvGRUClassifier,
    "resnet1d_dualattn": ResNet1D_DualAttn,
    "transformer_encoder": TransformerEncoder,
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

    "resnet1d_dualattn": {
        "d_model": 128,
        "n_blocks": 6,
        "ksize": 5,
        "attn_heads": 4,
        "attn_dropout": 0.1,
        "attn_stride": 8,
        "pool_hidden": 128,
        "dropout": 0.1,
        "norm": "gn",
        "gn_groups": 8,
    },

    "transformer_encoder": {
        "tcn_channels": [64, 128],
        "tcn_kernel_size": 3,
        "tcn_dropout": 0.05,
        "d_model": 96,
        "num_layers": 4,
        "num_heads": 4,
        "dim_feedforward": 384,  # 4 × d_model
        "dropout": 0.1,
        "activation": "gelu",
        "max_seq_len": 400,
        "pooling": "cls",
        "norm": "gn",
        "gn_groups": 8,
        "warmup_steps": 500,       # Increased for better stability
        "max_lr": 5e-5,            # Much lower LR for transformers
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
