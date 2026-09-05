"""Core-ML-exportable sequence classifiers for on-device ASL gloss recognition.

Architectures (all keep NMM channels first-class via tail features / aux head):
  - PoseLSTMClassifier: BiLSTM + NMM-conditioned temporal attention (v3-stable)
  - TemporalConvBiLSTM: depthwise temporal conv front-end + BiLSTM + NMM attn
  - PoseTransformerEncoder: lightweight Transformer encoder + NMM pool
"""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


if nn is not None:

    class PoseLSTMClassifier(nn.Module):
        def __init__(
            self,
            input_dim: int = 170,
            hidden_dim: int = 224,
            num_layers: int = 3,
            num_classes: int = 25,
            bidirectional: bool = True,
            dropout: float = 0.3,
            nmm_dim: int = 11,
            attn_heads: int = 4,  # ckpt compat; unused in stable attn
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes
            self.nmm_dim = nmm_dim
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            direction = 2 if bidirectional else 1
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim * direction + nmm_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
            self.nmm_aux = nn.Sequential(
                nn.Linear(hidden_dim * direction, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 3),
            )
            self.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * direction, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x, return_aux: bool = False):  # (B, T, D)
            out, _ = self.lstm(x)
            nmm = x[:, :, -self.nmm_dim :]
            scores = self.attn(torch.cat([out, nmm], dim=-1)).squeeze(-1)
            weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
            pooled = (out * weights).sum(dim=1)
            logits = self.fc(pooled)
            if return_aux:
                return logits, self.nmm_aux(pooled)
            return logits

    class TemporalConvBiLSTM(nn.Module):
        """Temporal conv frontend (local motion) + BiLSTM + NMM attention."""

        def __init__(
            self,
            input_dim: int = 170,
            hidden_dim: int = 192,
            num_layers: int = 2,
            num_classes: int = 25,
            bidirectional: bool = True,
            dropout: float = 0.3,
            nmm_dim: int = 11,
            conv_channels: int = 192,
            attn_heads: int = 4,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes
            self.nmm_dim = nmm_dim
            self.frontend = nn.Sequential(
                nn.Conv1d(input_dim, conv_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(conv_channels),
                nn.GELU(),
                nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2, groups=4, bias=False),
                nn.BatchNorm1d(conv_channels),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.lstm = nn.LSTM(
                input_size=conv_channels,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            direction = 2 if bidirectional else 1
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim * direction + nmm_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
            self.nmm_aux = nn.Sequential(
                nn.Linear(hidden_dim * direction, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 3),
            )
            self.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * direction, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x, return_aux: bool = False):
            # x: (B, T, D) → conv wants (B, D, T)
            h = self.frontend(x.transpose(1, 2)).transpose(1, 2)
            out, _ = self.lstm(h)
            nmm = x[:, :, -self.nmm_dim :]
            scores = self.attn(torch.cat([out, nmm], dim=-1)).squeeze(-1)
            weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
            pooled = (out * weights).sum(dim=1)
            logits = self.fc(pooled)
            if return_aux:
                return logits, self.nmm_aux(pooled)
            return logits

    class PoseTransformerEncoder(nn.Module):
        """Lightweight Transformer encoder; NMM channels gate the CLS pool."""

        def __init__(
            self,
            input_dim: int = 170,
            hidden_dim: int = 192,
            num_layers: int = 3,
            num_classes: int = 25,
            bidirectional: bool = True,  # unused; API compat
            dropout: float = 0.25,
            nmm_dim: int = 11,
            attn_heads: int = 4,
            max_len: int = 64,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_classes = num_classes
            self.nmm_dim = nmm_dim
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.pos = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=attn_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.nmm_gate = nn.Sequential(
                nn.Linear(nmm_dim, hidden_dim),
                nn.Sigmoid(),
            )
            self.nmm_aux = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 3),
            )
            self.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
            nn.init.normal_(self.pos, mean=0.0, std=0.02)

        def forward(self, x, return_aux: bool = False):
            b, t, _ = x.shape
            h = self.input_proj(x) + self.pos[:, :t, :]
            h = self.encoder(h)
            nmm = x[:, :, -self.nmm_dim :]
            # Mean NMM over time → gate; pool with NMM-biased attention scores
            gate = self.nmm_gate(nmm.mean(dim=1)).unsqueeze(1)  # (B,1,H)
            scores = (h * gate).sum(dim=-1)  # (B,T)
            weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
            pooled = (h * weights).sum(dim=1)
            logits = self.fc(pooled)
            if return_aux:
                return logits, self.nmm_aux(pooled)
            return logits

    def build_sequence_model(arch: str, **kwargs) -> nn.Module:
        arch = (arch or "poselstm").lower().replace("_", "-")
        if arch in {"poselstm", "poselstm-v3", "lstm"}:
            return PoseLSTMClassifier(**kwargs)
        if arch in {"tcn-bilstm", "temporal-conv", "conv-lstm", "tcn"}:
            return TemporalConvBiLSTM(**kwargs)
        if arch in {"transformer", "pose-transformer", "xfmr"}:
            return PoseTransformerEncoder(**kwargs)
        raise ValueError(f"unknown arch '{arch}'")

else:  # pragma: no cover

    class PoseLSTMClassifier:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for PoseLSTMClassifier")

    def build_sequence_model(arch: str, **kwargs):
        raise RuntimeError("PyTorch is required")
