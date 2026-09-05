"""PoseLSTM + NMM-aware temporal attention — drop-in for .pt / Core ML export.

Architecture matches ASL landmark classifiers (32-frame windows, FEATURE_DIM=170).
NMM channels (last 11 dims) gate temporal attention so face/body grammar is
first-class, not drowned by hand kinematics.
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
            hidden_dim: int = 192,
            num_layers: int = 2,
            num_classes: int = 25,
            bidirectional: bool = True,
            dropout: float = 0.25,
            nmm_dim: int = 11,
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
            # Auxiliary NMM head: question / negation / emphasis logits
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
            # Temporal attention conditioned on NMM channels
            scores = self.attn(torch.cat([out, nmm], dim=-1)).squeeze(-1)  # (B, T)
            weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
            pooled = (out * weights).sum(dim=1)
            logits = self.fc(pooled)
            if return_aux:
                return logits, self.nmm_aux(pooled)
            return logits

else:  # pragma: no cover

    class PoseLSTMClassifier:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for PoseLSTMClassifier")
