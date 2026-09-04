"""PoseLSTM scaffold — drop-in for optional .pt weights.

STUB / SCAFFOLD: architecture matches common ASL landmark classifiers
(AWS PoseLSTM-style 32-frame windows). Replace or extend when loading
Uni-Sign pose encoders (see MODELS.md).
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
            input_dim: int = 139,
            hidden_dim: int = 256,
            num_layers: int = 2,
            num_classes: int = 25,
            bidirectional: bool = True,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            direction = 2 if bidirectional else 1
            self.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * direction, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x):  # (B, T, D)
            out, _ = self.lstm(x)
            pooled = out.mean(dim=1)
            return self.fc(pooled)

else:  # pragma: no cover

    class PoseLSTMClassifier:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for PoseLSTMClassifier")
