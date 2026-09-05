"""PoseLSTM + NMM-aware temporal attention — drop-in for .pt / Core ML export.

v3-stable: proven BiLSTM + NMM-conditioned attention (ships well), with optional
deeper layers / larger hidden. Avoids fragile MHA export quirks that hurt holdout.
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
            attn_heads: int = 4,  # kept for ckpt compat; unused in stable attn
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

else:  # pragma: no cover

    class PoseLSTMClassifier:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for PoseLSTMClassifier")
