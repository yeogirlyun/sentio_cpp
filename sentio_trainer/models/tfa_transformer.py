import torch
from torch import nn


class TFA_Transformer(nn.Module):
    """
    Input:  x [B, T, F]  (F=55, T=64)
    Output: logits [B, 1] for next bar up/down
    """
    def __init__(self, F: int, T: int = 64, d_model: int = 96, nhead: int = 4,
                 num_layers: int = 2, ffn_hidden: int = 192, dropout: float = 0.0):
        super().__init__()
        self.F, self.T = F, T
        self.in_proj = nn.Linear(F, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=ffn_hidden,
                                         batch_first=True, dropout=dropout,
                                         activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)  # logits

    def forward(self, x):                 # x: [B, T, F]
        z = self.in_proj(x)               # [B,T,D]
        z = self.encoder(z)               # [B,T,D]
        z = z[:, -1, :]                   # last step representation
        return self.head(z)               # [B,1]


class SeqScaler(nn.Module):
    """
    Apply per-feature scaling at every timestep: (x - mean) * inv_std
    mean/inv_std are [F], broadcast across T.
    """
    def __init__(self, mean, inv_std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("inv_std", torch.tensor(inv_std, dtype=torch.float32))

    def forward(self, x):                 # x: [B,T,F]
        return (x - self.mean) * self.inv_std


class TFA_Wrapped(nn.Module):
    """
    TorchScript-exportable module:
      forward(x[B,T,F]) -> logits[B,1]
    """
    def __init__(self, core, scaler):
        super().__init__()
        self.scaler = scaler
        self.core = core

    def forward(self, x):
        x = self.scaler(x)
        return self.core(x)               # logits


