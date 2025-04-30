from torch import nn, Tensor
from typing import Optional


class GRUModel(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        num_layers: int = 2,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0  # dropout only if >1 layer
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    # -----------------------------------------------------------------------------------------------
    def forward(self, x: Tensor, h0: Optional[Tensor] = None, return_hidden: bool = False) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x  : [batch, seq_len] LongTensor - token IDs
        h0 : optional initial hidden state  [num_layers, batch, hidden]
        return_hidden : whether to return hidden state
        """
        emb = self.embed(x)
        out, h = self.gru(emb, h0)
        logits = self.fc_out(out)  # shape [batch, seq_len, vocab_size]
        return (logits, h) if return_hidden else logits


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=1024, num_layers=2,
                pad_idx=0, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None, return_hidden=False):
        """Forward pass.

        Parameters
        ----------
        x : [batch, seq_len] LongTensor - token IDs
        hidden : optional initial hidden state (h0, c0)
        return_hidden : whether to return hidden state
        """
        emb = self.embed(x)
        out, hidden = self.lstm(emb, hidden)
        out = self.dropout(out)
        logits = self.fc_out(out)  # shape [batch, seq_len, vocab_size]
        return (logits, hidden) if return_hidden else logits
