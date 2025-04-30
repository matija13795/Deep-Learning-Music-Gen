import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)


    def forward(self, input_seqs, lengths, hidden=None):
        """
        input_seqs: (B, T)
        lengths: (B,) -> actual lengths of sequences (before padding)
        hidden: initial hidden state (num_layers, B, H)

        Returns:
            logits: (B, T, vocab_size)
        """
        embedded = self.embedding(input_seqs)  # (B, T, E), where E = embed_dim
        # we don’t want the RNN to process the padding (the 0s). That’s where packing comes in:
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)  # (B, T, H)
        logits = self.fc(outputs)  # (B, T, vocab_size)

        return logits, hidden