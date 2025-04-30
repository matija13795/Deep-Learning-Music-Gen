import torch
from typing import List, Tuple
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from tokenizer import ABCTokenizer


class LeadSheetDataset(Dataset):
    """PyTorch `Dataset` for a list of ABC tunes."""

    def __init__(self, tunes: List[str], tokenizer: ABCTokenizer):
        self.tunes = tunes
        self.tokenizer = tokenizer
        self.encoded: List[List[int]] = [self.tokenizer.encode(t) for t in self.tunes]

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> Tensor:
        return torch.tensor(self.encoded[idx], dtype=torch.long)


def collate_fn(batch: List[Tensor], pad_idx: int) -> Tuple[Tensor, Tensor]:
    """Pad variable-length songs and create (input, target) tensors.

    Each song is shifted right by one char so the network predicts the next
    char at every timestep.

    Returns
    -------
    x : LongTensor  [batch, seq_len]
        Input sequence without the final char (because final char has no target).
    y : LongTensor  [batch, seq_len]
        Target sequence (next char for every input char).
    """
    # Pad to equal length … shape → [batch, max_len]
    padded = pad_sequence(batch, batch_first=True, padding_value=pad_idx)

    # Split into x / y (next‑token prediction).
    x = padded[:, :-1]
    y = padded[:, 1:]
    return x, y