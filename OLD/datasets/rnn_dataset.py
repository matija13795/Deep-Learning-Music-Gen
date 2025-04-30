import torch
from torch.utils.data import Dataset


class RNNDataset(Dataset):
    def __init__(self, melodies, start_token_idx, end_token_idx):
        self.data = melodies
        self.start = start_token_idx
        self.end = end_token_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        melody = self.data[idx]
        
        # Add <START> and <END> tokens
        input_seq = [self.start] + melody
        target_seq = melody + [self.end]

        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)


def collate_fn(batch, pad_token_idx=0):
    """
    Pads sequences to the max length in the batch
    Returns:
        inputs_padded: (B, T)
        targets_padded: (B, T)
        lengths: original lengths of sequences before padding
    """
    inputs, targets = zip(*batch)
    input_lengths = torch.tensor([len(seq) for seq in inputs], dtype=torch.long)

    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_token_idx)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_token_idx)

    return inputs_padded, targets_padded, input_lengths