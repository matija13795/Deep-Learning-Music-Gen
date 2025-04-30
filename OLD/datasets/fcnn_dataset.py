import torch
from torch.utils.data import Dataset


class FCNNDataset(Dataset):
    def __init__(self, indexed_sequences, vocab, inv_vocab, window_size=16):
        """
        Args:
            indexed_sequences (List[List[int]]): List of tokenized melody sequences (as integers).
            vocab (Dict[str, int]): Token-to-index mapping.
            inv_vocab (Dict[int, str]): Index-to-token mapping.
            window_size (int): Number of tokens to include in each input window.
        """
        self.window_size = window_size
        self.vocab = vocab
        self.inv_vocab = inv_vocab
        self.inputs = []
        self.labels = []

        # Go through each melody and turn it into sliding input windows
        for seq in indexed_sequences:
            context = self.extract_context(seq)  # this grabs the first 3 tokens (assumed M:/L:/K:)
            body = seq[3:]  # remove M:/L:/K: from melody body
            
            windows, labels = self.generate_windows(body, context)
            self.inputs.extend(windows)
            self.labels.extend(labels)

    def extract_context(self, seq):
        return seq[:3]  # assuming M:/L:/K: are always at the beginning

    def generate_windows(self, seq, context):
        windows = []
        labels = []
        for i in range(len(seq) - self.window_size):
            input_window = seq[i:i+self.window_size]
            target_token = seq[i+self.window_size]
            full_input = context + input_window
            windows.append(torch.tensor(full_input, dtype=torch.long))
            labels.append(torch.tensor(target_token, dtype=torch.long))
        return windows, labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]
