import torch.nn as nn

class FCNN(nn.Module):
    def __init__(self, window_size, vocab_size, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear((window_size + 3) * embed_dim, hidden_dim) # window_size + 3 because 3 is the context_size (M: L: K:)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)  # output logits for all tokens

    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)  # -> (batch_size, seq_len, embed_dim)
        x = x.view(x.size(0), -1)  # flatten: (batch_size, seq_len * embed_dim)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # logits (no softmax here since crossentropy expects logits)
        return x