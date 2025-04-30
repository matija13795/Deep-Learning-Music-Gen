import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from preprocessing.main_preprocess import preprocess_abc_dataset
from datasets.rnn_dataset import RNNDataset, collate_fn
from models.rnn import RNNModel
from utils.train import train_model, evaluate_rnn

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Preprocessing ===
    vocab, inv_vocab, indexed_melodies, token_freq, normalized_melodies = preprocess_abc_dataset("data/")
    dataset = RNNDataset(indexed_melodies, start_token_idx=1, end_token_idx=2)
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # === Model ===
    model = RNNModel(
        vocab_size=len(vocab),
        embed_dim=128,
        hidden_dim=512,
        num_layers=2,
        dropout=0.4
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    # === Training ===
    train_model(model, train_loader, val_loader, loss_fn, optimizer, device, epochs=100, save_path="saved_models/rnn_model.pt",  model_type="RNN")

    # === Evaluation ===
    test_loss = evaluate_rnn(model, test_loader, loss_fn, device)
    print(f"Final Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()