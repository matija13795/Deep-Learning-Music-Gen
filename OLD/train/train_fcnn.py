import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from preprocessing.main_preprocess import preprocess_abc_dataset
from datasets.fcnn_dataset import FCNNDataset
from models.fcnn import FCNN
from utils.train import train_model, evaluate

def main():
    # === Preprocessing ===
    vocab, inv_vocab, indexed_melodies, token_freq, normalized_melodies = preprocess_abc_dataset("data/")

    # === Dataset and DataLoaders ===
    WINDOW_SIZE = 16
    dataset = FCNNDataset(indexed_melodies, vocab, inv_vocab, WINDOW_SIZE)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024)
    test_loader = DataLoader(test_ds, batch_size=1024)

    # === Model, Loss, Optimizer ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FCNN(WINDOW_SIZE, vocab_size=len(vocab), embed_dim=128, hidden_dim=512, dropout=0.5).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    # === Training ===
    print("Starting training...")
    train_model(model, train_loader, val_loader, loss_fn, optimizer, device, epochs=100, save_path="saved_models/fcnn_model.pt", model_type="FCNN")

    # === Evaluate on Test Set ===
    test_loss = evaluate(model, test_loader, loss_fn, device)
    print(f"Test Loss = {test_loss:.4f}")


if __name__ == "__main__":
    main()
