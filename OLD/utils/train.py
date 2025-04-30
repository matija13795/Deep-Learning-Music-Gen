import os
import torch


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_one_epoch_rnn(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0

    for inputs, targets, lengths in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        logits, _ = model(inputs, lengths)
        # Flatten the logits and targets for the loss function
        logits = logits.view(-1, logits.size(-1))  # Shape: [batch_size * seq_len, vocab_size]
        targets = targets.view(-1)  # Shape: [batch_size * seq_len]

        loss = loss_fn(logits, targets)
        loss.backward()
        
        # Check the gradient norms
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                if grad_norm > 10:
                    print(f"Exploding gradient detected for {name}: {grad_norm}")
                elif grad_norm < 1e-4:
                    print(f"Vanishing gradient detected for {name}: {grad_norm}")

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_rnn(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs, targets, lengths in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs, lengths)
            logits = logits.view(-1, logits.size(-1))   # [batch_size * seq_len, vocab_size]
            targets = targets.view(-1)                  # [batch_size * seq_len]
            loss = loss_fn(logits, targets)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def train_model(model, train_loader, val_loader, loss_fn, optimizer, device, epochs, save_path, model_type):
    for epoch in range(epochs):
        if model_type == "FCNN":
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
            val_loss = evaluate(model, val_loader, loss_fn, device)

        elif model_type == "RNN":
            train_loss = train_one_epoch_rnn(model, train_loader, loss_fn, optimizer, device)
            val_loss = evaluate_rnn(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Make sure the directory exists
    os.makedirs("saved_models", exist_ok=True)
    
    # Save the final model to saved_models
    torch.save(model.state_dict(), save_path)
    print(f"Final model saved to {save_path}")