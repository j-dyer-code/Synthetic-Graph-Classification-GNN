import os
import torch

from src import config

def evaluate_one_epoch(model, loader, criterion, device):
    """
    Evaluates the model for one epoch.
    """
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, _ = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
    return total_loss / total, correct / total

def final_training_loop(model, arch, optimizer, scheduler, criterion, train_loader, val_loader, device):
    """
    Trains a model with early stopping and saves the best version.
    """
    best_val_loss, best_epoch, epochs_no_improve = float('inf'), 0, 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    save_path = os.path.join(config.FINAL_MODELS_DIR, f"{arch}_best.pt")

    for epoch in range(1, config.MAX_EPOCHS_TRAIN + 1):
        model.train()
        total_train_loss, correct_train, total_train = 0, 0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_train_loss += loss.item() * batch.num_graphs
            total_train += batch.num_graphs
            correct_train += (logits.argmax(dim=1) == batch.y).sum().item()
        
        history['train_loss'].append(total_train_loss / total_train)
        history['train_acc'].append(correct_train / total_train)

        val_loss, val_acc = evaluate_one_epoch(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch:03d} | Train Loss: {history['train_loss'][-1]:.4f} | Train Acc: {history['train_acc'][-1]:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved to {save_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}.")
            break

    return {**history, "best_epoch": best_epoch, "best_val_loss": best_val_loss, "checkpoint": save_path}
