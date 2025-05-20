import os
import gc
import torch
from tqdm import tqdm
from torch import nn

def train_memory_efficient_model(model, train_loader, val_loader=None, num_epochs=5, learning_rate=0.0001, checkpoint_dir="./checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    best_val_loss = float("inf")
    patience_counter = 0
    try:
        for epoch in range(num_epochs):
            gc.collect()
            torch.cuda.empty_cache()
            model.train()
            running_loss = 0.0
            valid_batches = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for batch_idx, (patches, _) in enumerate(pbar):
                if patches.numel() == 0:
                    continue
                patches = patches.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    outputs = model(patches)
                    loss = criterion(outputs, patches)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
                valid_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                if batch_idx % 10 == 0:
                    del outputs, loss, patches
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            if valid_batches > 0:
                train_loss = running_loss / valid_batches
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs}, No valid batches!")
                continue
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt"))
            if val_loader:
                val_loss = validate_memory_efficient(model, val_loader, criterion, device)
                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pt"))
                    print(f"New best model saved with val_loss: {val_loss:.6f}")
                else:
                    patience_counter += 1
                    if patience_counter >= 3:
                        print("Early stopping triggered!")
                        break
    except Exception as e:
        print(f"Error during training: {e}")
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "emergency_model.pt"))
        raise
    return model

def validate_memory_efficient(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    valid_batches = 0
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")
        for patches, _ in pbar:
            if patches.numel() == 0:
                continue
            patches = patches.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(patches)
                loss = criterion(outputs, patches)
            running_loss += loss.item()
            valid_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            del outputs, patches, loss
    if valid_batches > 0:
        val_loss = running_loss / valid_batches
        print(f"Validation Loss: {val_loss:.6f}")
        return val_loss
    else:
        print("No valid validation batches!")
        return float("inf")

def train_classifier(model, train_loader, val_loader, optimizer, criterion, scheduler=None,
                     device="cuda", num_epochs=5, early_stopping_patience=None, checkpoint_path=None, use_amp=False):
    model = model.to(device)
    scaler = torch.cuda.amp.GradScaler() if use_amp and device!="cpu" else None
    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{num_epochs}] Train", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(inputs)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(inputs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            running_loss += loss.item()*labels.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds==labels).sum().item()
            running_total += labels.size(0)
            pbar.set_postfix(loss=running_loss/running_total, acc=100.*running_correct/running_total)
        train_loss = running_loss/running_total
        train_acc = 100.*running_correct/running_total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        model.eval()
        val_loss = 0.0; val_correct =0; val_total =0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"[Epoch {epoch}/{num_epochs}] Val", leave=False)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits = model(inputs)
                        loss = criterion(logits, labels)
                else:
                    logits = model(inputs)
                    loss = criterion(logits, labels)
                val_loss += loss.item()*labels.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds==labels).sum().item()
                val_total += labels.size(0)
                pbar.set_postfix(val_loss=val_loss/val_total, val_acc=100.*val_correct/val_total)
        val_loss_epoch = val_loss/val_total
        val_acc_epoch = 100.*val_correct/val_total
        history["val_loss"].append(val_loss_epoch)
        history["val_acc"].append(val_acc_epoch)
        if scheduler is not None:
            if hasattr(scheduler, "step") and scheduler.__class__.__name__=="ReduceLROnPlateau":
                scheduler.step(val_loss_epoch)
            else:
                scheduler.step()
        if checkpoint_path is not None and val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            torch.save(model.state_dict(), checkpoint_path)
            epochs_no_improve = 0
            print(f"→ New best model saved (val_loss={best_val_loss:.4f})")
        elif early_stopping_patience is not None:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"→ Early stopping after {epoch} epochs without improvement.")
                break
        print(f"Epoch {epoch}/{num_epochs} Train: loss={train_loss:.4f}, acc={train_acc:.2f}% | Val: loss={val_loss_epoch:.4f}, acc={val_acc_epoch:.2f}%")
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model, history

def evaluate(model, test_loader, device="cuda", use_amp=False):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0; total_samples = 0; correct1 = 0; correct5 = 0
    test_pbar = tqdm(total=len(test_loader), desc="Testing", unit="batch", leave=True)
    with torch.no_grad():
        for inputs, labels in test_loader:
            B, P, C, H, W = inputs.shape
            inputs = inputs.view(B*P, C, H, W).to(device)
            labels = labels.to(device)
            if use_amp and device!="cpu":
                with torch.cuda.amp.autocast():
                    logits = model(inputs)
            else:
                logits = model(inputs)
            num_classes = logits.size(1)
            avg_logits = logits.view(B, P, num_classes).mean(dim=1)
            loss = criterion(avg_logits, labels)
            total_loss += loss.item()*B
            _, pred1 = avg_logits.max(1)
            correct1 += pred1.eq(labels).sum().item()
            _, pred5 = avg_logits.topk(5, dim=1, largest=True, sorted=True)
            correct5 += (pred5==labels.view(-1,1)).any(dim=1).sum().item()
            total_samples += B
            test_pbar.set_postfix({'loss': f"{loss.item():.4f}",
                                   'top1_acc': f"{100.*correct1/total_samples:.2f}%",
                                   'top5_acc': f"{100.*correct5/total_samples:.2f}%"})
            test_pbar.update()
    test_pbar.close()
    avg_loss = total_loss/total_samples
    top1_acc = 100.*correct1/total_samples
    top5_acc = 100.*correct5/total_samples
    print(f"\nTest Loss: {avg_loss:.4f}")
    print(f"Top-1 Accuracy: {top1_acc:.2f}% | Top-1 Error: {100.-top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}% | Top-5 Error: {100.-top5_acc:.2f}%")
    return 100.-top1_acc, 100.-top5_acc