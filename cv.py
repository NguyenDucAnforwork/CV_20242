import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
from io import BytesIO
import os
import random
import math # Needed for ceiling division
import torch.nn as nn
import torch.cuda.amp as amp
import gc
from tqdm import tqdm
from functools import partial
import warnings

# Helper function to extract patches
def extract_patches(image_array, num_patch=3, patch_size=(105, 105)):
    patches = []
    if image_array.ndim == 2: # Grayscale
        h, w = image_array.shape
    elif image_array.ndim == 3: # Color
        h, w, _ = image_array.shape
    else:
        return []

    patch_h, patch_w = patch_size
    if h < patch_h or w < patch_w:
        return []

    for _ in range(num_patch):
        x = np.random.randint(0, w - patch_w + 1)
        y = np.random.randint(0, h - patch_h + 1)
        patch = image_array[y:y + patch_h, x:x + patch_w]
        patches.append(patch)
    return patches

class CombinedImageDataset(Dataset):
    def __init__(self, jpeg_dir, bcf_file, label_file, num_patch=3, patch_size=(105, 105)):
        self.jpeg_dir = jpeg_dir
        self.bcf_file = bcf_file
        self.label_file = label_file
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.jpeg_data = []
        self.bcf_data = []
        self._load_jpeg_data(jpeg_dir)
        self._load_bcf_data(bcf_file, label_file)

    def _load_jpeg_data(self, jpeg_dir):
        image_filenames = [f for f in os.listdir(jpeg_dir) if f.endswith('.jpeg') or f.endswith('.jpg')]
        self.jpeg_data = [(os.path.join(jpeg_dir, f), 0) for f in image_filenames]
        print(f"Loaded {len(self.jpeg_data)} .jpeg images.")

    def _load_bcf_data(self, bcf_file, label_file):
        try:
            with open(label_file, 'rb') as f:
                self.labels = np.frombuffer(f.read(), dtype=np.uint32)
            with open(bcf_file, 'rb') as f:
                self.num_images = np.frombuffer(f.read(8), dtype=np.int64)[0]
                sizes_bytes = f.read(self.num_images * 8)
                self.image_sizes = np.frombuffer(sizes_bytes, dtype=np.int64)
                self.data_start_offset = 8 + self.num_images * 8
                self.image_offsets = np.zeros(self.num_images + 1, dtype=np.int64)
                np.cumsum(self.image_sizes, out=self.image_offsets[1:])
                for idx in range(self.num_images):
                    self.bcf_data.append((idx, self.labels[idx]))
            print(f"Loaded {len(self.bcf_data)} .bcf images.")
        except Exception as e:
            print(f"Error loading .bcf data: {e}")

    def __len__(self):
        return len(self.jpeg_data) + len(self.bcf_data)

    def _extract_patches(self, img_array):
        h, w = img_array.shape
        patch_h, patch_w = self.patch_size
        if h < patch_h or w < patch_w:
            return []
        return extract_patches(img_array, self.num_patch, self.patch_size)

    def __getitem__(self, idx):
        max_retries = 3
        for _ in range(max_retries):
            try:
                if idx < len(self.jpeg_data):
                    img_path, label = self.jpeg_data[idx]
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            img = Image.open(img_path)
                            img.verify()
                        img = Image.open(img_path).convert('L')
                        img_array = np.array(img)
                        patches = self._extract_patches(img_array)
                        del img, img_array
                        return patches, label
                    except (OSError, IOError, ValueError) as e:
                        print(f"Warning: Corrupt image at {img_path}: {e}")
                        return [], -1
                else:
                    bcf_idx = idx - len(self.jpeg_data)
                    if bcf_idx >= len(self.bcf_data): return [], -1
                    label = self.bcf_data[bcf_idx][1]
                    offset = self.image_offsets[bcf_idx]
                    size = self.image_sizes[bcf_idx]
                    try:
                        with open(self.bcf_file, 'rb') as f:
                            f.seek(self.data_start_offset + offset)
                            image_bytes = f.read(size)
                        buffer = BytesIO(image_bytes)
                        img = Image.open(buffer)
                        img.verify()
                        buffer.seek(0)
                        img = Image.open(buffer).convert('L')
                        img_array = np.array(img)
                        patches = self._extract_patches(img_array)
                        del img, img_array, buffer, image_bytes
                        return patches, label
                    except (OSError, IOError, ValueError) as e:
                        print(f"Warning: Corrupt BCF image at index {bcf_idx}: {e}")
                        return [], -1
            except Exception as e:
                print(f"Unexpected error processing idx {idx}: {e}")
            idx = (idx + 1) % len(self)
        return [], -1

def memory_efficient_patch_collate_fn(batch, patch_size_tuple):
    import gc
    all_patches = []
    all_labels = []
    for item in batch:
        patches, label = item
        if patches and label != -1:
            for patch in patches:
                all_patches.append(patch)
                all_labels.append(label)
    if len(all_patches) > 100: gc.collect()
    if not all_patches:
        patch_h, patch_w = patch_size_tuple
        return torch.empty((0, 1, patch_h, patch_w), dtype=torch.float), torch.empty((0,), dtype=torch.long)

    max_chunk_size = 64
    num_patches = len(all_patches)
    patches_tensor_list = []
    for i in range(0, num_patches, max_chunk_size):
        chunk = all_patches[i:i+max_chunk_size]
        chunk_np = np.stack(chunk)
        chunk_tensor = torch.from_numpy(chunk_np).float() / 255.0
        chunk_tensor = chunk_tensor.unsqueeze(1)
        patches_tensor_list.append(chunk_tensor)
        del chunk, chunk_np
    patches_tensor = torch.cat(patches_tensor_list, dim=0)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    del patches_tensor_list, all_patches, all_labels
    gc.collect()
    return patches_tensor, labels_tensor

def create_optimized_dataloaders(combined_dataset, batch_size=256, num_workers=2, patch_size=(105, 105), train_ratio=0.8, val_ratio=0.1):
    train_size = int(train_ratio * len(combined_dataset))
    val_size = int(val_ratio * len(combined_dataset))
    test_size = len(combined_dataset) - train_size - val_size
    train_subset, val_subset, test_subset = torch.utils.data.random_split(combined_dataset, [train_size, val_size, test_size])
    collate_fn = partial(memory_efficient_patch_collate_fn, patch_size_tuple=patch_size)
    
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=True, persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None, drop_last=False
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=True, persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader( # Kept for completeness, though not strictly used in the provided training snippet
        test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=True, persistent_workers=True if num_workers > 0 else False
    )
    return train_loader, val_loader, test_loader

class SCAE(nn.Module):
    def __init__(self, normalization_type="batch_norm", use_dropout=False, dropout_prob=0.3, activation="leaky_relu"):
        super(SCAE, self).__init__()
        def norm_layer(num_features):
            if normalization_type == "batch_norm": return nn.BatchNorm2d(num_features)
            elif normalization_type == "group_norm": return nn.GroupNorm(num_groups=8, num_channels=num_features)
            elif normalization_type == "layer_norm": return nn.LayerNorm([num_features, 26, 26])
            else: return nn.Identity()
        def activation_layer():
            return nn.LeakyReLU(inplace=True) if activation == "leaky_relu" else nn.ReLU(inplace=True)
        def dropout_layer():
            return nn.Dropout2d(dropout_prob) if use_dropout else nn.Identity()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=1, padding=5), norm_layer(64), activation_layer(), dropout_layer(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), norm_layer(128), activation_layer(), dropout_layer(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), norm_layer(64), activation_layer(), dropout_layer(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0, output_padding=1), norm_layer(32), activation_layer(), dropout_layer(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def validate_memory_efficient(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    valid_batches = 0
    with torch.no_grad():
        pbar = tqdm(val_loader, leave=False)
        pbar.set_description("Validating")
        for patches, _ in pbar:
            if patches.numel() == 0: continue
            patches = patches.to(device, non_blocking=True)
            with amp.autocast():
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
    return float('inf')

def train_memory_efficient_model(model, train_loader, val_loader=None, num_epochs=5, learning_rate=0.0001, checkpoint_dir="/kaggle/working/"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    model = model.to(device)
    scaler = amp.GradScaler()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        model.train()
        running_loss = 0.0
        valid_batches = 0
        pbar = tqdm(train_loader)
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, (patches, _) in enumerate(pbar):
            if patches.numel() == 0: continue
            patches = patches.to(device, non_blocking=True)
            with amp.autocast():
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
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        if valid_batches > 0: train_loss = running_loss / valid_batches
        else: print(f"Epoch {epoch+1}/{num_epochs}, No valid batches!"); continue
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}")
        
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': train_loss,},
                   f"{checkpoint_dir}/model_epoch_{epoch+1}.pt")

        if val_loader:
            val_loss = validate_memory_efficient(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pt")
                print(f"New best model saved with val_loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= 3: print("Early stopping triggered!"); break
    return model

# Main script execution
if __name__ == '__main__':
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Define paths - replace with your actual paths
    jpeg_dir = "/kaggle/input/deepfont-unlab/scrape-wtf-new/scrape-wtf-new"  # Example path
    bcf_file = "/kaggle/input/deepfont-unlab/syn_train/VFR_syn_train_extracted.bcf"  # Example path
    label_file = "/kaggle/input/deepfont-unlab/syn_train/VFR_syn_train_extracted.label" # Example path
    checkpoint_dir_path = "/kaggle/working/checkpoints" # Example path

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    combined_dataset = CombinedImageDataset(
        jpeg_dir=jpeg_dir,
        bcf_file=bcf_file,
        label_file=label_file,
        num_patch=1,
        patch_size=(105, 105)
    )

    train_loader, val_loader, _ = create_optimized_dataloaders( # _ for test_loader as it's not used in training
        combined_dataset,
        batch_size=128,
        num_workers=2,
        patch_size=(105, 105)
    )

    model = SCAE()
    trained_model = train_memory_efficient_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=3, # Adjust as needed
        learning_rate=0.0001,
        checkpoint_dir=checkpoint_dir_path
    )
    print("Training finished.")
    # Optionally save the final model
    # torch.save(trained_model.state_dict(), f"{checkpoint_dir_path}/final_trained_model.pt")