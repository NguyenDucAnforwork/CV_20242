import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from PIL import Image
from io import BytesIO
import os
import random
import math 

def extract_patches(image_array, num_patch=3, patch_size=(105, 105)):
    """
    Extracts a specified number of random patches from a single image array.
    Handles both grayscale (2D) and color (3D) images.

    Args:
        image_array (np.ndarray): The input image (Height, Width) or (Height, Width, Channels).
        num_patch (int): The number of patches to extract.
        patch_size (tuple): The (height, width) of the patches.

    Returns:
        list[np.ndarray]: A list containing the extracted patch arrays.
                          Patches will be 2D (H, W) if input is grayscale.
                          Returns an empty list if image is smaller than patch size.
    """
    patches = []
    if image_array.ndim == 2: # Grayscale
        h, w = image_array.shape
        is_grayscale = True
    elif image_array.ndim == 3: # Color
        h, w, _ = image_array.shape
        is_grayscale = False
    else:
        print(f"Warning: Unexpected image array dimension: {image_array.ndim}. Skipping patch extraction.")
        return []

    patch_h, patch_w = patch_size

    # Check if image is large enough for at least one patch
    if h < patch_h or w < patch_w:
        # print(f"Warning: Image shape ({h}, {w}) is smaller than patch size ({patch_h}, {patch_w}). Skipping patch extraction for this image.")
        return [] # Return empty list if image is too small

    for _ in range(num_patch):
        # Ensure random coordinates are within valid bounds
        x = np.random.randint(0, w - patch_w + 1)
        y = np.random.randint(0, h - patch_h + 1)
        if is_grayscale:
            patch = image_array[y:y + patch_h, x:x + patch_w] # Shape: (patch_h, patch_w)
        else:
             patch = image_array[y:y + patch_h, x:x + patch_w, :] # Shape: (patch_h, patch_w, C) - Kept for generality but not used in this specific request
        patches.append(patch)
    return patches

# Custom Dataset for lazy-loading from BCF
class BCFImagePatchDataset(Dataset):
    """
    PyTorch Dataset for loading images from a custom BCF file format lazily
    and extracting patches on the fly. Loads images as grayscale.
    """
    def __init__(self, bcf_file, label_file, num_patch=3, patch_size=(105, 105)):
        """
        Initializes the dataset by reading metadata but not image data.

        Args:
            bcf_file (str): Path to the BCF file.
            label_file (str): Path to the label file.
            num_patch (int): Number of patches to extract per image.
            patch_size (tuple): (height, width) of patches.
        """
        self.bcf_file = bcf_file
        self.label_file = label_file
        self.num_patch = num_patch
        self.patch_size = patch_size # Store patch_size for use in collate_fn reference

        self.labels = None
        self.num_images = 0
        self.image_sizes = None
        self.image_offsets = None
        self.data_start_offset = 0 # Byte offset in BCF where actual image data begins

        self._read_metadata()

    def _read_metadata(self):
        """Reads labels and image size/offset information from the files."""
        try:
            # Read label file
            with open(self.label_file, 'rb') as f:
                self.labels = np.frombuffer(f.read(), dtype=np.uint32)
                print(f"Read {len(self.labels)} labels.")

            # Read BCF header
            with open(self.bcf_file, 'rb') as f:
                self.num_images = np.frombuffer(f.read(8), dtype=np.int64)[0]
                print(f"BCF header indicates {self.num_images} images.")

                # Check for consistency
                if len(self.labels) != self.num_images:
                    raise ValueError(f"Mismatch between number of labels ({len(self.labels)}) and images in BCF header ({self.num_images}).")

                # Read all image sizes
                sizes_bytes = f.read(self.num_images * 8)
                self.image_sizes = np.frombuffer(sizes_bytes, dtype=np.int64)
                print(f"Read {len(self.image_sizes)} image sizes.")

                # Calculate the starting offset of the actual image data blob
                self.data_start_offset = 8 + self.num_images * 8 # 8 bytes for num_images + 8 bytes per size

                # Calculate cumulative offsets for seeking
                # Offset[i] is the starting byte of image i relative to data_start_offset
                self.image_offsets = np.zeros(self.num_images + 1, dtype=np.int64)
                np.cumsum(self.image_sizes, out=self.image_offsets[1:])
                print("Calculated image offsets.")

        except FileNotFoundError as e:
            print(f"Error: File not found - {e}")
            raise
        except Exception as e:
            print(f"Error reading metadata: {e}")
            raise

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return self.num_images

    def __getitem__(self, idx):
        """
        Loads one image as grayscale, extracts patches, and returns patches with the label.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            tuple: (list[np.ndarray], int): A tuple containing:
                     - A list of NumPy arrays, each representing a patch (H, W).
                     - The integer label for the image.
               Returns ([], -1) if image reading or patch extraction fails.
        """
        if idx >= self.num_images or idx < 0:
            raise IndexError(f"Index {idx} out of bounds for {self.num_images} images.")

        label = self.labels[idx]
        offset = self.image_offsets[idx]
        size = self.image_sizes[idx]

        try:
            # Open the BCF file, seek, read only the required bytes
            with open(self.bcf_file, 'rb') as f:
                f.seek(self.data_start_offset + offset)
                image_bytes = f.read(size)

            # Convert bytes to image (grayscale) and then to numpy array
            # Use 'L' for grayscale conversion
            img = Image.open(BytesIO(image_bytes)).convert('L')
            img_array = np.array(img) # Shape: (H, W)

            # Extract patches from this single grayscale image
            patches = extract_patches(img_array, self.num_patch, self.patch_size)

            return patches, label # Return list of patches and the single label

        except FileNotFoundError:
            print(f"Error: BCF file not found during __getitem__ for index {idx}.")
            return [], -1 # Indicate error
        except Exception as e:
            print(f"Error processing image index {idx}: {e}")
            return [], -1 # Indicate error


# Custom collate function for the DataLoader (updated for grayscale)
def patch_collate_fn(batch, patch_size_tuple):
    """
    Collates data from the BCFImagePatchDataset (handling grayscale).

    Takes a batch of [(patches_list_img1, label1), (patches_list_img2, label2), ...],
    flattens the patches, converts them to a tensor, adds a channel dimension,
    normalizes, and returns a single batch tensor for patches and labels.

    Args:
        batch (list): A list of tuples, where each tuple is the output
                      of BCFImagePatchDataset.__getitem__.
        patch_size_tuple (tuple): The (height, width) of patches, needed for empty tensor shape.


    Returns:
        tuple: (torch.Tensor, torch.Tensor): A tuple containing:
                 - Patches tensor (BatchSize * NumPatches, 1, Height, Width)
                 - Labels tensor (BatchSize * NumPatches)
    """
    all_patches = []
    all_labels = []
    valid_batch_items = 0

    for item in batch:
        patches, label = item
        # Ensure item is valid (e.g., image wasn't too small, no read errors)
        if patches and label != -1:
             # Only add patches if the list is not empty
            all_patches.extend(patches)
            # Repeat the label for each patch extracted from the image
            all_labels.extend([label] * len(patches))
            valid_batch_items += 1
        # else:
            # Optionally print a warning if an item was skipped
            # print(f"Skipping item in collate_fn due to previous error or no patches.")

    # If no valid patches were collected in the batch (e.g., all images too small)
    if not all_patches:
        # Return empty tensors of appropriate type but 0 size in the batch dimension
        # Shape for grayscale: (0, 1, H, W)
        patch_h, patch_w = patch_size_tuple
        return torch.empty((0, 1, patch_h, patch_w), dtype=torch.float), torch.empty((0,), dtype=torch.long)

    # Convert list of NumPy arrays (each H, W) to a single NumPy array
    patches_np = np.array(all_patches) # Shape: (TotalPatches, H, W)

    # Convert to PyTorch tensor, normalize
    patches_tensor = torch.tensor(patches_np).float() / 255.0 # Shape: (TotalPatches, H, W)

    # Add channel dimension: (TotalPatches, H, W) -> (TotalPatches, 1, H, W)
    patches_tensor = patches_tensor.unsqueeze(1)

    # Convert labels to PyTorch tensor
    labels_tensor = torch.tensor(all_labels, dtype=torch.long) # Use long for classification labels

    # print(f"Collate - Input Batch Size: {len(batch)}, Valid Items: {valid_batch_items}, Output Patches Shape: {patches_tensor.shape}, Output Labels Shape: {labels_tensor.shape}")

    return patches_tensor, labels_tensor

# Example usage:
bcf_train = '/kaggle/input/adobe-visual-font-recognition/train.bcf'
bcf_val = '/kaggle/input/adobe-visual-font-recognition/val.bcf'
bcf_test = '/kaggle/input/adobe-visual-font-recognition/test.bcf'

label_train = '/kaggle/input/adobe-visual-font-recognition/train.label'
label_val = '/kaggle/input/adobe-visual-font-recognition/val.label'
label_test = '/kaggle/input/adobe-visual-font-recognition/test.label'

BATCH_SIZE = 1024 # Adjust as needed for your GPU memory
NUM_PATCHES_PER_IMAGE = 1
PATCH_SIZE = (105, 105) # Define patch size tuple
NUM_WORKERS = 4 # Adjust based on your CPU cores, helps speed up loading

# 1. Create the full dataset instance
try:
    train_dataset = BCFImagePatchDataset(
        bcf_file=bcf_train,
        label_file=label_train,
        num_patch=NUM_PATCHES_PER_IMAGE,
        patch_size=PATCH_SIZE # Pass patch_size to dataset
    )

    val_dataset = BCFImagePatchDataset(
        bcf_file=bcf_val,
        label_file=label_val,
        num_patch=NUM_PATCHES_PER_IMAGE,
        patch_size=PATCH_SIZE # Pass patch_size to dataset
    )

    test_dataset = BCFImagePatchDataset(
        bcf_file=bcf_test,
        label_file=label_test,
        num_patch=NUM_PATCHES_PER_IMAGE,
        patch_size=PATCH_SIZE # Pass patch_size to dataset
    )

    # 2. Create indices for splitting
    # Ensure labels were loaded before stratifying
    if train_dataset.labels is None:
         raise ValueError("Labels could not be loaded. Cannot stratify split.")

    # 4. Create DataLoaders using the custom collate function
    # We need to pass the PATCH_SIZE to the collate function. functools.partial is good for this.
    from functools import partial
    collate_wrapper = partial(patch_collate_fn, patch_size_tuple=PATCH_SIZE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_wrapper, # Use the wrapper
        pin_memory=True # Set to True if using GPU for faster data transfer
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_wrapper, # Use the wrapper
        pin_memory=True # Set to True if using GPU for faster data transfer
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_wrapper, # Use the wrapper
        pin_memory=True
    )

    # 5. Example loop through the train loader
    print("\nTesting DataLoader...")
    num_batches_to_test = 5
    for i, (batch_patches, batch_labels) in enumerate(train_loader):
        if batch_patches.numel() == 0: # Check if the batch is empty
             print(f"Batch {i+1}: Skipped (likely due to all images being too small or read errors)")
             continue

        print(f"Batch {i+1}: Patches shape: {batch_patches.shape}, Labels shape: {batch_labels.shape}")
        # Example: Check channel dimension is 1
        if batch_patches.shape[1] != 1:
             print(f"Error: Unexpected channel dimension: {batch_patches.shape[1]}")
        # print(f"Batch {i+1}: Labels: {batch_labels}") # Optional: print labels

        # --- Your training code would go here ---
        # model(batch_patches) # Ensure your model expects input shape (B, 1, H, W)
        # loss = criterion(outputs, batch_labels)
        # ...
        # ----------------------------------------

        if i >= num_batches_to_test - 1:
            break

    print("\nDataLoader setup complete and test loop finished.")

except Exception as e:
    print(f"\nAn error occurred during dataset/dataloader setup: {e}")
    import traceback
    traceback.print_exc() # Print detailed traceback
    # Depending on the error, you might want to investigate file paths,
    # file formats, or permissions.