import os
import cv2
import math
import gc
import pickle
import random
import tempfile
import numpy as np
from PIL import Image, ImageFile
from io import BytesIO
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# --- OCR and patch extraction ---
_ocr_reader = None

def get_ocr_reader(languages=["en"]):
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(languages)
    return _ocr_reader

def extract_patches(image_array, num_patch=3, patch_size=(105,105),
                    extract_text=True, min_text_coverage=0.3, max_attempts=20):
    patch_h, patch_w = patch_size
    if image_array.ndim == 2:
        h, w = image_array.shape
        is_grayscale = True
    elif image_array.ndim == 3:
        h, w, _ = image_array.shape
        is_grayscale = False
    else:
        print(f"Unexpected image shape: {image_array.shape}")
        return []
    scale_factor = patch_h / h
    new_w = int(w * scale_factor)
    resized = cv2.resize(image_array, (new_w, patch_h), interpolation=cv2.INTER_LINEAR)
    if new_w < patch_w:
        return []
    if not extract_text:
        patches = []
        for _ in range(num_patch):
            x = np.random.randint(0, new_w - patch_w + 1)
            patch = resized[:, x:x + patch_w] if is_grayscale else resized[:, x:x + patch_w, :]
            patches.append(patch)
        return patches
    reader = get_ocr_reader()
    text_patches = []
    attempts = 0
    while len(text_patches) < num_patch and attempts < max_attempts:
        x = np.random.randint(0, new_w - patch_w + 1)
        patch = resized[:, x:x + patch_w] if is_grayscale else resized[:, x:x + patch_w, :]
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
            Image.fromarray(patch).save(tmp_path)
        try:
            ocr_results = reader.readtext(tmp_path)
            os.unlink(tmp_path)
            patch_area = patch_h * patch_w
            text_area = 0
            for bbox, text, conf in ocr_results:
                if conf < 0.5:
                    continue
                bbox = [[int(p[0]), int(p[1])] for p in bbox]
                min_x = max(0, min(p[0] for p in bbox))
                max_x = min(patch_w, max(p[0] for p in bbox))
                min_y = max(0, min(p[1] for p in bbox))
                max_y = min(patch_h, max(p[1] for p in bbox))
                if max_x > min_x and max_y > min_y:
                    text_area += (max_x - min_x) * (max_y - min_y)
            if text_area / patch_area >= min_text_coverage:
                text_patches.append(patch)
        except Exception as e:
            print(f"OCR error: {e}")
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        attempts += 1
    return text_patches

# --- Augmentation functions ---
TARGET_SIZE = (105,105)

def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)

def noise_image(img: np.ndarray, mean=0.0, std=3.0) -> np.ndarray:
    f = img.astype(np.float32)
    n = np.random.normal(mean, std, f.shape).astype(np.float32)
    return to_uint8(f + n)

def blur_image(img: np.ndarray, sigma_range=(0.5, 1.5)) -> np.ndarray:
    f = img.astype(np.float32)
    sigma = random.uniform(*sigma_range)
    if f.ndim == 2:
        blurred = cv2.GaussianBlur(f, ksize=(0,0), sigmaX=sigma, sigmaY=sigma)
    else:
        channels = cv2.split(f)
        channels = [cv2.GaussianBlur(ch, ksize=(0,0), sigmaX=sigma, sigmaY=sigma) for ch in channels]
        blurred = cv2.merge(channels)
    return np.clip(blurred, 0, 255).astype(np.uint8)

def affine_rotation(img: np.ndarray, max_deg=10) -> np.ndarray:
    h, w = img.shape[:2]
    src = np.float32([[0,0],[w-1,0],[0,h-1]])
    dx = w * 0.05; dy = h * 0.05
    dst = src + np.random.uniform([-dx,-dy],[dx,dy],src.shape).astype(np.float32)
    M = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def shading_gradient(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    start, end = random.uniform(0.6,1.4), random.uniform(0.6,1.4)
    if random.choice([True,False]):
        grad = np.linspace(start, end, w, dtype=np.float32)[None,:]
        mask = np.repeat(grad, h, axis=0)
    else:
        grad = np.linspace(start, end, h, dtype=np.float32)[:,None]
        mask = np.repeat(grad, w, axis=1)
    if img.ndim==3:
        mask = mask[:,:,None]
    shaded = img.astype(np.float32) * mask
    return to_uint8(shaded)

def variable_aspect_ratio_preprocess(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    factor = random.uniform(5/6, 7/6)
    new_w = max(1, int(w/factor))
    resized = cv2.resize(img, (new_w, h), interpolation=cv2.INTER_LINEAR)
    if new_w < w:
        pad = w - new_w
        left = pad//2; right = pad - left
        resized = np.pad(resized, ((0,0), (left,right)) if img.ndim==2 else ((0,0),(left,right),(0,0)), mode='reflect')
    else:
        x0 = (new_w - w)//2
        resized = resized[:, x0:x0+w]
    return resized

def final_resize(img: np.ndarray, size=TARGET_SIZE) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

def augmentation_pipeline(img: np.ndarray) -> np.ndarray:
    img = to_uint8(img)
    img = variable_aspect_ratio_preprocess(img)
    pool = [noise_image, blur_image, affine_rotation, shading_gradient]
    for fn in pool:
        img = fn(img)
    return final_resize(img)

# --- Dataset class ---
class ImageDataset(Dataset):
    """
    A dataset class that loads .jpeg and .bcf images,
    extracting image patches (with optional OCR based extraction).
    """
    def __init__(self, jpeg_dir, bcf_file, label_file, testing=False, num_patch=3, patch_size=(105,105),
                 extract_text=False, min_text_coverage=0.3, max_attempts=20, ocr_languages=["en"]):
        self.jpeg_dir = jpeg_dir
        self.bcf_file = bcf_file
        self.label_file = label_file
        self.testing = testing
        self.num_patch = num_patch
        self.patch_size = patch_size
        self.extract_text = extract_text
        self.min_text_coverage = min_text_coverage
        self.max_attempts = max_attempts
        self.ocr_languages = ocr_languages
        if extract_text:
            self.reader = get_ocr_reader(ocr_languages)
        self.jpeg_data = []
        self.bcf_data = []
        self._load_jpeg_data(jpeg_dir)
        self._load_bcf_data(bcf_file, label_file)

    def _load_jpeg_data(self, jpeg_dir):
        if not os.path.exists(jpeg_dir):
            print(f"Warning: JPEG directory {jpeg_dir} does not exist.")
            return
        image_filenames = [f for f in os.listdir(jpeg_dir) if f.lower().endswith(('.jpeg','.jpg'))]
        self.jpeg_data = [(os.path.join(jpeg_dir, f), 0) for f in image_filenames]
        print(f"Loaded {len(self.jpeg_data)} .jpeg images.")

    def _load_bcf_data(self, bcf_file, label_file):
        if not (os.path.exists(bcf_file) and os.path.exists(label_file)):
            print(f"Warning: BCF file {bcf_file} or label file {label_file} does not exist.")
            return
        try:
            with open(label_file, 'rb') as f:
                self.labels = np.frombuffer(f.read(), dtype=np.uint32)
                print(f"Loaded {len(self.labels)} labels from {label_file}.")
            with open(bcf_file, 'rb') as f:
                self.num_images = np.frombuffer(f.read(8), dtype=np.int64)[0]
                print(f"Loaded {self.num_images} images from {bcf_file}.")
                sizes_bytes = f.read(self.num_images * 8)
                self.image_sizes = np.frombuffer(sizes_bytes, dtype=np.int64)
                self.data_start_offset = 8 + self.num_images * 8
                self.image_offsets = np.zeros(self.num_images+1, dtype=np.int64)
                np.cumsum(self.image_sizes, out=self.image_offsets[1:])
                for idx in range(self.num_images):
                    self.bcf_data.append((idx, self.labels[idx]))
            print(f"Loaded {len(self.bcf_data)} .bcf images.")
        except Exception as e:
            print(f"Error loading .bcf data: {e}")

    def _extract_patches(self, img_array):
        return extract_patches(img_array, num_patch=self.num_patch, patch_size=self.patch_size,
                               extract_text=self.extract_text, min_text_coverage=self.min_text_coverage,
                               max_attempts=self.max_attempts)

    def _extract_patches_test(self, img_array):
        h, w = img_array.shape[:2]
        target_h, target_w = self.patch_size
        new_w = int(w * (target_h/h))
        img = cv2.resize(img_array, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
        patches = []
        for _ in range(3):
            factor = np.random.uniform(1.5, 3.5)
            sw = max(1, int(new_w / factor))
            squeezed = cv2.resize(img, (sw, target_h), interpolation=cv2.INTER_LINEAR)
            if sw < target_w:
                pad = target_w - sw
                left = pad//2; right = pad - left
                squeezed = np.pad(squeezed, ((0,0),(left,right)) if squeezed.ndim==2 else ((0,0),(left,right),(0,0)), mode='reflect')
            else:
                x0 = (sw - target_w)//2
                squeezed = squeezed[:, x0:x0+target_w]
            for _ in range(5):
                x = np.random.randint(0, 1)
                y = np.random.randint(0, 1)
                patch = squeezed[y:y+target_h, x:x+target_w]
                patches.append(patch)
        return patches

    def __len__(self):
        return len(self.jpeg_data) + len(self.bcf_data)

    def __getitem__(self, idx):
        if isinstance(idx, list):
            results = []
            labels = []
            for single_idx in idx:
                try:
                    patches, label = self.__getitem__(single_idx)
                    if patches and label != -1:
                        results.append(patches)
                        labels.append(label)
                except Exception as e:
                    print(f"Error processing index {single_idx}: {e}")
            return results, labels
        max_retries = 3
        for retry in range(max_retries):
            try:
                if idx < len(self.jpeg_data):
                    img_path, label = self.jpeg_data[idx]
                    try:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            img = Image.open(img_path)
                            img.verify()
                        img = Image.open(img_path).convert('L')
                        img_array = np.array(img)
                        patches = self._extract_patches(img_array)
                        del img, img_array
                        return patches, label
                    except Exception as e:
                        print(f"Warning: Corrupt image at {img_path}: {e}")
                        return [], -1
                else:
                    bcf_idx = idx - len(self.jpeg_data)
                    if bcf_idx >= len(self.bcf_data):
                        return [], -1
                    label = self.bcf_data[bcf_idx][1]
                    offset = self.image_offsets[bcf_idx]
                    size = self.image_sizes[bcf_idx]
                    with open(self.bcf_file, 'rb') as f:
                        f.seek(self.data_start_offset + offset)
                        image_bytes = f.read(size)
                    buffer = BytesIO(image_bytes)
                    img = Image.open(buffer)
                    img.verify()
                    buffer.seek(0)
                    img = Image.open(buffer).convert('L')
                    img_array = np.array(img)
                    if self.testing:
                        patches = self._extract_patches_test(img_array)
                    else:
                        patches = self._extract_patches(img_array)
                        patches = [augmentation_pipeline(p) for p in patches]
                    del img, img_array, buffer, image_bytes
                    return patches, label
            except Exception as e:
                print(f"Unexpected error processing idx {idx}: {e}")
            if retry < max_retries-1:
                idx = (int(idx) + 1) % len(self)
        return [], -1

# --- Collate and DataLoader helpers ---
def test_collate_fn(batch):
    images = []
    labels = []
    for patches_list, label in batch:
        patch_tensors = []
        for patch in patches_list:
            arr = patch
            if arr.ndim == 2:
                t = torch.from_numpy(arr).unsqueeze(0)
            else:
                t = torch.from_numpy(arr).permute(2,0,1)
            t = t.float().div(255.0)
            patch_tensors.append(t)
        images.append(torch.stack(patch_tensors, dim=0))
        labels.append(label)
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels

def train_collate_fn(batch):
    imgs = []
    lbls = []
    for patches, label in batch:
        for patch in patches:
            arr = patch
            if arr.ndim == 2:
                t = torch.from_numpy(arr).unsqueeze(0)
            else:
                t = torch.from_numpy(arr).permute(2,0,1)
            imgs.append(t.float().div(255.0))
            lbls.append(label)
    images = torch.stack(imgs, dim=0)
    labels = torch.tensor(lbls, dtype=torch.long)
    return images, labels

def create_optimized_dataloaders(dataset, batch_size=512, num_workers=2, val_split=0.1):
    import numpy as np
    dataset_size = len(dataset)
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    split_idx = int(np.floor(val_split * dataset_size))
    train_indices, val_indices = indices[split_idx:], indices[:split_idx]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    def safe_collate(batch):
        valid_batch = [(patches, label) for patches, label in batch if patches and label != -1]
        if not valid_batch:
            return torch.empty((0, 1, 105, 105)), torch.empty((0,), dtype=torch.long)
        all_patches = []
        all_labels = []
        for patches, label in valid_batch:
            if isinstance(patches, list) and patches:
                all_patches.extend(patches)
                all_labels.extend([label]*len(patches))
        try:
            patches_np = np.array(all_patches)
            patches_tensor = torch.tensor(patches_np, dtype=torch.float).div(255.0)
            if len(patches_tensor.shape) == 3:
                patches_tensor = patches_tensor.unsqueeze(1)
            labels_tensor = torch.tensor(all_labels, dtype=torch.long)
            return patches_tensor, labels_tensor
        except Exception as e:
            print(f"Error in collate function: {e}")
            return torch.empty((0, 1, 105, 105)), torch.empty((0,), dtype=torch.long)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=safe_collate,
                              pin_memory=False, persistent_workers=True if num_workers>0 else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=safe_collate,
                            pin_memory=False, persistent_workers=True if num_workers>0 else False)
    return train_loader, val_loader

def save_dataset(dataset, filepath: str):
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {filepath!r}")

def load_dataset(filepath: str):
    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
    print(f"Dataset loaded from {filepath!r}")
    return dataset

# --- Visualization helper ---
def visualize_simple_images_and_patches(dataset, num_images=2, seed=None):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    if seed is not None:
        random.seed(seed)
    valid_indices = []
    attempts = 0
    max_attempts = min(len(dataset)*2, 100)
    while len(valid_indices) < num_images and attempts < max_attempts:
        idx = random.randint(0, len(dataset)-1)
        if idx not in valid_indices:
            try:
                patches, label = dataset[idx]
                if patches and len(patches) > 0:
                    valid_indices.append(idx)
            except Exception as e:
                print(f"Error loading index {idx}: {e}")
            attempts += 1
    if len(valid_indices) < num_images:
        print(f"Warning: Could only find {len(valid_indices)} valid images with patches")
        if len(valid_indices)==0:
            print("No valid images found. Check your dataset.")
            return
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(valid_indices), 4, figsize=(16,5*len(valid_indices)))
    if len(valid_indices)==1:
        axes = axes.reshape(1,-1)
    for i, idx in enumerate(valid_indices):
        try:
            patches, label = dataset[idx]
            img_array = None
            source = ""
            if hasattr(dataset, 'jpeg_data') and idx < len(dataset.jpeg_data):
                img_path, _ = dataset.jpeg_data[idx]
                img = Image.open(img_path).convert('L')
                img_array = np.array(img)
                source = f"JPEG: {os.path.basename(img_path)}"
            else:
                bcf_idx = idx - len(dataset.jpeg_data)
                offset = dataset.image_offsets[bcf_idx]
                size = dataset.image_sizes[bcf_idx]
                with open(dataset.bcf_file, 'rb') as f:
                    f.seek(dataset.data_start_offset+offset)
                    image_bytes = f.read(size)
                img = Image.open(BytesIO(image_bytes)).convert('L')
                img_array = np.array(img)
                source = f"BCF idx: {bcf_idx}"
            if img_array is not None:
                axes[i,0].imshow(img_array, cmap='gray')
                axes[i,0].set_title(f"Original\nLabel: {label}\nSource: {source}")
                axes[i,0].axis('off')
            else:
                axes[i,0].text(0.5,0.5,"Image loading failed", ha='center', va='center')
                axes[i,0].axis('off')
            if patches and len(patches)>0:
                for j in range(3):
                    if j < len(patches):
                        patch = patches[j]
                        axes[i,j+1].imshow(patch, cmap='gray')
                        axes[i,j+1].set_title(f"Patch {j+1}\n{patch.shape}")
                    else:
                        axes[i,j+1].text(0.5,0.5,"No patch", ha='center', va='center')
                    axes[i,j+1].axis('off')
            else:
                for j in range(3):
                    axes[i,j+1].text(0.5,0.5,"No patches extracted", ha='center', va='center')
                    axes[i,j+1].axis('off')
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            for j in range(4):
                axes[i,j].text(0.5,0.5, f"Error: {str(e)[:50]}...", ha='center', va='center')
                axes[i,j].axis('off')
    plt.tight_layout()
    plt.show()
    return valid_indices