import torch
from dataset import create_optimized_dataloaders, ImageDataset
from models import SCAE
from train_val import train_memory_efficient_model
import os

# Parameters (customize paths as needed)
jpeg_dir = "path/to/jpeg_dir"
bcf_file = "path/to/data.bcf"
label_file = "path/to/labels.label"

# Create dataset and dataloaders
dataset = ImageDataset(jpeg_dir, bcf_file, label_file, num_patch=3)
train_loader, val_loader = create_optimized_dataloaders(dataset, batch_size=128, num_workers=4, val_split=0.1)

# Create model and train SCAE
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SCAE()
trained_model = train_memory_efficient_model(model, train_loader, val_loader, num_epochs=5, learning_rate=0.0001, checkpoint_dir="./scae_checkpoints")

# Save final model
torch.save(trained_model.state_dict(), os.path.join("./scae_checkpoints", "final_scae.pt"))
print("SCAE training completed and model saved.")