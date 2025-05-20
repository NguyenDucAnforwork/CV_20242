import torch
from dataset import create_optimized_dataloaders, ImageDataset
from models import SCAE, FontClassifier
from train_val import train_classifier, evaluate
import os

# Parameters (customize paths as needed)
jpeg_dir = "path/to/jpeg_dir"
bcf_file = "path/to/data.bcf"
label_file = "path/to/labels.label"

# Create dataset and dataloaders (for classification, assume dataset labels are provided in BCF)
dataset = ImageDataset(jpeg_dir, bcf_file, label_file, num_patch=3)
train_loader, val_loader = create_optimized_dataloaders(dataset, batch_size=128, num_workers=4, val_split=0.1)

# Load pretrained SCAE weights (assume scae is pre-trained and saved)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_scae = SCAE().to(device)
scae_ckpt = torch.load("path/to/scae_checkpoints/best_model.pt", map_location=device)
pretrained_scae.load_state_dict(scae_ckpt)
pretrained_scae.eval()

# Create FontClassifier using the pretrained SCAE encoder
classifier = FontClassifier(pretrained_scae, num_classes=200).to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

# Train the classifier
checkpoint_path = "./classifier_checkpoints/best_font_model.pt"
os.makedirs("./classifier_checkpoints", exist_ok=True)
classifier, history = train_classifier(classifier, train_loader, val_loader, optimizer, criterion,
                                       scheduler=scheduler, device=device, num_epochs=5,
                                       early_stopping_patience=7, checkpoint_path=checkpoint_path,
                                       use_amp=False)
# Optionally, evaluate on test loader if available
# top1_error, top5_error = evaluate(classifier, test_loader, device=device, use_amp=False)
# print(f"Top-1 Error: {top1_error:.2f}%, Top-5 Error: {top5_error:.2f}%")
print("FontClassifier training completed.")