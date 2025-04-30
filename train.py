import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from tqdm import tqdm
import yaml
import wandb
import os
from models import SCAE, ConvolutionUnsupervised

# Get model type from environment variable, with a default value
model_type = os.environ.get('MODEL_TYPE', 'scae')

# Validate model type
if model_type not in ['scae', 'convolution_unsupervised']:
    raise ValueError(f"Invalid model type: {model_type}. Must be 'scae' or 'convolution_unsupervised'")

print(f"Training model: {model_type}")

# Load configuration from the repository
config_path = 'config.yaml'  # Path relative to the repo root
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Select the appropriate model configuration
model_config = config[model_type]
wandb_config = config['wandb']

# Initialize wandb
wandb.init(project=wandb_config['project'], entity=wandb_config['entity'], config=model_config)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model based on model type
if model_type == 'scae':
    model = SCAE().to(device)
else:
    # For ConvolutionUnsupervised, we need a pre-trained SCAE
    pretrained_scae = SCAE().to(device)
    # Load pre-trained weights from the repo
    pretrained_path = 'pretrained_scae.pth'  # Path relative to the repo root
    
    pretrained_scae.load_state_dict(torch.load(pretrained_path, map_location=device))
    # Freeze SCAE parameters
    for param in pretrained_scae.parameters():
        param.requires_grad = False
    model = ConvolutionUnsupervised(pretrained_scae).to(device)

# Define criterion based on config
if model_config['criterion'] == 'MSELoss':
    criterion = nn.MSELoss()
elif model_config['criterion'] == 'CrossEntropyLoss':
    criterion = nn.CrossEntropyLoss()
else:
    raise ValueError(f"Unsupported criterion: {model_config['criterion']}")

# Define optimizer based on config
if model_config['optimizer'] == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=model_config['learning_rate'])
elif model_config['optimizer'] == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=model_config['learning_rate'])
else:
    raise ValueError(f"Unsupported optimizer: {model_config['optimizer']}")

# Note: Replace this with your actual dataset and dataloader implementation
class CustomDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx], torch.zeros(1)  # Dummy label for autoencoder

# Placeholder for your data loading
# Replace with your actual data
dummy_data = torch.randn(100, 1, 105, 105)  # Example size for SCAE input
dummy_labels = torch.randint(0, 2383, (100,))  # For ConvolutionUnsupervised
train_dataset = CustomDataset(dummy_data, dummy_labels)
train_loader = DataLoader(train_dataset, batch_size=model_config['batch_size'], shuffle=True)

# Training loop
num_epochs = model_config['num_epochs']
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    processed_batches = 0
    
    for patches, labels in tqdm(train_loader):
        # Skip empty batches if any occurred
        if patches.numel() == 0:
            continue

        patches = patches.to(device)
        
        # Different handling for SCAE and ConvolutionUnsupervised
        if model_type == 'scae':
            # For autoencoder, input and target are the same
            targets = patches
        else:
            # For ConvolutionUnsupervised, we need the labels
            labels = labels.to(device)
            targets = labels

        optimizer.zero_grad()
        outputs = model(patches)
        
        # Calculate loss based on model type
        if model_type == 'scae':
            loss = criterion(outputs, targets)
        else:
            loss = criterion(outputs, targets)
            
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        processed_batches += 1
    
    # Calculate average loss for the epoch
    if processed_batches > 0:
        avg_loss = running_loss / processed_batches
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        # Log metrics to wandb
        wandb.log({"loss": avg_loss, "epoch": epoch + 1})
    else:
        print(f"Epoch [{epoch+1}/{num_epochs}], No batches processed.")

# Save the model to the current directory
model_save_path = f"{model_type}_model.pth"
torch.save(model.state_dict(), model_save_path)
wandb.save(model_save_path)

# Close wandb run
wandb.finish()
