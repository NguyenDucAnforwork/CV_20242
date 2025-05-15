import gc
import torch
from cv import BcfDataset, create_optimized_dataloaders, FontClassifier, train_font_classifier

if __name__ == '__main__':
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    # Define dataset paths (adjust these to your actual paths)
    bcf_file = "/kaggle/input/deepfont-unlab/syn_train/VFR_syn_train_extracted.bcf"  # BCF file path
    label_file = "/kaggle/input/deepfont-unlab/syn_train/VFR_syn_train_extracted.label"  # Label file path
    checkpoint_dir = "/kaggle/working/checkpoints"  # Directory to save checkpoints

    # Create a dataset that uses only the BCF file.
    bcf_dataset = BcfDataset(bcf_file=bcf_file, label_file=label_file)

    train_loader, val_loader, _ = create_optimized_dataloaders(
        bcf_dataset,
        batch_size=128,
        num_workers=2,
        patch_size=(105, 105)
    )

    # Instantiate the FontClassifier (adjust num_classes as needed)
    model = FontClassifier(num_classes=10).to(device)

    # Train the font classifier on the syn_train dataset
    trained_model = train_font_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=3,        # Adjust epochs as needed
        learning_rate=0.0001,
        checkpoint_dir=checkpoint_dir
    )
    
    print("Font classifier training finished.")