import torch
import torch.nn as nn
"""
First we train the SCAE model on the patches.
It contains two conv and 2 deconv layers.
- conv1: 60x60, 64 filters, stride 1, padding 1 (on all four directions up down left right)
- pool1: stride 2, kernel size 2, padding 0 (on all four directions up down left right)
- conv2: 3x3, 128 filters, stride 1, padding 1 (on all four directions up down left right)
- deconv1: (from 24x24x128 to 24x24x64), 64 filters, stride 1, padding 1, kernel size 3 (on all four directions up down left right)
- unpool1: (from 24x24x64 to 48x48x64), 64 filters, stride 2, kernel size 2, padding 0 (on all four directions up down left right)
- deconv2: (from 48x48x64 to 105x105x3), 3 filters, kernel size 60, stride 1, padding 1, dilation 0, output_padding 0 
"""

class SCAE(nn.Module):
    def __init__(self):
        super(SCAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=60, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(64, 1, kernel_size=(60, 60), stride=1, padding=(1, 1), output_padding=(0, 0)),
            nn.Sigmoid() # not sure about this
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, patch):
        # print(patch.shape)
        for layer in self.encoder:
            patch = layer(patch)
            # print(patch.shape)
        for layer in self.decoder:
            patch = layer(patch)
            # print(patch.shape)
        return patch

class ConvolutionUnsupervised(nn.Module):
    def __init__(self, pretrained_scae, compressed=False):
        super().__init__()
        self.pretrained_scae = pretrained_scae
        self.encoder = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(147456, 4096),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(4096, 2383),
            nn.Softmax(dim=1)
        )
        if compressed:
            for name, layer in self.encoder.named_children():
                if isinstance(layer, nn.Linear):
                    # Compress the linear layer to rank 100
                    compressed_layer1, compressed_layer2 = svd_compress_linear_layer(layer, rank=100)
                    # Replace the original layer with the compressed layers
                    self.encoder[int(name)] = compressed_layer1
                    self.encoder.add_module(f"compressed_{name}", compressed_layer2)
                    
    def forward(self, patches):
        # Get encoded representations from pretrained SCAE encoder
        encoded = self.pretrained_scae.encoder(patches)
        
        # Pass through additional encoder layers
        features = self.encoder(encoded)
        
        # Flatten the features
        flattened = features.view(features.size(0), -1)
        
        # Pass through fully connected layers
        output = self.fully_connected(flattened)
        
        return output

def svd_compress_linear_layer(linear_layer, rank):
    """
    Compress a Linear layer using truncated SVD.
    Returns two Linear layers that approximate the original.
    """
    weight = linear_layer.weight.data  # shape: (out_features, in_features)
    out_features, in_features = weight.shape

    # Compute SVD
    U, S, V = torch.svd(weight)

    # Truncate to rank
    U_k = U[:, :rank]          # (out_features, rank)
    S_k = torch.diag(S[:rank]) # (rank, rank)
    V_k = V[:, :rank]          # (in_features, rank)

    # First linear layer: from input features to rank
    linear1 = nn.Linear(in_features, rank, bias=False)
    # Second linear layer: from rank to output features
    linear2 = nn.Linear(rank, out_features, bias=True)

    # Initialize weights
    linear1.weight.data = V_k.t().contiguous()
    linear2.weight.data = (U_k @ S_k).contiguous()
    if linear_layer.bias is not None:
        linear2.bias.data = linear_layer.bias.data.clone()
    else:
        linear2.bias.data.zero_()

    return linear1, linear2