import torch
import torch.nn as nn
class FontClassifier(nn.Module):
    def __init__(self, normalization_type="batch_norm", use_dropout=False, dropout_prob=0.3, activation="relu", num_classes=10):
        super(FontClassifier, self).__init__()
        # Functions for common blocks
        def norm_layer(num_features, spatial_size=None):
            if normalization_type == "batch_norm":
                return nn.BatchNorm2d(num_features)
            elif normalization_type == "group_norm":
                return nn.GroupNorm(num_groups=8, num_channels=num_features)
            elif normalization_type == "layer_norm" and spatial_size is not None:
                return nn.LayerNorm([num_features, spatial_size, spatial_size])
            else:
                return nn.Identity()

        def activation_layer():
            return nn.LeakyReLU(inplace=True) if activation == "leaky_relu" else nn.ReLU(inplace=True)

        def dropout_layer():
            return nn.Dropout2d(dropout_prob) if use_dropout else nn.Identity()

        # ===== Encoder of SCAE =====
        # Input: 1 x 105 x 105
        self.conv1 = nn.Conv2d(1, 64, kernel_size=11, stride=2, padding=0)  # Out: 64 x 48 x 48 
        self.norm1 = norm_layer(64, spatial_size=48)
        self.act1 = activation_layer()
        self.drop1 = dropout_layer()
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)                # Out: 64 x 24 x 24

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # Out: 128 x 24 x 24
        self.norm2 = norm_layer(128, spatial_size=24)
        self.act2 = activation_layer()
        self.drop2 = dropout_layer()

        # ===== Additional Convolutional Layers =====
        # Conv layer 3: change channels and reduce spatial dims via pooling
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1) # Out: 256 x 24 x 24
        self.norm3 = norm_layer(256, spatial_size=24)
        self.act3 = activation_layer()
        self.pool2 = nn.MaxPool2d(2,2)                                        # Out: 256 x 12 x 12

        # Conv layer 4
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # Out: 256 x 12 x 12
        self.norm4 = norm_layer(256, spatial_size=12)
        self.act4 = activation_layer()

        # Conv layer 5
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)  # Out: 256 x 12 x 12
        self.norm5 = norm_layer(256, spatial_size=12)
        self.act5 = activation_layer()

        # ===== Dense (Fully Connected) Layers =====
        # Flatten features: 256 x 12 x 12 = 256 * 144 = 36864
        self.fc1 = nn.Linear(256 * 12 * 12, 4096)  # First dense layer
        self.fc_act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc_act2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(2048, num_classes)
        # Softmax activation
        self.final_act = nn.Softmax(dim=1)

    def forward(self, x):
        # Encoder (SCAE part)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x, indices = self.pool1(x)  # indices not used further in this network
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.drop2(x)
        # Additional conv layers block
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act3(x)
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.norm4(x)
        x = self.act4(x)
        x = self.conv5(x)
        x = self.norm5(x)
        x = self.act5(x)
        # Flatten and Dense layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc_act1(x)
        x = self.fc2(x)
        x = self.fc_act2(x)
        x = self.fc3(x)
        x = self.final_act(x)
        return x
    
if __name__ == "__main__":
    # Test the model
    model = FontClassifier(num_classes=200)
    print(model)
    # Create a random input tensor with the shape (batch_size, channels, height, width)
    input_tensor = torch.randn(8, 1, 105, 105)  # Example batch size of 8
    # Forward pass
    output = model(input_tensor)
    print("Output shape:", output.shape)  # Should be (8, num_classes)
    