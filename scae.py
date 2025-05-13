import torch
import torch.nn as nn
class SCAE(nn.Module):
    def __init__(self, normalization_type="batch_norm", use_dropout=False, dropout_prob=0.3, activation="relu"):
        super(SCAE, self).__init__()
        def norm_layer(num_features):
            if normalization_type == "batch_norm": return nn.BatchNorm2d(num_features)
            elif normalization_type == "group_norm": return nn.GroupNorm(num_groups=8, num_channels=num_features)
            elif normalization_type == "layer_norm": return nn.LayerNorm([num_features, 48, 48])
            else: return nn.Identity()
        def activation_layer():
            return nn.LeakyReLU(inplace=True) if activation == "leaky_relu" else nn.ReLU(inplace=True)
        def dropout_layer():
            return nn.Dropout2d(dropout_prob) if use_dropout else nn.Identity()

        # Encoder: conv1 → pool1 → conv2
        self.conv1 = nn.Conv2d(1, 64, kernel_size=11, stride=2, padding=0)  # 105x105 → 48x48
        self.norm1 = norm_layer(64)
        self.act1 = activation_layer()
        self.drop1 = dropout_layer()
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)                # 48x48 → 24x24

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # 24x24 → 24x24
        self.norm2 = norm_layer(128)
        self.act2 = activation_layer()
        self.drop2 = dropout_layer()

        # Decoder: deconv1 → unpool1 → deconv2
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1) # 24x24 → 24x24
        self.norm3 = norm_layer(64)
        self.act3 = activation_layer()
        self.drop3 = dropout_layer()
        self.unpool1 = nn.MaxUnpool2d(2, 2)                                           # 24x24 → 48x48

        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=11, stride=2, padding=0) # 48x48 → 105x105
        # No normalization/activation after last layer for output
        self.final_act = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x, indices = self.pool1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.drop2(x)
        # Decoder
        x = self.deconv1(x)
        x = self.norm3(x)
        x = self.act3(x)
        x = self.drop3(x)
        x = self.unpool1(x, indices, output_size=torch.Size([x.size(0), x.size(1), 48, 48]))
        x = self.deconv2(x)
        x = self.final_act(x)
        return x
    
if __name__ == "__main__":
    test_tensor = torch.randn(4, 1, 105, 105).cuda() if torch.cuda.is_available() else torch.randn(1, 1, 105, 105)
    model = SCAE().cuda() if torch.cuda.is_available() else SCAE()
    with torch.no_grad():
        output = model(test_tensor)
    print(f"Output shape: {output.shape}")