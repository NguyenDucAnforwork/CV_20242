import torch.nn as nn

class SCAE(nn.Module):
    def __init__(self, normalization_type="batch_norm", use_dropout=False, dropout_prob=0.3, activation="relu"):
        super(SCAE, self).__init__()
        def norm_layer(num_features):
            if normalization_type=="batch_norm":
                return nn.BatchNorm2d(num_features)
            elif normalization_type=="group_norm":
                return nn.GroupNorm(num_groups=8, num_channels=num_features)
            elif normalization_type=="layer_norm":
                return nn.LayerNorm([num_features, 12, 12])
            else:
                return nn.Identity()
        def activation_layer():
            return nn.LeakyReLU(inplace=True) if activation=="leaky_relu" else nn.ReLU(inplace=True)
        def dropout_layer():
            return nn.Dropout2d(dropout_prob) if use_dropout else nn.Identity()
        # Encoder: 105x105 -> 12x12
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=2, padding=0),
            norm_layer(64),
            activation_layer(),
            dropout_layer(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            norm_layer(128),
            activation_layer(),
            dropout_layer(),
            nn.MaxPool2d(2,2)
        )
        # Decoder: 12x12 -> 105x105
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            norm_layer(64),
            activation_layer(),
            dropout_layer(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            norm_layer(32),
            activation_layer(),
            dropout_layer(),
            nn.ConvTranspose2d(32, 1, kernel_size=14, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        if x.size(1)==3:
            x = 0.299 * x[:,0:1] + 0.587 * x[:,1:2] + 0.114 * x[:,2:3]
        for layer in self.encoder:
            x = layer(x)
        for layer in self.decoder:
            x = layer(x)
        return x

class FontClassifier(nn.Module):
    def __init__(self, pretrained_scae, num_classes=200, normalization_type="batch_norm", 
                 use_dropout=False, dropout_prob=0.3, activation="relu"):
        super(FontClassifier, self).__init__()
        self.pretrained_scae = pretrained_scae
        def norm_layer(num_features, spatial_size=None):
            if normalization_type=="batch_norm":
                return nn.BatchNorm2d(num_features)
            elif normalization_type=="group_norm":
                return nn.GroupNorm(num_groups=8, num_channels=num_features)
            elif normalization_type=="layer_norm" and spatial_size is not None:
                return nn.LayerNorm([num_features, spatial_size, spatial_size])
            else:
                return nn.Identity()
        def activation_layer():
            return nn.LeakyReLU(inplace=True) if activation=="leaky_relu" else nn.ReLU(inplace=True)
        def dropout_layer():
            return nn.Dropout2d(dropout_prob) if use_dropout else nn.Identity()
        self.cnn_head = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            norm_layer(256, 12),
            activation_layer(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            norm_layer(256, 13),
            activation_layer(),
            dropout_layer()
        )
        self.flatten = nn.Flatten()
        self.fully_connected = nn.Sequential(
            nn.Linear(256*12*12, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob if use_dropout else 0),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob if use_dropout else 0),
            nn.Linear(2048, num_classes)
        )
    def forward(self, x):
        x = self.pretrained_scae.encoder(x)
        x = self.cnn_head(x)
        x = self.flatten(x)
        x = self.fully_connected(x)
        return x