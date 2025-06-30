import torch
import monai

import torch.nn as nn

class EfficientNet(nn.Module):

    def __init__(self, in_channels: int, channels: int = 320, num_classes: int = 1, dropout: float = 0.2):
        super().__init__()

        self.model = monai.networks.nets.EfficientNetBNFeatures(
            model_name="efficientnet-b3",
            pretrained=False,
            progress=False,
            spatial_dims=3,
            in_channels=in_channels,
            num_classes=num_classes,
            # norm=('batch', {'eps': 1e-5, 'momentum': 0.1}),
            adv_prop=False,
            )
        #self.conv_head = nn.Conv3d(320, channels, kernel_size=1, stride=1, bias=False) for b0 and b1
        self.conv_head = nn.Conv3d(384, channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(channels, num_classes)

        self.silu = nn.SiLU()
        
    def forward(self, x):
        features = self.model(x)
        x = self.conv_head(features[-1])
        x = self.bn1(x)
        x = self.silu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.linear1(x)
        return x


if __name__ == "__main__":
    model = EfficientNet(2)
    x = torch.randn(3, 2, 224, 224, 224)
    out = model(x)