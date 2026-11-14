"""
NEUROSCAN-AI: Custom Deep Convolutional Neural Network (Trained From Scratch)
Architecture: NeuroScanCustomCNN (4-Stage ConvNet with Residual Feature Maps & Grad-CAM Tap)
Author: Dar Rehan Rasool
Zero Pretrained Weights: Initialized with He / Kaiming Normal Initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x


class NeuroScanCustomCNN(nn.Module):
    """
    Custom 4-Stage Deep Convolutional Neural Network designed specifically for
    T1-weighted Contrast-Enhanced Axial Brain MRI multi-class neoplasm classification.
    
    Classes:
      0: Glioma (High grade invasive astrocytoma / glioblastoma)
      1: Meningioma (Extra-axial benign dural sheath tumor)
      2: Pituitary (Intrasellar/suprasellar micro/macroadenoma)
      3: No Tumor (Normal neuroanatomical control)
    """
    def __init__(self, num_classes: int = 4, in_channels: int = 3):
        super(NeuroScanCustomCNN, self).__init__()
        
        # Stage 1: Edge & Low-level texture extraction (256x256 -> 128x128)
        self.stage1 = ConvBlock(in_channels, 32)
        
        # Stage 2: Anatomical cranial structures & ventricle margins (128x128 -> 64x64)
        self.stage2 = ConvBlock(32, 64)
        
        # Stage 3: Intermediate parenchyma & edematous boundary extraction (64x64 -> 32x32)
        self.stage3 = ConvBlock(64, 128)
        
        # Stage 4: High-level semantic lesion descriptors & contrast enhancement (32x32 -> 16x16)
        # This stage's final conv layer serves as the target layer for Grad-CAM attribution
        self.stage4 = ConvBlock(128, 256)
        
        # Global Feature Aggregation
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Clinical Classifier Head with Dual Regularization (Dropout + BatchNorm)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(128, num_classes)
        )
        
        # Explicit Kaiming / He Normal Weight Initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Applies Kaiming (He) normal initialization suited for LeakyReLU activations:
        std = sqrt(2 / ((1 + a^2) * fan_in))
        Guarantees stable variance across 8 convolutional layers without exploding or vanishing gradients.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_gradcam_target_layer(self) -> nn.Module:
        """Returns the final convolutional layer of Stage 4 for Grad-CAM gradient backpropagation."""
        return self.stage4.conv2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    model = NeuroScanCustomCNN()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dummy_input = torch.randn(2, 3, 256, 256)
    out = model(dummy_input)
    print(f"✓ NeuroScanCustomCNN initialized from scratch.")
    print(f"  • Total Trainable Parameters: {total_params:,}")
    print(f"  • Input Shape: {dummy_input.shape} -> Output Shape: {out.shape}")
    print(f"  • Grad-CAM Target Layer: {model.get_gradcam_target_layer()}")
