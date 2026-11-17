"""
NEUROSCAN-AI: Advanced Deep Convolutional Neural Network Architecture
Specialized for High-Resolution Brain MRI Neoplastic Diagnosis & Grad-CAM Explainability.
Features:
  - 5-Stage Residual-Attention Feature Backbone
  - Squeeze-and-Excitation (SE) Channel Recalibration
  - Dual Regularization (BatchNorm2d + Spatial/Dense Dropout)
  - Native Grad-CAM Gradient Attribution Hooks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention mechanism."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y), inplace=True)
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class ResConvBlock(nn.Module):
    """Residual convolutional block with SE attention and Batch Normalization."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.act2(out + res)
        return out

class NeuroScanCustomCNN(nn.Module):
    """
    Primary 5-Stage Deep CNN with Residual SE Blocks for Clinical Brain MRI Classification.
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()

        # Stage 1: Stem (256x256 -> 128x128)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2, 2)
        )

        # Stage 2: Low-level features (128x128 -> 64x64)
        self.stage2 = nn.Sequential(
            ResConvBlock(32, 64),
            nn.MaxPool2d(2, 2)
        )

        # Stage 3: Mid-level features (64x64 -> 32x32)
        self.stage3 = nn.Sequential(
            ResConvBlock(64, 128),
            nn.MaxPool2d(2, 2)
        )

        # Stage 4: High-level neoplastic features (32x32 -> 16x16)
        self.stage4 = nn.Sequential(
            ResConvBlock(128, 256),
            nn.MaxPool2d(2, 2)
        )

        # Stage 5: Deep semantic lesion representation (16x16)
        self.stage5 = nn.Sequential(
            ResConvBlock(256, 512),
            ResConvBlock(512, 512)
        )

        # Multi-Class Diagnostic Classifier Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

        # Hooks storage for Grad-CAM
        self.gradients = None
        self.activations = None

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

    def get_gradcam_target_layer(self) -> nn.Module:
        """Returns the final convolutional block of Stage 5 for Grad-CAM attribution."""
        return self.stage5[-1]

class GradCAMExplainer:
    """Computes exact Grad-CAM feature attribution heatmaps."""
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        def forward_hook(module, input, output):
            self.activations = output.detach()

        self.target_layer.register_forward_hook(forward_hook)
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.target_layer.register_full_backward_hook(backward_hook)
        else:
            self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> torch.Tensor:
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        loss = output[0, target_class]
        loss.backward()

        # Global average pool of gradients -> channel importance weights alpha_k
        alpha_k = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        weighted_activations = torch.sum(alpha_k * self.activations, dim=1, keepdim=True)
        # Rectified linear unit (ReLU) to filter negative influences
        cam = F.relu(weighted_activations)

        # Normalize between 0 and 1
        cam_min, cam_max = torch.min(cam), torch.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        # Upsample to match input MRI resolution (256x256)
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy()
