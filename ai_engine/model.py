import torch
import torch.nn as nn
import torchvision.models as models

class NeuroScanDenseNet(nn.Module):
    """
    NEUROSCAN-AI Clinical Deep Learning Architecture for Brain MRI Classification.
    
    Backbone: DenseNet-121 pre-trained on ImageNet with fine-tuned dense blocks.
    Grad-CAM Target Layer: features.denseblock4.denselayer16 (final convolutional block).
    Classification Head: AdaptiveAvgPool2d -> Linear(1024, 256) -> ReLU -> Dropout(0.4) -> Linear(256, 4)
    Classes: [0: 'Glioma', 1: 'Meningioma', 2: 'Pituitary', 3: 'Healthy']
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super(NeuroScanDenseNet, self).__init__()
        
        # Load DenseNet-121 backbone
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        densenet = models.densenet121(weights=weights)
        
        # Feature extractor
        self.features = densenet.features
        
        # Target layer for Grad-CAM backpropagation
        self.target_layer = self.features.denseblock4.denselayer16
        
        # Classification head with regularization
        in_features = densenet.classifier.in_features # 1024
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through convolutional feature blocks and classification head.
        """
        features = self.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = self.pooling(out)
        out = torch.flatten(out, 1)
        logits = self.classifier(out)
        return logits
    
    def get_target_layer(self) -> nn.Module:
        """Returns the specific convolutional layer inspected by Grad-CAM."""
        return self.target_layer


CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "Healthy"]
