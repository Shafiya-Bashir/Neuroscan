"""
NEUROSCAN-AI: PyTorch Autograd Grad-CAM (Gradient-Weighted Class Activation Mapping)
Author: Dar Rehan Rasool
Extracts activation gradients from convolutional layers to produce interpretability heatmaps.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.cm as cm

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register forward and backward hooks to capture intermediate feature maps and gradients
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
        
    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Executes forward & backward pass to compute Grad-CAM activation heatmap.
        """
        self.model.eval()
        self.model.zero_grad()
        
        # 1. Forward Pass
        logits = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
            
        # 2. Backward Pass for target class score
        score = logits[0, target_class]
        score.backward(retain_graph=True)
        
        # 3. Global Average Pooling on gradients -> alpha weights [C]
        alpha = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # 4. Weighted combination of activation maps
        cam = torch.sum(alpha * self.activations, dim=1, keepdim=True)
        
        # 5. Apply ReLU: only positive feature influence matters for class c
        cam = F.relu(cam)
        
        # 6. Interpolate / Resize to original input spatial dimensions
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False
        )
        
        # 7. Min-Max Normalization to [0, 1]
        cam_np = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)
            
        return cam_np

    def overlay_heatmap(self, raw_img: Image.Image, cam: np.ndarray, save_path: str = None, colormap: str = "jet", alpha: float = 0.55) -> Image.Image:
        """Blends the Grad-CAM activation heatmap over the original MRI scan."""
        import matplotlib
        cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize(raw_img.size, Image.Resampling.BILINEAR)
        cam_norm = np.array(cam_resized) / 255.0
        
        cmap = matplotlib.colormaps[colormap]
        heatmap_colored = (cmap(cam_norm)[:, :, :3] * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(heatmap_colored)
        
        overlay = Image.blend(raw_img.convert("RGB"), heatmap_img, alpha=alpha)
        if save_path:
            overlay.save(save_path)
        return overlay
