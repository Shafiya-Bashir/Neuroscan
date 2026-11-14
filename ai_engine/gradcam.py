import torch
import torch.nn.functional as F
import numpy as np

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) implementation in PyTorch.
    
    Computes coarse 2D visual explanations of decisions from convolutional neural networks.
    Mathematical Formulation:
      alpha_k^c = (1 / Z) * sum_i sum_j (d y^c / d A_ij^k)
      L_GradCAM^c = ReLU( sum_k alpha_k^c * A^k )
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        
        self.activations = None
        self.gradients = None
        
        # Register PyTorch autograd hooks
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
        
    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Executes forward & backward pass to compute Grad-CAM activation heatmap.
        
        Args:
            input_tensor: PyTorch tensor of shape [1, 3, H, W]
            target_class: Index of target class (if None, uses argmax predicted class)
            
        Returns:
            heatmap: 2D numpy array [H, W] normalized between 0.0 and 1.0
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
        # Gradients: [1, C, H_feat, W_feat]
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
