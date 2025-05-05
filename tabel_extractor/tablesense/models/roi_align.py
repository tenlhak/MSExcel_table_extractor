'''ROI Align'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class ROIAlign(nn.Module):
    def __init__(self, output_size=7, sampling_ratio=2):
        super().__init__()
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio

    def forward(self, features, rois):
        """
        Args:
            features: [batch_size, channels, height, width]
            rois: [batch_size, num_rois_per_batch, 4]
        Returns:
            output: [batch_size * num_rois_per_batch, channels, output_size, output_size]
            Input:
                - features: [4, 64, 38, 38]    # 4 batches
                - rois: [4, 2904, 4]          # 4 batches, 2904 ROIs per batch

            Output:
                - [11616, 64, 7, 7] 
        """
        device = features.device
        batch_size = features.size(0)
        channels = features.size(1)
        num_rois_per_batch = rois.size(1)
        
        # Initialize output tensor
        output = torch.zeros(
            batch_size * num_rois_per_batch,
            channels, 
            self.output_size, 
            self.output_size, 
            device=device
        )
        
        # Process each batch separately
        for batch_idx in range(batch_size):
            batch_features = features[batch_idx].unsqueeze(0)  # [1, channels, H, W]
            batch_rois = rois[batch_idx]  # [num_rois_per_batch, 4]
            
            # Process each ROI in this batch
            for roi_idx, roi in enumerate(batch_rois):
                x1, y1, x2, y2 = roi
                
                # Calculate output index considering batch
                output_idx = batch_idx * num_rois_per_batch + roi_idx
                
                # Rest of the ROI processing remains similar...
                roi_width = max((x2 - x1).item(), 1)
                roi_height = max((y2 - y1).item(), 1)
                
                x_steps = torch.linspace(0, 1, self.output_size, device=device)
                y_steps = torch.linspace(0, 1, self.output_size, device=device)
                y_grid, x_grid = torch.meshgrid(y_steps, x_steps, indexing='ij')
                
                x_grid = x1 + x_grid * roi_width
                y_grid = y1 + y_grid * roi_height
                
                x_grid = 2 * x_grid / (features.size(3) - 1) - 1
                y_grid = 2 * y_grid / (features.size(2) - 1) - 1
                
                grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0)
                
                # Sample features for this ROI from its corresponding batch
                roi_features = F.grid_sample(
                    batch_features,
                    grid,
                    mode='bilinear',
                    align_corners=True
                )
                
                output[output_idx] = roi_features[0]
        
        return output
