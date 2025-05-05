
'''Table Detection head'''

import torch
import torch.nn as nn


class TableDetectionHead(nn.Module):
    def __init__(self, in_channels, roi_size=7):
        """
        The final detection head that processes each region of interest.
        This component is responsible for precise table detection.
        
        Args:
            in_channels: Number of input feature channels
            roi_size: Size of the ROI feature maps after alignment
        """
        super().__init__()
        
        self.roi_size = roi_size
        roi_area = roi_size * roi_size
        
        # First, we'll create a shared network for processing ROI features
        self.shared_network = nn.Sequential(
            # Flatten the ROI features
            nn.Flatten(),
            # Process through fully connected layers
            nn.Linear(in_channels * roi_area, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Now we create three specialized heads:
        
        # 1. Classification head: Is this really a table?
        self.classifier = nn.Linear(512, 2)  # 2 classes: table or not
        
        # 2. Bounding Box Regression (BBR) head: Coarse adjustment
        self.bbr_head = nn.Linear(512, 4)  # (dx, dy, dw, dh)
        
        # 3. Precise Boundary Regression (PBR) head: Fine-tuning
        self.pbr_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # Precise adjustments to boundaries
        )

    def forward(self, roi_features):
        """
        Process ROI features to make final predictions.
        
        Args:
            roi_features: Features from ROI Align operation
                Shape: (num_rois, channels, roi_size, roi_size)
        """
        # Get shared features
        shared_features = self.shared_network(roi_features)
        
        # Get predictions from each head
        class_scores = self.classifier(shared_features)
        bbr_deltas = self.bbr_head(shared_features)
        pbr_deltas = self.pbr_head(shared_features)
        
        return class_scores, bbr_deltas, pbr_deltas