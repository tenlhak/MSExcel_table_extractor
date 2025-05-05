
    
import torch
import torch.nn as nn

class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels, anchor_scales=[4, 8, 16, 32, 64, 128], anchor_ratios=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0]):
        """
        The Region Proposal Network suggests areas that might contain tables.
        
        Args:
            in_channels: Number of input feature channels
            anchor_scales: Different sizes of anchors to use
            anchor_ratios: Different height/width ratios for anchors
        """
        super().__init__()
        
        # We'll generate anchors dynamically based on the feature map
        # These parameters are kept for initializing the classification and regression heads
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        
        # The RPN first processes features through a 3x3 conv layer
        # This helps it look at local patterns that might indicate table boundaries
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Number of anchors per location
        num_anchors = len(anchor_scales) * len(anchor_ratios)
        
        # Two parallel heads:
        # 1. Classification head: Predicts if an anchor contains a table
        self.classification_head = nn.Conv2d(
            in_channels, 
            num_anchors * 2,  # 2 classes: table vs not-table
            kernel_size=1
        )
        
        # 2. Regression head: Predicts adjustments to anchor boxes
        self.regression_head = nn.Conv2d(
            in_channels,
            num_anchors * 4,  # 4 values: dx, dy, dw, dh
            kernel_size=1
        )

    def generate_anchors(self, feature_map_size):
        """
        Generate anchors that match the proportions of the sheet exactly
        """
        height, width = feature_map_size
        device = next(self.parameters()).device
        
        # Scale factors to create progressively smaller anchors
        scale_factors = [1.0, 0.75, 0.5, 0.25]
        
        # Generate anchor points with adaptive stride
        stride_y = max(1, height // 16)
        stride_x = max(1, width // 8)
        
        shifts_x = torch.arange(0, width, stride_x, device=device)
        shifts_y = torch.arange(0, height, stride_y, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        
        # Generate base anchors preserving H:W ratio
        anchors = []
        
        # 1. Anchors that match sheet proportions
        for factor in scale_factors:
            w = width * factor
            h = height * factor
            anchors.append(torch.tensor([
                -w/2, -h/2, w/2, h/2
            ], device=device))
        
        # 2. Anchors with half width (for column-like tables)
        for factor in scale_factors:
            w = max(1, width * factor * 0.5)
            h = height * factor
            anchors.append(torch.tensor([
                -w/2, -h/2, w/2, h/2
            ], device=device))
        
        # 3. Anchors with half height (for row-like tables)
        for factor in scale_factors:
            w = width * factor
            h = max(1, height * factor * 0.5)
            anchors.append(torch.tensor([
                -w/2, -h/2, w/2, h/2
            ], device=device))
        
        # Stack all base anchors
        base_anchors = torch.stack(anchors)
        
        # Generate shifts
        shifts = torch.stack([
            shift_x.reshape(-1),
            shift_y.reshape(-1),
            shift_x.reshape(-1),
            shift_y.reshape(-1)
        ], dim=1)
        
        # Add shifts to base anchors
        all_anchors = (base_anchors.reshape(1, -1, 4) + 
                       shifts.reshape(-1, 1, 4))
        
        return all_anchors.reshape(-1, 4)

    def forward(self, features):
        """
        Process features to generate table proposals.
        
        Args:
            features: Feature maps from the backbone network
            
        Returns:
            objectness_scores: Probability of table presence for each anchor
            bbox_deltas: Predicted adjustments to anchor boxes
            anchors: Generated anchor boxes
        """
        # Extract shared features with the common convolutional layer
        shared_features = self.feature_extractor(features)
        
        # Get predictions from both heads
        objectness_logits = self.classification_head(shared_features)
        bbox_deltas = self.regression_head(shared_features)
        
        # Get feature map dimensions
        N, _, H, W = features.shape
        
        
        # Generate anchors for this feature map size
        all_anchors = self.generate_anchors((H, W))
        
        # Get total number of anchors per position in the original RPN
        A = len(self.anchor_scales) * len(self.anchor_ratios)
        
        # Reshape predictions based on original anchor structure
        # (needed because the classification and regression heads expect a certain structure)
        objectness_logits = objectness_logits.permute(0, 2, 3, 1).reshape(N, H * W * A, 2)
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).reshape(N, H * W * A, 4)
        
        # Now we need to match the number of predictions with the number of anchors
        # from our adaptive anchor generation
        
        # Method 1: Pad the anchors to match the predictions
        # This approach keeps all predictions and adds dummy anchors to match
        num_predictions = H * W * A
        num_anchors = all_anchors.size(0)
        
        if num_anchors < num_predictions:
            # Pad anchors with copies of the last anchor
            padding_needed = num_predictions - num_anchors
            padding = all_anchors[-1].unsqueeze(0).repeat(padding_needed, 1)
            padded_anchors = torch.cat([all_anchors, padding], dim=0)
            
            # Now all tensors have compatible sizes
            return objectness_logits, bbox_deltas, padded_anchors
        
        # Method 2: If we have more anchors than predictions (unlikely)
        elif num_anchors > num_predictions:
            # Truncate anchors to match predictions
            truncated_anchors = all_anchors[:num_predictions]
            return objectness_logits, bbox_deltas, truncated_anchors
        
        # Method 3: Equal sizes (ideal case)
        else:
            return objectness_logits, bbox_deltas, all_anchors #objectness_logits: [N, (H * W * A), 2], bbox_deltas: [N, (H * W * A), 4], anchors: [H * W * A, 4]    