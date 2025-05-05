import torch

def calculate_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Calculates Intersection over Union between sets of boxes.
        This is a key metric for determining how well predicted regions
        match with actual table locations.
        
        Args:
            boxes1: Tensor of shape [N, 4]
            boxes2: Tensor of shape [M, 4]
            
        Returns:
            ious: Tensor of shape [N, M] containing IoU values
        """
        # Calculate intersection coordinates
        x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # [N,M]
        y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
        
        # Calculate intersection area
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate union area
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2 - intersection
        
        return intersection / (union + 1e-5)
    
    
def convert_deltas_to_rois(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """
        Convert RPN predicted deltas into actual region proposals with proper batch handling.
        
        Args:
            anchors: Base anchor boxes of shape [num_anchors, 4] (x1, y1, x2, y2)
            deltas: Predicted adjustments of shape [batch_size, num_anchors, 4] or [num_anchors, 4]
                
        Returns:
            Tensor of ROIs:
                - If batched input: [batch_size, num_anchors, 4]
                - If single input: [num_anchors, 4]
        """
        # Determine if we're processing batches
        is_batched = len(deltas.shape) > 2
        
        if is_batched:
            batch_size = deltas.shape[0]
            num_anchors = deltas.shape[1]
            
            # Initialize tensor to store all batch results
            device = anchors.device
            rois = torch.zeros(batch_size, num_anchors, 4, device=device)
            
            # Process each batch separately
            for batch_idx in range(batch_size):
                # Get the deltas for this batch
                batch_deltas = deltas[batch_idx]  # Shape: [num_anchors, 4]
                
                # Calculate anchor properties for this batch
                widths = anchors[:, 2] - anchors[:, 0]
                heights = anchors[:, 3] - anchors[:, 1]
                ctr_x = anchors[:, 0] + 0.5 * widths
                ctr_y = anchors[:, 1] + 0.5 * heights

                # Extract delta components for this batch
                dx = batch_deltas[:, 0]
                dy = batch_deltas[:, 1]
                dw = batch_deltas[:, 2]
                dh = batch_deltas[:, 3]

                # Apply transformations for this batch
                pred_ctr_x = dx * widths + ctr_x
                pred_ctr_y = dy * heights + ctr_y
                pred_w = torch.exp(dw) * widths
                pred_h = torch.exp(dh) * heights

                # Convert to corner coordinates for this batch
                x1 = pred_ctr_x - 0.5 * pred_w
                y1 = pred_ctr_y - 0.5 * pred_h
                x2 = pred_ctr_x + 0.5 * pred_w
                y2 = pred_ctr_y + 0.5 * pred_h

                # Stack coordinates into ROIs for this batch
                batch_rois = torch.stack([x1, y1, x2, y2], dim=1)

                # Clip to valid coordinate ranges if needed
                if hasattr(self, 'feature_map_size'):
                    h, w = self.feature_map_size
                    batch_rois = torch.stack([
                        batch_rois[:, 0].clamp(0, w),
                        batch_rois[:, 1].clamp(0, h),
                        batch_rois[:, 2].clamp(0, w),
                        batch_rois[:, 3].clamp(0, h)
                    ], dim=1)
                
                # Store the results for this batch
                rois[batch_idx] = batch_rois
                
            return rois
            
        else:
            # Single batch case - process directly
            widths = anchors[:, 2] - anchors[:, 0]
            heights = anchors[:, 3] - anchors[:, 1]
            ctr_x = anchors[:, 0] + 0.5 * widths
            ctr_y = anchors[:, 1] + 0.5 * heights

            dx = deltas[:, 0]
            dy = deltas[:, 1]
            dw = deltas[:, 2]
            dh = deltas[:, 3]

            pred_ctr_x = dx * widths + ctr_x
            pred_ctr_y = dy * heights + ctr_y
            pred_w = torch.exp(dw) * widths
            pred_h = torch.exp(dh) * heights

            x1 = pred_ctr_x - 0.5 * pred_w
            y1 = pred_ctr_y - 0.5 * pred_h
            x2 = pred_ctr_x + 0.5 * pred_w
            y2 = pred_ctr_y + 0.5 * pred_h

            rois = torch.stack([x1, y1, x2, y2], dim=1)

            if hasattr(self, 'feature_map_size'):
                h, w = self.feature_map_size
                rois = torch.stack([
                    rois[:, 0].clamp(0, w),
                    rois[:, 1].clamp(0, h),
                    rois[:, 2].clamp(0, w),
                    rois[:, 3].clamp(0, h)
                ], dim=1)

            return rois