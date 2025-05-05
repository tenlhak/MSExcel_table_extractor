import torch
import torch.nn as nn
import torch.nn.functional as F
from .rpn import RegionProposalNetwork
from .roi_align import ROIAlign
from .detection_head import TableDetectionHead
from tablesense.utils.monitoring import PerformanceMonitor

class CompleteTableDetectionSystem(nn.Module):
    def __init__(self, input_channels=20, hidden_dim=64):
        super().__init__()
        
        # Same initialization code as before...
        self.feature_processor = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        self.rpn = RegionProposalNetwork(hidden_dim)
        self.roi_align = ROIAlign(output_size=7)
        self.detection_head = TableDetectionHead(hidden_dim)
        
        self.nms_threshold = 0.7
        self.score_threshold = 0.01

    def convert_deltas_to_rois(self, anchors, deltas):
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
        
    def get_final_detections(self, rois, class_scores, bbr_deltas, pbr_deltas):
        """
        Convert network outputs into final table detections.
        
        Args:
            rois: Region proposals from RPN
            class_scores: Classification scores for each ROI
            bbr_deltas: Coarse boundary adjustments
            pbr_deltas: Precise boundary adjustments
            
        Returns:
            List of final detections [x1, y1, x2, y2, score]
        """
        # Get table probabilities
        table_probs = F.softmax(class_scores, dim=1)[:, 1]
        print(table_probs)
        print(max(table_probs))
        
        # Filter by confidence threshold
        keep = self.score_threshold
        #print(f'\n the Keep is {sum(keep)} and the threshold is {self.score_threshold}')
        rois = rois[keep]
        table_probs = table_probs[keep]
        bbr_deltas = bbr_deltas[keep]
        pbr_deltas = pbr_deltas[keep]
        
        if rois.shape[0] == 0:
            return []
        
        # Apply boundary adjustments
        refined_rois = self.convert_deltas_to_rois(rois, bbr_deltas)
        final_boxes = self.convert_deltas_to_rois(refined_rois, pbr_deltas)
        
        # Apply NMS
        keep = self.nms(final_boxes, table_probs)
        final_boxes = final_boxes[keep]
        final_scores = table_probs[keep]
        print(f'The final boxes are  {final_boxes}')
        # Convert to list of detections
        detections = torch.cat([final_boxes, final_scores.unsqueeze(1)], dim=1)
        return detections.cpu().numpy().tolist()

    def nms(self, boxes, scores):
        """
        Non-maximum suppression to remove overlapping detections.
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        _, order = scores.sort(0, descending=True)
        keep = []

        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
            i = order[0]
            keep.append(i.item())

            # Compute IoU
            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])
            
            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            ids = (ovr <= self.nms_threshold).nonzero().squeeze()
            
            if ids.numel() == 0:
                break
            order = order[ids + 1]
            
        return torch.tensor(keep)

    

    def get_rpn_labels_and_positive_anchors(self, anchors, targets):
        batch_size = targets.shape[0]
        num_anchors = anchors.shape[0]
        
        # Initialize labels tensor
        labels = torch.zeros(batch_size, num_anchors, dtype=torch.long, device=anchors.device)
        # Initialize sample weights
        sample_weights = torch.ones(batch_size, device=anchors.device)
        # Initialize valid batch mask
        valid_batch_mask = torch.zeros(batch_size, dtype=torch.bool, device=anchors.device)
        
        # Store positive anchor masks for each batch
        positive_anchors_per_batch = []
        
        for batch_idx in range(batch_size):
            target = targets[batch_idx]
            
            # Calculate IoU
            overlaps = self.calculate_iou(anchors, target.unsqueeze(0))
            max_iou = overlaps.max().item()
            print(f'The max anchor overlap is {max_iou}')
            # Determine if this batch should be included
            if max_iou > 0.3:
                valid_batch_mask[batch_idx] = True
                
                # Determine sample weight based on max IoU
                if max_iou > 0.7:
                    sample_weights[batch_idx] = 1.0
                elif max_iou > 0.5:
                    sample_weights[batch_idx] = 0.7
                else:  # Between 0.3 and 0.5
                    sample_weights[batch_idx] = 0.4
                
                # Find positive anchors
                positive_anchors = (overlaps > 0.6).squeeze(1)
                
                # If no anchors above threshold, use top 5
                if not positive_anchors.any():
                    _, top_indices = overlaps.squeeze(1).topk(k=5)
                    positive_anchors = torch.zeros_like(positive_anchors, dtype=torch.bool)
                    positive_anchors[top_indices] = True
                
                # Set labels
                labels[batch_idx][positive_anchors] = 1
            else:
                # Very low IoU batch - create empty positive anchors
                positive_anchors = torch.zeros(num_anchors, dtype=torch.bool, device=anchors.device)
                sample_weights[batch_idx] = 0.0  # Zero weight
            
            # Store positive anchors for this batch
            positive_anchors_per_batch.append(positive_anchors)
        
        # Return all the computed information
        return labels.view(-1), sample_weights, valid_batch_mask, positive_anchors_per_batch
    def get_rpn_bbox_targets(self, anchors, targets, positive_anchors_per_batch):
        """
        Compute bbox targets for RPN using pre-computed positive anchors.
        
        Args:
            anchors: Tensor of shape [N, 4] containing anchor boxes
            targets: Tensor of shape [batch_size, 4] containing table coordinates
            positive_anchors_per_batch: List of boolean tensors indicating positive anchors
            
        Returns:
            bbox_targets: Tensor of shape [batch_size * N, 4] containing adjustments
        """
        batch_size = targets.shape[0]
        num_anchors = anchors.shape[0]
        
        # Initialize targets tensor
        bbox_targets = torch.zeros(batch_size, num_anchors, 4, device=anchors.device)
        
        # Process each batch
        for batch_idx in range(batch_size):
            target = targets[batch_idx]
            positive_anchors = positive_anchors_per_batch[batch_idx]
            
            # Skip if no positive anchors
            if not positive_anchors.any():
                continue
            
            # Convert coordinates to deltas for positive anchors
            pos_anchors = anchors[positive_anchors]
            
            # Compute width and height of anchors
            anchor_widths = pos_anchors[:, 2] - pos_anchors[:, 0]
            anchor_heights = pos_anchors[:, 3] - pos_anchors[:, 1]
            anchor_ctr_x = pos_anchors[:, 0] + 0.5 * anchor_widths
            anchor_ctr_y = pos_anchors[:, 1] + 0.5 * anchor_heights
            
            # Compute target width and height
            target_width = target[2] - target[0]
            target_height = target[3] - target[1]
            target_ctr_x = target[0] + 0.5 * target_width
            target_ctr_y = target[1] + 0.5 * target_height
            
            # Compute deltas
            dx = (target_ctr_x - anchor_ctr_x) / anchor_widths
            dy = (target_ctr_y - anchor_ctr_y) / anchor_heights
            dw = torch.log(target_width / anchor_widths)
            dh = torch.log(target_height / anchor_heights)
            
            # Store targets for this batch
            bbox_targets[batch_idx][positive_anchors] = torch.stack([dx, dy, dw, dh], dim=1)
                
        return bbox_targets.view(-1, 4)


    # def get_roi_labels_and_weights(self, rois, targets):
    #     """
    #     Get labels for ROIs with weights based on IoU quality.
    #     Returns:
    #         - labels: Tensor of shape [batch_size, num_rois]
    #         - weights: Tensor of shape [batch_size, num_rois]
    #     """
    #     batch_size, num_rois, _ = rois.shape
    #     labels = torch.zeros((batch_size, num_rois), dtype=torch.long, device=rois.device)
    #     weights = torch.zeros((batch_size, num_rois), device=rois.device)
        
    #     for i, target in enumerate(targets):
    #         # Compute IoU for rois in the i-th batch against the target
    #         overlaps = self.calculate_iou(rois[i], target.unsqueeze(0)).squeeze(1)
    #         max_iou = overlaps.max().item()
    #         print(f'The max ROI overlap is {max_iou}')
    #         # Find positive ROIs
    #         positive_rois = (overlaps > 0.7).squeeze() if overlaps.dim() > 1 else overlaps > 0.7
            
    #         # If fewer than minimum required positive ROIs
    #         min_required = 3  # You can adjust this
    #         if positive_rois.sum() < min_required:
    #             # Only select additional ROIs if they exceed a minimum threshold
    #             min_acceptable_iou = 0.4  # Higher than for anchors
    #             acceptable_rois = (overlaps >= min_acceptable_iou) & (~positive_rois)
                
    #             if acceptable_rois.sum() > 0:
    #                 # Get top-k from acceptable ROIs where k is the shortfall
    #                 shortfall = min_required - positive_rois.sum()
    #                 _, top_indices = overlaps[acceptable_rois].topk(
    #                     k=min(shortfall, acceptable_rois.sum())
    #                 )
                    
    #                 # Map back to original indices
    #                 acceptable_indices = torch.where(acceptable_rois)[0]
    #                 additional_positives = acceptable_indices[top_indices]
                    
    #                 # Update positive ROIs
    #                 positive_rois[additional_positives] = True
                    
    #             # If we still don't have enough, cautiously use top ROIs regardless of IoU
    #             if positive_rois.sum() < min_required and max_iou > 0.2:
    #                 remaining = ~positive_rois
    #                 if remaining.any():
    #                     # Get indices of top remaining ROIs
    #                     shortfall = min_required - positive_rois.sum()
    #                     _, top_indices = overlaps[remaining].topk(
    #                         k=min(shortfall, remaining.sum())
    #                     )
                        
    #                     # Map back to original indices
    #                     remaining_indices = torch.where(remaining)[0]
    #                     last_resort_positives = remaining_indices[top_indices]
                        
    #                     # Update positive ROIs
    #                     positive_rois[last_resort_positives] = True
            
    #         # Assign labels
    #         labels[i][positive_rois] = 1
            
    #         # Assign weights based on IoU - higher weight for higher IoU
    #         roi_weights = torch.zeros_like(overlaps)
            
    #         # High quality ROIs get full weight
    #         high_quality = overlaps > 0.7
    #         roi_weights[high_quality] = 1.0
            
    #         # Medium quality ROIs get partial weight
    #         medium_quality = (overlaps > 0.5) & (~high_quality)
    #         roi_weights[medium_quality] = 0.7
            
    #         # Low quality ROIs get lower weight
    #         low_quality = (overlaps > 0.3) & (~high_quality) & (~medium_quality)
    #         roi_weights[low_quality] = 0.4
            
    #         # Very low quality ROIs (those selected as a last resort) get minimal weight
    #         very_low = positive_rois & (~high_quality) & (~medium_quality) & (~low_quality)
    #         roi_weights[very_low] = 0.2
            
    #         weights[i] = roi_weights
        
    #     return labels, weights
    def get_roi_labels_and_weights(self, rois, targets):
        """Vectorized implementation for ROI label and weight assignment"""
        batch_size, num_rois, _ = rois.shape
        labels = torch.zeros((batch_size, num_rois), dtype=torch.long, device=rois.device)
        weights = torch.zeros((batch_size, num_rois), device=rois.device)
        
        # Process each batch in parallel where possible
        for i in range(batch_size):
            # Calculate IoU for all ROIs in this batch against target
            overlaps = self.calculate_iou(rois[i], targets[i].unsqueeze(0)).squeeze(1)
            print(f'The max ROI overlap is {max(overlaps)}')
            # Create masks for different quality levels (vectorized)
            high_quality = overlaps > 0.7
            medium_quality = (overlaps > 0.5) & (~high_quality)
            low_quality = (overlaps > 0.3) & (~high_quality) & (~medium_quality)
            
            # Set labels for high quality ROIs
            labels[i][high_quality] = 1
            
            # For batches with insufficient high-quality ROIs
            if high_quality.sum() < 3:
                # Select additional ROIs from medium or low quality if available
                remaining_needed = 3 - high_quality.sum()
                medium_indices = torch.where(medium_quality)[0]
                
                if len(medium_indices) > 0:
                    # Select top medium quality ROIs
                    top_medium_count = min(remaining_needed, len(medium_indices))
                    _, top_medium_indices = overlaps[medium_quality].topk(top_medium_count)
                    additional_indices = medium_indices[top_medium_indices]
                    labels[i][additional_indices] = 1
                    remaining_needed -= top_medium_count
                
                if remaining_needed > 0:
                    # If still need more, select from low quality
                    low_indices = torch.where(low_quality)[0]
                    if len(low_indices) > 0:
                        top_low_count = min(remaining_needed, len(low_indices))
                        _, top_low_indices = overlaps[low_quality].topk(top_low_count)
                        additional_indices = low_indices[top_low_indices]
                        labels[i][additional_indices] = 1
            
            # Assign weights based on quality (vectorized)
            weights[i][high_quality] = 1.0
            weights[i][medium_quality & (labels[i] == 1)] = 0.7
            weights[i][low_quality & (labels[i] == 1)] = 0.4
            weights[i][(overlaps <= 0.3) & (labels[i] == 1)] = 0.2
        
        return labels, weights

    def get_roi_bbox_targets(self, rois, targets):
        """
        Now rois has shape [batch_size, num_rois, 4] and targets has shape [batch_size, 4].
        Returns:
            bbox_targets: Tensor of shape [batch_size, num_rois, 4]
        """
        batch_size, num_rois, _ = rois.shape
        bbox_targets = torch.zeros((batch_size, num_rois, 4), dtype=rois.dtype, device=rois.device)
        
        for i, target in enumerate(targets):
            # Calculate IoU for the i-th batch only.
            overlaps = self.calculate_iou(rois[i], target.unsqueeze(0))
            positive_rois = (overlaps > 0.8).squeeze(1)  # Should now be of shape [num_rois]
            
            # If no IoUs > 0.8, take top 5 anchors
            if not positive_rois.any():
                # Get indices of top 5 IoUs
                _, top_indices = overlaps.squeeze(1).topk(k=5)
                positive_rois = torch.zeros_like(positive_rois, dtype=torch.bool)
                positive_rois[top_indices] = True
                
            # Now we're guaranteed to have some positive ROIs
            pos_rois = rois[i][positive_rois]
            roi_widths = pos_rois[:, 2] - pos_rois[:, 0]
            roi_heights = pos_rois[:, 3] - pos_rois[:, 1]
            roi_ctr_x = pos_rois[:, 0] + 0.5 * roi_widths
            roi_ctr_y = pos_rois[:, 1] + 0.5 * roi_heights
            
            target_width = target[2] - target[0]
            target_height = target[3] - target[1]
            target_ctr_x = target[0] + 0.5 * target_width
            target_ctr_y = target[1] + 0.5 * target_height
            
            dx = (target_ctr_x - roi_ctr_x) / (roi_widths + 1e-5)
            dy = (target_ctr_y - roi_ctr_y) / (roi_heights + 1e-5)
            dw = torch.log(target_width / (roi_widths + 1e-5))
            dh = torch.log(target_height / (roi_heights + 1e-5))
            
            # Assign the computed targets back to the corresponding proposals for batch i.
            bbox_targets[i][positive_rois] = torch.stack([dx, dy, dw, dh], dim=1)
        
        return bbox_targets

    # def calculate_iou(self, boxes1, boxes2):
    #     """
    #     Calculates Intersection over Union between sets of boxes.
    #     This is a key metric for determining how well predicted regions
    #     match with actual table locations.
        
    #     Args:
    #         boxes1: Tensor of shape [N, 4]
    #         boxes2: Tensor of shape [M, 4]
            
    #     Returns:
    #         ious: Tensor of shape [N, M] containing IoU values
    #     """
    #     # Calculate intersection coordinates
    #     x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # [N,M]
    #     y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    #     x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    #     y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
        
    #     # Calculate intersection area
    #     intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
    #     # Calculate union area
    #     area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    #     area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    #     union = area1[:, None] + area2 - intersection
        
    #     return intersection / (union + 1e-5)
    def calculate_iou(self, boxes1, boxes2):
        """Vectorized and optimized IoU calculation"""
        # Expand dimensions for broadcasting
        b1_x1, b1_y1, b1_x2, b1_y2 = boxes1[:, 0:1], boxes1[:, 1:2], boxes1[:, 2:3], boxes1[:, 3:4]
        b2_x1, b2_y1, b2_x2, b2_y2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
        
        # Calculate intersection area
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        
        # Clamp ensures no negative width/height due to non-overlapping boxes
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Calculate union area (vectorized)
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area
        
        return inter_area / (union_area + 1e-6)  # Add epsilon to avoid division by zero
    

    def forward(self, x, targets):    
        """
        Forward pass handling both training and inference modes.
        During training, it computes all necessary losses.
        During inference, it returns the final table detections.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width] 
            containing the cell features
            targets: During training, tensor of shape [batch_size, 4] 
                    containing the true table coordinates (x1, y1, x2, y2)
        
        Returns:
            Training mode: Dictionary containing all loss components
            Inference mode: List of detected table coordinates
        """
        # First, process the input through our feature extraction backbone
        self.feature_map_size = (x.shape[2], x.shape[3])
        features = self.feature_processor(x)
        #print(f"Feature shape after processor: {features.shape}")
        
        # Get region proposals from RPN
        rpn_scores, rpn_deltas, anchors = self.rpn(features)
        #print(f"RPN outputs - scores: {rpn_scores.shape}, deltas: {rpn_deltas.shape}, anchors: {anchors.shape}")
        
        torch.cuda.empty_cache()
        
        # Convert RPN proposals to ROIs
        rois = self.convert_deltas_to_rois(anchors, rpn_deltas)
        #print(f"ROIs shape after conversion: {rois.shape}")
        
        # Process ROIs through ROI Align and Detection Head
        roi_features = self.roi_align(features, rois)
        class_scores, bbr_deltas, pbr_deltas = self.detection_head(roi_features)
        #print(f"the dection head - class_scores: {class_scores.shape}, bbr_deltas: {bbr_deltas.shape}, pbr_deltas: {pbr_deltas.shape}")
        
        torch.cuda.empty_cache()
        
        if self.training and targets is not None:

            rpn_labels, sample_weights, valid_batch_mask, positive_anchors_per_batch = self.get_rpn_labels_and_positive_anchors(anchors, targets)
            if not valid_batch_mask.any():
                print("\n*********Skipping batch - no valid examples with good IoU matches***********************")
                # Create a zero tensor connected to the computation graph
                # by multiplying an existing tensor by 0
                dummy_loss = 0
                for name, param in self.named_parameters():
                    dummy_loss = dummy_loss + param.sum() * 0.0
                
                return {
                    'rpn_class_loss': dummy_loss,
                    'rpn_bbox_loss': dummy_loss,
                    'det_class_loss': dummy_loss,
                    'bbr_loss': dummy_loss,
                    'pbr_loss': dummy_loss
                }
            print(f"\nRPN Labels statistics:")
            print(f"Total proposals: {rpn_labels.numel()}")
            print(f"Positive anchors proposals: {(rpn_labels == 1).sum().item()}")
            print(f"Negative anchors proposals: {(rpn_labels == 0).sum().item()}")
            
            filtered_targets = targets[valid_batch_mask]
            filtered_positive_anchors = [positive_anchors_per_batch[i] for i in range(1) if valid_batch_mask[i]]
            rpn_bbox_targets = self.get_rpn_bbox_targets(anchors, filtered_targets, filtered_positive_anchors)

            

            rpn_class_loss = F.cross_entropy(
                rpn_scores.view(-1, 2),  # Reshape for binary classification
                rpn_labels
            )
            
            rpn_class_loss = F.cross_entropy( rpn_scores.view(-1, 2), rpn_labels,reduction='none')

            # Reshape and apply batch weights
            rpn_class_loss = rpn_class_loss.view(1, -1)
            weighted_loss = rpn_class_loss * sample_weights.unsqueeze(1)
            rpn_class_loss = weighted_loss.mean() 
            
            
            # Only compute regression loss for positive anchors
            #print(f'calculating the positive ancors')
            positive_anchors = rpn_labels == 1
            rpn_deltas_flat = rpn_deltas.view(-1, 4)
            #print(f'clculating the regression loss rpn_delta shape: {(rpn_deltas_flat[positive_anchors]).shape} and rpn_box_target shape: {(rpn_bbox_targets[positive_anchors]).shape}')
            

            if positive_anchors.any():
                rpn_bbox_loss = F.smooth_l1_loss(
                    rpn_deltas_flat[positive_anchors],
                    rpn_bbox_targets[positive_anchors]
                )
                
            else:
                rpn_bbox_loss = torch.tensor(0.0, device=x.device)
                
            # 2. Detection Head Losses
            # Get labels and targets for final detection
            #print(f'calculating the roi labels')
            
            
            
            # roi_labels = self.get_roi_labels(rois, targets)
            # print(f"Positive ROI proposals: {(roi_labels == 1).sum().item()}")
            # print(f"Negative proposals: {(roi_labels == 0).sum().item()}")
            # #print(f'calculating the roi box targets')
            roi_labels, roi_weights = self.get_roi_labels_and_weights(rois, targets)
            print(f"\nROI Labels statistics:")
            print(f"Total ROI proposals: {roi_labels.numel()}")
            print(f"Positive ROI proposals: {(roi_labels == 1).sum().item()}")
            print(f"Negative ROI proposals: {(roi_labels == 0).sum().item()}")
            
            roi_bbox_targets = self.get_roi_bbox_targets(rois, targets)
            
            # #print(f'Calculating the cross entopy for head %%%%%%%%% {class_scores.shape} and {roi_labels.shape}')
            # # Compute classification loss for detection head
            # det_class_loss = F.cross_entropy(
            #     class_scores,
            #     roi_labels.view(-1)
            # )
            
            # # Compute regression losses only for positive ROIs
            # #print(f'calculating the regression for the head')
            # positive_rois = roi_labels.view(-1) == 1
            # bbr_deltas_flat = bbr_deltas.view(-1, 4)
            # pbr_deltas_flat = pbr_deltas.view(-1, 4)
            #roi_bbox_targets_flat = roi_bbox_targets.view(-1, 4)
            # #print(f'bbr_deltas $$$$$$$$$: {(bbr_deltas_flat[positive_rois]).shape} and roi_bbox_targets {(roi_bbox_targets_flat[positive_rois]).shape}')
            # if positive_rois.any():
            #     # Bounding Box Regression (BBR) loss
            #     bbr_loss = F.smooth_l1_loss(
            #         bbr_deltas_flat[positive_rois],
            #         roi_bbox_targets_flat[positive_rois]
            #     )
                
            #     # Precise Boundary Regression (PBR) loss
            #     pbr_loss = F.smooth_l1_loss(
            #         pbr_deltas_flat[positive_rois],
            #         roi_bbox_targets_flat[positive_rois]
            #     )
            # else:
            #     bbr_loss = torch.tensor(0.0, device=x.device)
            #     pbr_loss = torch.tensor(0.0, device=x.device)
            
            # Weighted classification loss
            det_class_loss = F.cross_entropy(
                class_scores,
                roi_labels.view(-1),
                reduction='none'  # Get per-sample loss
            )
            det_class_loss = (det_class_loss * roi_weights.view(-1)).mean()
            
            # For regression losses
            positive_rois = roi_labels.view(-1) == 1
            bbr_deltas_flat = bbr_deltas.view(-1, 4)
            pbr_deltas_flat = pbr_deltas.view(-1, 4)
            roi_bbox_targets_flat = roi_bbox_targets.view(-1, 4)
            roi_weights_flat = roi_weights.view(-1)
            
            if positive_rois.any():
                # Get positive indices
                pos_indices = torch.where(positive_rois)[0]
                
                # Compute unweighted losses
                bbr_losses = F.smooth_l1_loss(
                    bbr_deltas_flat[pos_indices],
                    roi_bbox_targets_flat[pos_indices],
                    reduction='none'  # Keep per-element loss
                )
                
                pbr_losses = F.smooth_l1_loss(
                    pbr_deltas_flat[pos_indices],
                    roi_bbox_targets_flat[pos_indices],
                    reduction='none'
                )
                
                # Apply weights and take mean
                bbr_loss = (bbr_losses.mean(dim=1) * roi_weights_flat[pos_indices]).mean()
                pbr_loss = (pbr_losses.mean(dim=1) * roi_weights_flat[pos_indices]).mean()
            else:
                bbr_loss = torch.tensor(0.0, device=x.device)
                pbr_loss = torch.tensor(0.0, device=x.device)
            
            print(f'\n*******Following are the LOSSes***********')
            print(f'The rpn class loss is {rpn_class_loss}') 
            print(f'The rpn bbox loss is {rpn_bbox_loss}')  
            print(f'The det_class_loss is {det_class_loss}') 
            print(f'The detection head bbox loss is {bbr_loss}') 
            print(f'The detection head precise bbox loss is {pbr_loss}')  
            
            torch.cuda.empty_cache()
            # Return all losses in a dictionary
            return {
                'rpn_class_loss': rpn_class_loss,
                'rpn_bbox_loss': rpn_bbox_loss,
                'det_class_loss': det_class_loss,
                'bbr_loss': bbr_loss,
                'pbr_loss': pbr_loss
            }
        else:
            # Inference mode: return detected tables
            return self.get_final_detections(rois.squeeze(0), class_scores, bbr_deltas, pbr_deltas)