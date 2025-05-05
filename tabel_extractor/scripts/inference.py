import argparse
import logging
import os
from typing import List, Tuple
import torch
import numpy as np

from tablesense.models.tablesense import CompleteTableDetectionSystem
from tablesense.data.feature_extractor import extract_table_features
from tablesense.config.default import ModelConfig

def setup_logging():
    """Configure logging for inference."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def parse_args():
    """Parse command line arguments for inference."""
    parser = argparse.ArgumentParser(description='Run TableSense inference')
    parser.add_argument('--model-path', required=True,
                      help='Path to trained model checkpoint')
    parser.add_argument('--excel-file', required=True,
                      help='Path to Excel file for inference')
    parser.add_argument('--sheet-name', required=True,
                      help='Name of the sheet to process')
    parser.add_argument('--output-file', default='predictions.txt',
                      help='Path to save detection results')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Detection confidence threshold')
    return parser.parse_args()

def load_model(model_path: str, device: torch.device) -> CompleteTableDetectionSystem:
    """Load the trained model."""
    model_config = ModelConfig()
    model = CompleteTableDetectionSystem(
        input_channels=model_config.input_channels,
        hidden_dim=model_config.hidden_dim
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def excel_coords_to_range(x1: int, y1: int, x2: int, y2: int) -> str:
    """Convert numeric coordinates to Excel range format (e.g., 'A1:B2')."""
    def num_to_col(n):
        result = ""
        while n > 0:
            n -= 1
            result = chr(n % 26 + ord('A')) + result
            n //= 26
        return result
    
    return f'{num_to_col(x1+1)}{y1+1}:{num_to_col(x2+1)}{y2+1}'

def detect_tables(
    model: CompleteTableDetectionSystem,
    features: torch.Tensor,
    threshold: float,
    device: torch.device
) -> List[Tuple[str, float]]:
    """Run inference and return detected tables."""
    with torch.no_grad():
        # Add batch dimension and move to device
        features = features.unsqueeze(0).to(device)
        
        # Run inference
        detections = model(features, targets=None)  # None for inference mode
        
        # Process detections
        results = []
        for det in detections:
            if det[4] >= threshold:  # Check confidence threshold
                x1, y1, x2, y2 = map(int, det[:4])
                confidence = float(det[4])
                excel_range = excel_coords_to_range(x1, y1, x2, y2)
                results.append((excel_range, confidence))
        
        return results

def main():
    # Setup
    args = parse_args()
    setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    try:
        # Load model
        model = load_model(args.model_path, device)
        logging.info('Model loaded successfully')
        
        # Extract features
        features = extract_table_features(args.excel_file, args.sheet_name)
        features = torch.FloatTensor(features).permute(2, 0, 1)  # [C, H, W]
        logging.info(f'Extracted features of shape {features.shape}')
        
        # Detect tables
        detections = detect_tables(model, features, args.threshold, device)
        
        # Save results
        with open(args.output_file, 'w') as f:
            f.write(f'Sheet: {args.sheet_name}\n')
            f.write('Detected Tables:\n')
            for excel_range, confidence in detections:
                f.write(f'Range: {excel_range}, Confidence: {confidence:.4f}\n')
        
        logging.info(f'Found {len(detections)} tables. Results saved to {args.output_file}')
        
    except Exception as e:
        logging.error(f'Inference failed: {str(e)}')
        return

if __name__ == '__main__':
    main()