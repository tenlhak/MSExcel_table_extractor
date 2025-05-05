import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from .feature_extractor import extract_table_features



class TableDetectionDataset(Dataset):
    """
    Dataset class for loading Excel files and their table annotations.
    This class handles the complex task of converting Excel files and their 
    annotations into a format our model can understand.
    """
    def __init__(self, annotation_file, root_dir='/home/dapgrad/tenzinl2/lumina/luminaData/'):
        """
        Initialize the dataset with annotations file path.
        
        Args:
            annotation_file: Path to the CSV containing Excel file paths and annotations
            root_dir: Optional root directory to prepend to file paths
        """
        # Read our annotations file
        self.annotations = pd.read_csv(annotation_file)
        self.root_dir = root_dir
        
        # Parse the target labels into a more usable format
        self.parsed_targets = []
        for _, row in self.annotations.iterrows():
            # The target_label column contains cell ranges like 'A1:B14'
            excel_path = os.path.join(root_dir, row['full_path']).replace('\\', '/') if root_dir else row['full_path']
            sheet_name = row['sheet_name']
            table_range = row['target_label']
            
            # Convert Excel-style cell references to numeric coordinates
            start_cell, end_cell = table_range.split(':')
            x1, y1 = self._convert_excel_ref_to_coords(start_cell)
            x2, y2 = self._convert_excel_ref_to_coords(end_cell)
            
            self.parsed_targets.append({
                'excel_path': excel_path,
                'sheet_name': sheet_name,
                'table_coords': [x1, y1, x2, y2]
            })

    def _convert_excel_ref_to_coords(self, cell_ref):
        """
        Convert Excel-style cell references (e.g., 'A1') to (x, y) coordinates.
        
        Args:
            cell_ref: String like 'A1', 'B2', etc.
            
        Returns:
            Tuple of (x, y) coordinates (0-based)
        """
        # Split the column letters from row numbers
        col = ''
        row = ''
        for char in cell_ref:
            if char.isalpha():
                col += char
            else:
                row += char
        
        # Convert column letters to numbers (A=0, B=1, etc.)
        col_num = 0
        for i, letter in enumerate(reversed(col.upper())):
            col_num += (ord(letter) - ord('A') + 1) * (26 ** i)
        col_num -= 1  # Make 0-based
        
        row_num = int(row) - 1  # Make 0-based
        
        return col_num, row_num

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.parsed_targets)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to fetch
            
        Returns:
            Dictionary containing:
                - features: Tensor of shape [20, H, W] containing cell features
                - target: Tensor of table coordinates [x1, y1, x2, y2]
                - meta: Dictionary of metadata (file path, sheet name)
        """
        target = self.parsed_targets[idx]
        
        # Extract cell features from the Excel file
        features = extract_table_features(
            target['excel_path'], 
            target['sheet_name']
        )
        
        # Convert features to tensor
        features = torch.FloatTensor(features)
        # Reshape from [H, W, 20] to [20, H, W] for PyTorch convention
        features = features.permute(2, 0, 1)
        
        # Convert target coordinates to tensor
        target_tensor = torch.tensor(target['table_coords'], dtype=torch.float32)
        
        return {
            'features': features,
            'target': target_tensor,
            'meta': {
                'excel_path': target['excel_path'],
                'sheet_name': target['sheet_name']
            }
        }
