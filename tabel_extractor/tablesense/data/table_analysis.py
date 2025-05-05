import torch
import numpy as np
from tablesense.data.dataset import TableDetectionDataset
from collections import defaultdict

class TableStatisticsAnalyzer:
    """
    Analyzes table statistics from the dataset to help optimize anchor generation.
    Designed for CLI environments without visualization requirements.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.stats = defaultdict(list)
        
    def collect_statistics(self):
        """
        Gathers comprehensive statistics about tables in the dataset.
        """
        print("\nAnalyzing dataset statistics...")
        print(f"Total number of samples: {len(self.dataset)}")
        
        for idx in range(len(self.dataset)):
            if idx % 100 == 0:
                print(f"Processing sample {idx}/{len(self.dataset)}")
                
            # Get sample and extract table coordinates
            sample = self.dataset[idx]
            coords = sample['target'].numpy()
            x1, y1, x2, y2 = coords
            print(f'\nThe first coord is {coords}')
            # Calculate dimensions
            width = x2 - x1 + 1  # Adding 1 since coordinates are inclusive
            height = y2 - y1 + 1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            area = width * height
            aspect_ratio = width / height
            
            # Store all measurements
            self.stats['widths'].append(width)
            self.stats['heights'].append(height)
            self.stats['ratios'].append(aspect_ratio)
            self.stats['areas'].append(area)
            self.stats['center_x'].append(center_x)
            self.stats['center_y'].append(center_y)
            
    def print_detailed_statistics(self):
        """
        Prints comprehensive statistical analysis of the tables.
        """
        def get_stats(data, name):
            percentiles = np.percentile(data, [10, 25, 50, 75, 90])
            return (
                f"\n{name} Statistics:"
                f"\n  Mean: {np.mean(data):.2f}"
                f"\n  Std Dev: {np.std(data):.2f}"
                f"\n  Min: {np.min(data):.2f}"
                f"\n  10th percentile: {percentiles[0]:.2f}"
                f"\n  25th percentile: {percentiles[1]:.2f}"
                f"\n  Median: {percentiles[2]:.2f}"
                f"\n  75th percentile: {percentiles[3]:.2f}"
                f"\n  90th percentile: {percentiles[4]:.2f}"
                f"\n  Max: {np.max(data):.2f}"
            )

        print("\n" + "="*50)
        print("DETAILED TABLE STATISTICS")
        print("="*50)
        
        # Print statistics for each measurement
        for name in ['widths', 'heights', 'ratios', 'areas']:
            print(get_stats(self.stats[name], name.capitalize()))
            
        # Location analysis
        print("\nLocation Statistics:")
        print(f"X-coordinate range: {min(self.stats['center_x']):.1f} to {max(self.stats['center_x']):.1f}")
        print(f"Y-coordinate range: {min(self.stats['center_y']):.1f} to {max(self.stats['center_y']):.1f}")
        
    def get_anchor_recommendations(self):
        """
        Analyzes statistics to make specific recommendations for anchor configurations.
        """
        # Calculate recommended scales based on size distribution
        width_percentiles = np.percentile(self.stats['widths'], [25, 50, 75])
        height_percentiles = np.percentile(self.stats['heights'], [25, 50, 75])
        ratio_percentiles = np.percentile(self.stats['ratios'], [25, 50, 75])
        
        # Calculate scale recommendations
        min_dimension = min(np.median(self.stats['widths']), 
                          np.median(self.stats['heights']))
        recommended_scales = [
            max(8, min_dimension / 4),    # Small tables
            max(16, min_dimension / 2),   # Medium-small tables
            max(32, min_dimension),       # Medium tables
            max(64, min_dimension * 2)    # Large tables
        ]
        
        # Calculate ratio recommendations
        recommended_ratios = [
            max(0.5, ratio_percentiles[0]),    # Common narrow ratio
            1.0,                               # Square ratio
            min(2.0, ratio_percentiles[2])     # Common wide ratio
        ]
        
        print("\n" + "="*50)
        print("ANCHOR RECOMMENDATIONS")
        print("="*50)
        print("\nRecommended scales (in cells):")
        for i, scale in enumerate(recommended_scales, 1):
            print(f"Scale {i}: {scale:.1f}")
            
        print("\nRecommended aspect ratios:")
        for i, ratio in enumerate(recommended_ratios, 1):
            print(f"Ratio {i}: {ratio:.2f}")
            
        print("\nCurrent vs Recommended Configuration:")
        print("Current scales: [8, 16, 32, 64]")
        print(f"Recommended scales: {[round(s) for s in recommended_scales]}")
        print("Current ratios: [0.5, 1.0, 2.0]")
        print(f"Recommended ratios: {[round(r, 2) for r in recommended_ratios]}")

def main():
    # Update these paths for your environment
    annotation_file = "/home/dapgrad/tenzinl2/lumina/luminaData/processed_enron_paths.csv"
    root_dir = "/home/dapgrad/tenzinl2/lumina/luminaData/"
    
    print("Initializing dataset...")
    dataset = TableDetectionDataset(annotation_file, root_dir)
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Create analyzer and run analysis
    analyzer = TableStatisticsAnalyzer(dataset)
    analyzer.collect_statistics()
    analyzer.print_detailed_statistics()
    analyzer.get_anchor_recommendations()

if __name__ == "__main__":
    main()