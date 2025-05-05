import time
import torch
import psutil
from contextlib import contextmanager

class PerformanceMonitor:
    """
    A utility class to track detailed performance metrics during RPN processing.
    This helps us understand where time is being spent and how memory is being used.
    """
    def __init__(self, rank):
        self.rank = rank  # GPU rank for distributed training
        self.timings = {}
        self.memory_stats = {}
        self.spatial_stats = {}
    
    @contextmanager
    def timer(self, name):
        """
        Context manager to measure time spent in different code sections.
        Usage:
            with monitor.timer("section_name"):
                # code to measure
        """
        start = time.time()
        try:
            yield
        finally:
            end = time.time()
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(end - start)
    
    def log_memory(self, name):
        """
        Log current GPU and CPU memory usage.
        """
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated(self.rank) / 1024**2
            gpu_memory_cached = torch.cuda.memory_reserved(self.rank) / 1024**2
        else:
            gpu_memory_used = gpu_memory_cached = 0
            
        cpu_memory = psutil.Process().memory_info().rss / 1024**2
        
        self.memory_stats[name] = {
            'gpu_used': gpu_memory_used,
            'gpu_cached': gpu_memory_cached,
            'cpu_used': cpu_memory
        }
    
    def log_spatial_distribution(self, name, anchors, positive_indices):
        """
        Analyze the spatial distribution of proposals.
        """
        if isinstance(anchors, torch.Tensor):
            # Calculate spatial statistics
            anchor_widths = anchors[:, 2] - anchors[:, 0]
            anchor_heights = anchors[:, 3] - anchors[:, 1]
            
            self.spatial_stats[name] = {
                'mean_width': anchor_widths.mean().item(),
                'mean_height': anchor_heights.mean().item(),
                'width_std': anchor_widths.std().item(),
                'height_std': anchor_heights.std().item(),
                'positive_mean_width': anchor_widths[positive_indices].mean().item() if len(positive_indices) > 0 else 0,
                'positive_mean_height': anchor_heights[positive_indices].mean().item() if len(positive_indices) > 0 else 0
            }
    
    def print_summary(self):
        """
        Print a comprehensive summary of all collected metrics.
        """
        print(f"\n{'='*20} Performance Summary {'='*20}")
        
        print("\nTiming Statistics:")
        for section, times in self.timings.items():
            avg_time = sum(times) / len(times)
            print(f"{section:30s}: {avg_time:.4f}s (avg of {len(times)} runs)")
        
        print("\nMemory Statistics:")
        for point, stats in self.memory_stats.items():
            print(f"\n{point}:")
            print(f"  GPU Memory Used: {stats['gpu_used']:.2f} MB")
            print(f"  GPU Memory Cached: {stats['gpu_cached']:.2f} MB")
            print(f"  CPU Memory Used: {stats['cpu_used']:.2f} MB")
        
        print("\nSpatial Statistics:")
        for point, stats in self.spatial_stats.items():
            print(f"\n{point}:")
            print(f"  Mean Anchor Width: {stats['mean_width']:.2f}")
            print(f"  Mean Anchor Height: {stats['mean_height']:.2f}")
            print(f"  Width Std Dev: {stats['width_std']:.2f}")
            print(f"  Height Std Dev: {stats['height_std']:.2f}")
            print(f"  Positive Anchors Mean Width: {stats['positive_mean_width']:.2f}")
            print(f"  Positive Anchors Mean Height: {stats['positive_mean_height']:.2f}")