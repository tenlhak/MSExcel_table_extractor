import torch
from tablesense.models.tablesense import CompleteTableDetectionSystem

def count_parameters(model):
    """
    Count the total number of learnable parameters in the model.
    Also breaks down parameters by layer for better understanding.
    
    Args:
        model: PyTorch model
        
    Returns:
        total_params: Total number of learnable parameters
    """
    # Get total parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Print detailed breakdown by layer
    print("\nDetailed parameter count by layer:")
    print("-" * 60)
    print(f"{'Layer':<40} {'Parameters':<10}")
    print("-" * 60)
    
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            param_count = parameter.numel()
            print(f"{name:<40} {param_count:<10,}")
    
    print("-" * 60)
    print(f"Total trainable parameters: {total_params:,}")
    
    return total_params

def model_summary(model):
    """
    Print a comprehensive summary of the model architecture and parameters.
    Shows both trainable and non-trainable parameters.
    """
    print("\nModel Summary:")
    print("=" * 70)
    
    trainable_params = 0
    non_trainable_params = 0
    
    # Dictionary to store layer types and their parameter counts
    layer_stats = {}
    
    for name, param in model.named_parameters():
        shape = tuple(param.shape)
        num_params = param.numel()
        
        # Get layer type from parameter name
        layer_type = name.split('.')[0]
        if layer_type not in layer_stats:
            layer_stats[layer_type] = {'trainable': 0, 'non_trainable': 0}
        
        if param.requires_grad:
            trainable_params += num_params
            layer_stats[layer_type]['trainable'] += num_params
        else:
            non_trainable_params += num_params
            layer_stats[layer_type]['non_trainable'] += num_params
            
        print(f"Layer: {name:<30} | Shape: {str(shape):<20} | Parameters: {num_params:,}")
    
    print("\nParameter statistics by layer type:")
    print("-" * 70)
    for layer_type, stats in layer_stats.items():
        print(f"Layer type: {layer_type}")
        print(f"  Trainable parameters: {stats['trainable']:,}")
        print(f"  Non-trainable parameters: {stats['non_trainable']:,}")
        print("-" * 40)
    
    print("\nTotal Statistics:")
    print(f"Total trainable parameters: {trainable_params:,}")
    print(f"Total non-trainable parameters: {non_trainable_params:,}")
    print(f"Total parameters: {trainable_params + non_trainable_params:,}")
    print("=" * 70)

if __name__ == "__main__":
    # Initialize the model and move it to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    try:
        # Create the model
        print("\nInitializing model...")
        model = CompleteTableDetectionSystem()
        model = model.to(device)
        print("Model initialized successfully!")
        
        # Print the model summary
        print("\nGenerating model summary...")
        model_summary(model)
        
        # Print parameter count
        print("\nCounting parameters...")
        count_parameters(model)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nTraceback:")
        import traceback
        traceback.print_exc()