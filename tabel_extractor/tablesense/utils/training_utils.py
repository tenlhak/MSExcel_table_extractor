import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import logging
from tqdm import tqdm
import wandb  # Import wandb


def create_data_loader(
    dataset: Any,
    batch_size: int,
    num_workers: int = 8,
    shuffle: bool = True,
    sampler: Optional[torch.utils.data.Sampler] = None,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create a DataLoader with proper configuration for both single and multi-GPU training.
    
    Args:
        dataset: Dataset instance
        batch_size: Number of samples per batch per GPU
        num_workers: Number of data loading worker processes
        shuffle: Whether to shuffle the data (ignored if sampler is provided)
        sampler: Optional sampler for distributed training
        pin_memory: Whether to pin memory in GPU training (default: True)
        
    Returns:
        Configured DataLoader instance
    """
    try:
        # If a sampler is provided, we shouldn't shuffle
        # This is important for distributed training
        if sampler is not None:
            shuffle = False
            logging.info(f"Using provided sampler, shuffle set to False")
        
        # Configure the DataLoader with all necessary parameters
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            sampler=sampler,
            #collate_fn=pad_to_max,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            drop_last=True  # Prevents issues with the last incomplete batch
        )
        
        # Log configuration for debugging
        logging.info(
            f"Created DataLoader with: batch_size={batch_size}, "
            f"num_workers={num_workers}, shuffle={shuffle}, "
            f"sampler={'provided' if sampler else 'None'}"
        )
        
        return loader
        
    except Exception as e:
        logging.error(f"Failed to create DataLoader: {str(e)}")
        raise

import os

# Increase the timeout for GPU communication
os.environ['NCCL_TIMEOUT'] = '1800'  # 30 minutes instead of 10

# Enable detailed debugging information
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

# Enable flight recorder for better error tracking
os.environ['TORCH_NCCL_TRACE_BUFFER_SIZE'] = '1048576' 


def train_epoch(model, data_loader, optimizer, device, epoch, use_amp=False, scaler=None, log_wandb=True):
    """Train the model for one epoch with optional mixed precision support and wandb logging."""
    model.train()
    total_loss = 0
    num_batches = len(data_loader)
    
    pbar = tqdm(data_loader, desc=f'Epoch {epoch}')
    
    # Track batch-level metrics for wandb
    batch_losses = []
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # Move data to correct device (with non_blocking for potential speedup)
            features = batch['features'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)
            
            # Zero gradients more efficiently
            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision training path
            if use_amp and scaler is not None:
                # Forward pass with autocast
                with torch.cuda.amp.autocast():
                    losses = model(features, targets)
                    
                    if losses is None or not isinstance(losses, dict):
                        logging.warning(f"GPU {device}: Invalid losses returned for batch {batch_idx}")
                        continue
                        
                    batch_loss = sum(losses.values())
                    
                    # Early detection of invalid loss
                    if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                        logging.warning(f"GPU {device}: Invalid loss in batch {batch_idx}")
                        continue
                
                # Scaled backward pass
                scaler.scale(batch_loss).backward()
                
                # Unscale before gradient clipping
                scaler.unscale_(optimizer)
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                
                # Check for invalid gradients after clipping
                valid_gradients = True
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            valid_gradients = False
                            logging.warning(f"GPU {device}: Invalid gradients in {name} after clipping in batch {batch_idx}")
                            break
                
                # Skip optimizer step if gradients are invalid
                if not valid_gradients:
                    logging.warning(f"GPU {device}: Skipping parameter update for batch {batch_idx} due to invalid gradients")
                    continue
                
                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
            
            # Standard precision training path
            else:
                # Forward pass
                losses = model(features, targets)
                
                if losses is None or not isinstance(losses, dict):
                    logging.warning(f"GPU {device}: Invalid losses returned for batch {batch_idx}")
                    continue
                    
                batch_loss = sum(losses.values())
                
                # Early detection of invalid loss
                if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                    logging.warning(f"GPU {device}: Invalid loss in batch {batch_idx}")
                    continue
                
                # Backward pass
                batch_loss.backward()
                
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                
                # Check for invalid gradients after clipping
                valid_gradients = True
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            valid_gradients = False
                            logging.warning(f"GPU {device}: Invalid gradients in {name} after clipping in batch {batch_idx}")
                            break
                
                # Skip optimizer step if gradients are invalid
                if not valid_gradients:
                    logging.warning(f"GPU {device}: Skipping parameter update for batch {batch_idx} due to invalid gradients")
                    continue
                
                # Standard optimizer step
                optimizer.step()
            
            # Update metrics
            batch_loss_value = batch_loss.item()
            total_loss += batch_loss_value
            batch_losses.append(batch_loss_value)
            
            # For wandb logging - collect detailed metrics
            if log_wandb and device == 0 and batch_idx % 10 == 0:  # Log every 10th batch to avoid overwhelming wandb
                metrics = {
                    'batch': batch_idx + epoch * num_batches,
                    'batch_loss': batch_loss_value,
                    'running_loss': total_loss / (batch_idx + 1),
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'memory_used_GB': torch.cuda.max_memory_allocated(device) / (1024 ** 3),
                }
                
                # Add individual loss components
                for loss_name, loss_value in losses.items():
                    metrics[f'loss_{loss_name}'] = loss_value.item()
                
                # Log gradient norm if available
                if 'grad_norm' in locals():
                    metrics['gradient_norm'] = grad_norm.item()
                
                # Log to wandb
                wandb.log(metrics)
            
            # Update progress bar
            if device == 0 and isinstance(pbar, tqdm):
                memory_used = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
                pbar.set_postfix({
                    'gpu': device,
                    'loss': f'{batch_loss.item():.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                    'mem': f'{memory_used:.2f}GB'
                })
            
        except Exception as e:
            logging.error(f"Error in batch {batch_idx} on GPU {device}: {str(e)}")
            torch.cuda.synchronize()
            continue
    
    # Clean up
    torch.cuda.synchronize()
    if hasattr(model, 'module'):
        torch.distributed.barrier()
        
    # Calculate average loss for this epoch
    avg_loss = total_loss / num_batches
    
    # Log epoch-level metrics to wandb
    if log_wandb and device == 0:
        wandb.log({
            'epoch': epoch,
            'epoch_loss': avg_loss,
            'epoch_loss_min': min(batch_losses) if batch_losses else float('inf'),
            'epoch_loss_max': max(batch_losses) if batch_losses else float('inf'),
            'epoch_loss_std': torch.tensor(batch_losses).std().item() if batch_losses else 0,
        })
    
    return avg_loss

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    log_wandb: bool = True
) -> None:
    """
    Save a model checkpoint and optionally log to wandb.
    
    Args:
        model: The neural network model
        optimizer: Optimizer instance
        epoch: Current epoch number
        loss: Current loss value
        filepath: Path to save checkpoint
        log_wandb: Whether to log the checkpoint to wandb
    """
    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        
        torch.save(checkpoint, filepath)
        logging.info(f"Checkpoint saved successfully to {filepath}")
        
        # Log checkpoint to wandb if enabled
        if log_wandb:
            artifact = wandb.Artifact(
                name=f"model-checkpoint-epoch-{epoch}", 
                type="model", 
                description=f"Model checkpoint from epoch {epoch} with loss {loss:.4f}"
            )
            artifact.add_file(filepath)
            wandb.log_artifact(artifact)
            logging.info(f"Uploaded checkpoint to wandb as artifact")
            
    except Exception as e:
        logging.error(f"Failed to save checkpoint: {str(e)}")

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str,
    download_from_wandb: bool = False,
    wandb_artifact_name: str = None
) -> tuple:
    """
    Load a model checkpoint, with optional downloading from wandb.
    
    Args:
        model: The neural network model
        optimizer: Optimizer instance
        filepath: Path to load checkpoint from
        download_from_wandb: Whether to download the checkpoint from wandb
        wandb_artifact_name: Name of the wandb artifact to download (if download_from_wandb is True)
        
    Returns:
        Tuple of (epoch, loss)
    """
    try:
        # Download the checkpoint from wandb if requested
        if download_from_wandb and wandb_artifact_name:
            artifact = wandb.use_artifact(wandb_artifact_name)
            artifact_dir = artifact.download()
            filepath = os.path.join(artifact_dir, os.path.basename(filepath))
            logging.info(f"Downloaded checkpoint from wandb to {filepath}")
        
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        logging.info(f"Checkpoint loaded successfully from {filepath}")
        return epoch, loss
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {str(e)}")
        raise