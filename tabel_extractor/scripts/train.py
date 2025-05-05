
# import argparse
# import logging
# import os
# from datetime import datetime, timedelta
# import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler
# from torch.optim import Adam
# from torch.cuda.amp import GradScaler  # Add this import for mixed precision
# from torch.profiler import profile, record_function, ProfilerActivity  # Add for profiling

# from tablesense.data.dataset import TableDetectionDataset
# from tablesense.models.tablesense import CompleteTableDetectionSystem
# from tablesense.utils.training_utils import create_data_loader, train_epoch, save_checkpoint
# from tablesense.config.default import ModelConfig, TrainingConfig
# import socket
# import random

# def setup_logging(log_dir: str) -> None:
#     """Configure logging to both file and console with timestamps."""
#     os.makedirs(log_dir, exist_ok=True)
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_file),
#             logging.StreamHandler()
#         ]
#     )
# def find_free_port():
#     """
#     Find a free port by testing ports in a reasonable range.
#     Think of it like finding an empty conference room in a large office building.
#     We'll try different rooms until we find one that's available.
#     """
#     # Try ports in range 20000-65000 to avoid common system ports
#     while True:
#         port = random.randint(20000, 65000)
#         try:
#             with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#                 s.bind(('', port))
#                 # Test if we can actually listen on this port
#                 s.listen(1)
#                 s.close()
#                 return port
#         except OSError:
#             # If port is taken, try another one
#             continue

# def setup(rank, world_size):
#     """
#     Set up the distributed training environment.
#     This is like establishing a secure conference line between our GPUs.
#     """
#     try:
#         # Only the main process (rank 0) should find the port
#         if rank == 0:
#             port = find_free_port()
#             print(f"Found free port: {port}")
#         else:
#             port = None
            
#         # Set up the basic environment
#         os.environ['MASTER_ADDR'] = 'localhost'
        
#         # If we're on rank 0 (main process), set the port
#         if rank == 0:
#             os.environ['MASTER_PORT'] = str(port)
#             # Save the port to a file so other processes can read it
#             with open('.port.txt', 'w') as f:
#                 f.write(str(port))
#         else:
#             # Other processes wait briefly and read the port
#             import time
#             time.sleep(1)  # Give rank 0 time to write the port
#             with open('.port.txt', 'r') as f:
#                 port = f.read().strip()
#             os.environ['MASTER_PORT'] = port
        
#         print(f"Process {rank} using port {port}")
        
#         # Initialize the process group
#         dist.init_process_group(
#             backend="nccl",
#             init_method='env://',
#             world_size=world_size,
#             rank=rank,
#             timeout=timedelta(minutes=120)
#         )
        
#         # Set the device for this process
#         torch.cuda.set_device(rank)
#         torch.cuda.empty_cache()
        
#         print(f"Process {rank} initialized successfully")
        
#     except Exception as e:
#         print(f"Error in setup for rank {rank}: {str(e)}")
#         raise

# def cleanup():
#     """Clean up the distributed environment after training."""
#     dist.destroy_process_group()
# def parse_args():
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(description='Train TableSense model')
#     parser.add_argument('--annotation-file', required=True,
#                       help='Path to annotation CSV file')
#     parser.add_argument('--checkpoint-dir', default='checkpoints',
#                       help='Directory to save checkpoints')
#     parser.add_argument('--log-dir', default='logs',
#                       help='Directory to save logs')
#     parser.add_argument('--batch-size', type=int, default=1,
#                       help='Batch size per GPU')
#     parser.add_argument('--epochs', type=int, default=50,
#                       help='Number of epochs to train')
#     parser.add_argument('--lr', type=float, default=0.001,
#                       help='Learning rate')
#     parser.add_argument('--num-workers', type=int, default=8,
#                       help='Number of data loading workers (total)')
#     # Add new arguments for mixed precision and profiling
#     parser.add_argument('--mixed-precision', action='store_true', 
#                       help='Enable mixed precision training')
#     parser.add_argument('--profile', action='store_true',
#                       help='Enable profiling (will run only a few batches)')
#     parser.add_argument('--profile-batches', type=int, default=5,
#                       help='Number of batches to profile')
#     return parser.parse_args()

# def train_model_distributed(rank, world_size, args):
#     """Training function that runs on each GPU."""
#     try:
#         # Initialize distributed environment
#         setup(rank, world_size)
        
#         # Set up logging for this process
#         if rank == 0:  
#             setup_logging(args.log_dir)
        
#         logging.info(f"Process {rank}/{world_size} starting training")

#         # Create model and move it to the correct GPU
#         model = CompleteTableDetectionSystem()
#         model = model.to(rank)
#         model = DDP(model, device_ids=[rank], output_device=rank)
        
#         # Create dataset
#         dataset = TableDetectionDataset(args.annotation_file)
        
#         # Create sampler for distributed training
#         sampler = DistributedSampler(
#             dataset,
#             num_replicas=world_size,
#             rank=rank,
#             shuffle=True
#         )
        
#         # Create data loader - keeping same batch size per GPU
#         data_loader = create_data_loader(
#             dataset,
#             batch_size=args.batch_size,
#             sampler=sampler,
#             num_workers=args.num_workers // world_size
#         )
        
#         # Create optimizer
#         optimizer = Adam(model.parameters(), lr=args.lr)
        
#         # Create GradScaler for mixed precision training
#         scaler = GradScaler() if args.mixed_precision else None
        
#         # Run profiling if enabled (only on rank 0)
#         if args.profile and rank == 0:
#             logging.info("Running profiling for a few batches...")
#             profile_results = run_profiling(
#                 model=model,
#                 data_loader=data_loader,
#                 optimizer=optimizer,
#                 device=rank,
#                 num_batches=args.profile_batches,
#                 use_amp=args.mixed_precision,
#                 scaler=scaler
#             )
#             logging.info("Profiling completed. Check profile_logs directory for results.")
#             # Clean up and exit
#             cleanup()
#             return
        
#         # Training loop
#         for epoch in range(args.epochs):
#             sampler.set_epoch(epoch)  # Important for proper shuffling
            
#             avg_loss = train_epoch(
#                 model=model,
#                 data_loader=data_loader,
#                 optimizer=optimizer,
#                 device=rank,
#                 epoch=epoch,
#                 use_amp=args.mixed_precision,  # Pass mixed precision flag
#                 scaler=scaler  # Pass scaler
#             )
            
#             # Save checkpoints only from the main process
#             if rank == 0 and (epoch + 1) % 5 == 0:
#                 save_checkpoint(
#                     model.module,  # Access the underlying model
#                     optimizer,
#                     epoch,
#                     avg_loss,
#                     os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
#                 )
#                 logging.info(f"Saved checkpoint at epoch {epoch + 1}")
                
#     except Exception as e:
#         logging.error(f"Error in process {rank}: {str(e)}")
#         raise
#     finally:
#         cleanup()

# def run_profiling(model, data_loader, optimizer, device, num_batches=5, use_amp=False, scaler=None):
#     """Run profiling for a few batches."""
#     model.train()
    
#     # Create profiling logs directory
#     os.makedirs("profile_logs", exist_ok=True)
    
#     logging.info(f"Starting profiling for {num_batches} batches (Device: {device})")
    
#     # Define activities to profile
#     activities = [
#         ProfilerActivity.CPU,
#         ProfilerActivity.CUDA,
#     ]
    
#     # Configure the profiler with appropriate settings
#     with profile(
#         activities=activities,
#         schedule=torch.profiler.schedule(
#             wait=1,       # Skip first batch for warmup
#             warmup=1,     # Collect for 1 batch for warmup
#             active=num_batches,  # Collect for num_batches batches
#             repeat=1
#         ),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile_logs'),
#         record_shapes=True,      # Record tensor shapes
#         profile_memory=True,     # Track tensor memory allocation
#         with_stack=True          # Record source code information
#     ) as prof:
        
#         # Process a few batches for profiling
#         for batch_idx, batch in enumerate(data_loader):
#             if batch_idx >= num_batches + 2:  # wait + warmup + active
#                 break
                
#             # Record data loading
#             with record_function("data_loading"):
#                 features = batch['features'].to(device, non_blocking=True)
#                 targets = batch['target'].to(device, non_blocking=True)
            
#             # Zero gradients
#             optimizer.zero_grad(set_to_none=True)
            
#             # Forward pass (with or without mixed precision)
#             if use_amp:
#                 with record_function("forward_amp"):
#                     with torch.cuda.amp.autocast():
#                         losses = model(features, targets)
#                         if losses is not None and isinstance(losses, dict):
#                             batch_loss = sum(losses.values())
#                         else:
#                             continue
                            
#                 # Backward pass with scaling
#                 with record_function("backward_amp"):
#                     scaler.scale(batch_loss).backward()
                
#                 # Optimizer step with scaling
#                 with record_function("optimizer_amp"):
#                     # Unscale before gradient clipping
#                     scaler.unscale_(optimizer)
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
#                     scaler.step(optimizer)
#                     scaler.update()
#             else:
#                 # Standard precision training
#                 with record_function("forward"):
#                     losses = model(features, targets)
#                     if losses is not None and isinstance(losses, dict):
#                         batch_loss = sum(losses.values())
#                     else:
#                         continue
                        
#                 # Standard backward pass
#                 with record_function("backward"):
#                     batch_loss.backward()
                
#                 # Standard optimizer step
#                 with record_function("optimizer"):
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
#                     optimizer.step()
            
#             # Step the profiler
#             prof.step()
    
#     # Print summarized results
#     logging.info("Top 10 operations by CUDA time:")
#     logging.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
#     logging.info("\nTop 10 operations by CPU time:")
#     logging.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
#     logging.info("\nTop memory-consuming operations:")
#     logging.info(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
    
#     # Save detailed results to a file
#     with open('profile_results.txt', 'w') as f:
#         f.write("CUDA Time Analysis:\n")
#         f.write(str(prof.key_averages().table(sort_by="cuda_time_total")))
#         f.write("\n\nCPU Time Analysis:\n")
#         f.write(str(prof.key_averages().table(sort_by="cpu_time_total")))
#         f.write("\n\nMemory Usage Analysis:\n")
#         f.write(str(prof.key_averages().table(sort_by="self_cuda_memory_usage")))
    
#     return prof


# def main():
#     """Main function to start distributed training."""
#     args = parse_args()
    
#     # Create necessary directories
#     os.makedirs(args.checkpoint_dir, exist_ok=True)
#     os.makedirs(args.log_dir, exist_ok=True)
    
#     # Get number of available GPUs
#     world_size = torch.cuda.device_count()
#     print(f"Starting distributed training on {world_size} GPUs")
    
#     # Launch processes for distributed training
#     mp.spawn(
#         train_model_distributed,
#         args=(world_size, args),
#         nprocs=world_size,
#         join=True
#     )

# if __name__ == "__main__":
#     main()




'''Traning with wandb'''


import argparse
import logging
import os
from datetime import datetime, timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from torch.cuda.amp import GradScaler  # Add this import for mixed precision
from torch.profiler import profile, record_function, ProfilerActivity  # Add for profiling
import wandb  # Import wandb

from tablesense.data.dataset import TableDetectionDataset
from tablesense.models.tablesense import CompleteTableDetectionSystem
from tablesense.utils.training_utils import create_data_loader, train_epoch, save_checkpoint
from tablesense.config.default import ModelConfig, TrainingConfig
import socket
import random

def setup_logging(log_dir: str) -> None:
    """Configure logging to both file and console with timestamps."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def find_free_port():
    """
    Find a free port by testing ports in a reasonable range.
    Think of it like finding an empty conference room in a large office building.
    We'll try different rooms until we find one that's available.
    """
    # Try ports in range 20000-65000 to avoid common system ports
    while True:
        port = random.randint(20000, 65000)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                # Test if we can actually listen on this port
                s.listen(1)
                s.close()
                return port
        except OSError:
            # If port is taken, try another one
            continue

def setup(rank, world_size):
    """
    Set up the distributed training environment.
    This is like establishing a secure conference line between our GPUs.
    """
    try:
        # Only the main process (rank 0) should find the port
        if rank == 0:
            port = find_free_port()
            print(f"Found free port: {port}")
        else:
            port = None
            
        # Set up the basic environment
        os.environ['MASTER_ADDR'] = 'localhost'
        
        # If we're on rank 0 (main process), set the port
        if rank == 0:
            os.environ['MASTER_PORT'] = str(port)
            # Save the port to a file so other processes can read it
            with open('.port.txt', 'w') as f:
                f.write(str(port))
        else:
            # Other processes wait briefly and read the port
            import time
            time.sleep(1)  # Give rank 0 time to write the port
            with open('.port.txt', 'r') as f:
                port = f.read().strip()
            os.environ['MASTER_PORT'] = port
        
        print(f"Process {rank} using port {port}")
        
        # Initialize the process group
        dist.init_process_group(
            backend="nccl",
            init_method='env://',
            world_size=world_size,
            rank=rank,
            timeout=timedelta(minutes=120)
        )
        
        # Set the device for this process
        torch.cuda.set_device(rank)
        torch.cuda.empty_cache()
        
        print(f"Process {rank} initialized successfully")
        
    except Exception as e:
        print(f"Error in setup for rank {rank}: {str(e)}")
        raise

def cleanup():
    """Clean up the distributed environment after training."""
    dist.destroy_process_group()

def initialize_wandb(args, rank):
    """
    Initialize Weights & Biases for experiment tracking.
    Only initializes on the main process (rank 0).
    
    Args:
        args: Command line arguments
        rank: The rank of the current process
    """
    if rank != 0:
        # Only initialize wandb on the main process
        return
    
    # Prepare config dictionary for wandb
    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'mixed_precision': args.mixed_precision,
        'model_type': 'TableSense',
        'optimizer': 'Adam',
        'num_gpus': torch.cuda.device_count(),
        'num_workers': args.num_workers,
        'annotation_file': args.annotation_file,
    }
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or f"tablesense-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config=config,
        notes=args.wandb_notes,
        tags=args.wandb_tags.split(',') if args.wandb_tags else None,
        mode="offline" if args.wandb_offline else "online",
    )
    
    # Log the architecture of the model (optional)
    # wandb.watch requires the model to be on the device, which happens later
    
    logging.info(f"Initialized wandb run: {wandb.run.name}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train TableSense model')
    parser.add_argument('--annotation-file', required=True,
                      help='Path to annotation CSV file')
    parser.add_argument('--checkpoint-dir', default='checkpoints',
                      help='Directory to save checkpoints')
    parser.add_argument('--log-dir', default='logs',
                      help='Directory to save logs')
    parser.add_argument('--batch-size', type=int, default=1,
                      help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=5,
                      help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=8,
                      help='Number of data loading workers (total)')
    # Add new arguments for mixed precision and profiling
    parser.add_argument('--mixed-precision', action='store_true', 
                      help='Enable mixed precision training')
    parser.add_argument('--profile', action='store_true',
                      help='Enable profiling (will run only a few batches)')
    parser.add_argument('--profile-batches', type=int, default=5,
                      help='Number of batches to profile')
                      
    # Add wandb arguments
    parser.add_argument('--use-wandb', action='store_true',
                      help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', default='tablesense',
                      help='wandb project name')
    parser.add_argument('--wandb-entity', default=None,
                      help='wandb entity name (username or team name)')
    parser.add_argument('--wandb-run-name', default=None,
                      help='wandb run name (default: auto-generated)')
    parser.add_argument('--wandb-notes', default=None,
                      help='Notes to add to the wandb run')
    parser.add_argument('--wandb-tags', default=None,
                      help='Comma-separated list of tags for the wandb run')
    parser.add_argument('--wandb-offline', action='store_true',
                      help='Run wandb in offline mode')
    parser.add_argument('--resume-wandb', default=None,
                      help='Resume a wandb run by providing its ID')
                      
    return parser.parse_args()

def train_model_distributed(rank, world_size, args):
    """Training function that runs on each GPU."""
    try:
        # Initialize distributed environment
        setup(rank, world_size)
        
        # Set up logging for this process
        if rank == 0:  
            setup_logging(args.log_dir)
            
            # Initialize wandb if enabled
            if args.use_wandb:
                initialize_wandb(args, rank)
        
        logging.info(f"Process {rank}/{world_size} starting training")

        # Create model and move it to the correct GPU
        model = CompleteTableDetectionSystem()
        model = model.to(rank)
        model = DDP(model, device_ids=[rank], output_device=rank)
        
        # Watch the model in wandb to track gradients and parameters
        if rank == 0 and args.use_wandb:
            wandb.watch(
                model,
                log="all",  # Log gradients and parameters
                log_freq=100,  # Log every 100 batches
                log_graph=True,  # Log model graph
            )
        
        # Create dataset
        dataset = TableDetectionDataset(args.annotation_file)
        
        # Create sampler for distributed training
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        # Create data loader - keeping same batch size per GPU
        data_loader = create_data_loader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers // world_size
        )
        
        # Create optimizer
        optimizer = Adam(model.parameters(), lr=args.lr)
        
        # Create GradScaler for mixed precision training
        scaler = GradScaler() if args.mixed_precision else None
        
        # Run profiling if enabled (only on rank 0)
        if args.profile and rank == 0:
            logging.info("Running profiling for a few batches...")
            profile_results = run_profiling(
                model=model,
                data_loader=data_loader,
                optimizer=optimizer,
                device=rank,
                num_batches=args.profile_batches,
                use_amp=args.mixed_precision,
                scaler=scaler
            )
            logging.info("Profiling completed. Check profile_logs directory for results.")
            
            # Upload profiling results to wandb if enabled
            if args.use_wandb:
                wandb.save('profile_results.txt')
                # Also save TensorBoard profiling data
                wandb.save('./profile_logs/*')
            
            # Clean up and exit
            cleanup()
            return
        
        # Create a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3,
            verbose=True
        )
        
        # Training loop
        best_loss = float('inf')
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)  # Important for proper shuffling
            
            # Train for one epoch
            avg_loss = train_epoch(
                model=model,
                data_loader=data_loader,
                optimizer=optimizer,
                device=rank,
                epoch=epoch,
                use_amp=args.mixed_precision,
                scaler=scaler,
                log_wandb=args.use_wandb and rank == 0  # Only log wandb from rank 0
            )
            
            # Step the learning rate scheduler
            if rank == 0:
                lr_scheduler.step(avg_loss)
                
                # Log learning rate
                if args.use_wandb:
                    wandb.log({
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'epoch': epoch
                    })
                
                # Check if this is the best model so far
                is_best = avg_loss < best_loss
                if is_best:
                    best_loss = avg_loss
                    # Save the best model
                    best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                    save_checkpoint(
                        model.module,
                        optimizer,
                        epoch,
                        avg_loss,
                        best_model_path,
                        log_wandb=args.use_wandb
                    )
                    logging.info(f"New best model saved with loss: {best_loss:.4f}")
                    
                    # Log best model metrics to wandb
                    if args.use_wandb:
                        wandb.run.summary['best_loss'] = best_loss
                        wandb.run.summary['best_epoch'] = epoch
            
            # Save checkpoints at regular intervals
            if rank == 0 and (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
                save_checkpoint(
                    model.module,  # Access the underlying model
                    optimizer,
                    epoch,
                    avg_loss,
                    checkpoint_path,
                    log_wandb=args.use_wandb
                )
                logging.info(f"Saved checkpoint at epoch {epoch + 1}")
                
        # Finish the wandb run
        if rank == 0 and args.use_wandb:
            wandb.finish()
                
    except Exception as e:
        logging.error(f"Error in process {rank}: {str(e)}")
        if rank == 0 and args.use_wandb:
            # Log the error to wandb
            wandb.finish(exit_code=1)
        raise
    finally:
        cleanup()

def run_profiling(model, data_loader, optimizer, device, num_batches=5, use_amp=False, scaler=None):
    """Run profiling for a few batches."""
    model.train()
    
    # Create profiling logs directory
    os.makedirs("profile_logs", exist_ok=True)
    
    logging.info(f"Starting profiling for {num_batches} batches (Device: {device})")
    
    # Define activities to profile
    activities = [
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ]
    
    # Configure the profiler with appropriate settings
    with profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=1,       # Skip first batch for warmup
            warmup=1,     # Collect for 1 batch for warmup
            active=num_batches,  # Collect for num_batches batches
            repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile_logs'),
        record_shapes=True,      # Record tensor shapes
        profile_memory=True,     # Track tensor memory allocation
        with_stack=True          # Record source code information
    ) as prof:
        
        # Process a few batches for profiling
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= num_batches + 2:  # wait + warmup + active
                break
                
            # Record data loading
            with record_function("data_loading"):
                features = batch['features'].to(device, non_blocking=True)
                targets = batch['target'].to(device, non_blocking=True)
            
            # Zero gradients
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass (with or without mixed precision)
            if use_amp:
                with record_function("forward_amp"):
                    with torch.cuda.amp.autocast():
                        losses = model(features, targets)
                        if losses is not None and isinstance(losses, dict):
                            batch_loss = sum(losses.values())
                        else:
                            continue
                            
                # Backward pass with scaling
                with record_function("backward_amp"):
                    scaler.scale(batch_loss).backward()
                
                # Optimizer step with scaling
                with record_function("optimizer_amp"):
                    # Unscale before gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                    scaler.step(optimizer)
                    scaler.update()
            else:
                # Standard precision training
                with record_function("forward"):
                    losses = model(features, targets)
                    if losses is not None and isinstance(losses, dict):
                        batch_loss = sum(losses.values())
                    else:
                        continue
                        
                # Standard backward pass
                with record_function("backward"):
                    batch_loss.backward()
                
                # Standard optimizer step
                with record_function("optimizer"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                    optimizer.step()
            
            # Step the profiler
            prof.step()
    
    # Print summarized results
    logging.info("Top 10 operations by CUDA time:")
    logging.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    logging.info("\nTop 10 operations by CPU time:")
    logging.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    logging.info("\nTop memory-consuming operations:")
    logging.info(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
    
    # Save detailed results to a file
    with open('profile_results.txt', 'w') as f:
        f.write("CUDA Time Analysis:\n")
        f.write(str(prof.key_averages().table(sort_by="cuda_time_total")))
        f.write("\n\nCPU Time Analysis:\n")
        f.write(str(prof.key_averages().table(sort_by="cpu_time_total")))
        f.write("\n\nMemory Usage Analysis:\n")
        f.write(str(prof.key_averages().table(sort_by="self_cuda_memory_usage")))
    
    return prof


def main():
    """Main function to start distributed training."""
    args = parse_args()
    
    # Create necessary directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    print(f"Starting distributed training on {world_size} GPUs")
    
    # Launch processes for distributed training
    mp.spawn(
        train_model_distributed,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()








