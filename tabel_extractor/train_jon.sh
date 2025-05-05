#!/bin/bash
#SBATCH --job-name=epoch5          # Name of the job
#SBATCH --output=output_%j.log         # Output file (with job ID in the name)
#SBATCH --error=error_%j.log           # Error file (with job ID in the name)
#SBATCH --ntasks=1                   # Number of tasks (processes)
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=32G                     # Memory required per node
#SBATCH --partition=a2il             # Partition (queue) to submit to
#SBATCH --gres=gpu:1                 # Request 2 GPUs
#SBATCH --exclude=deepbull1,deepbull6     
#SBATCH --mail-type=ALL              # Email notifications (BEGIN, END, FAIL)
#SBATCH --mail-user=tenzinl2@buffalo.edu  # Email address for notifications

# Load necessary modules (if required)
# module load python/3.8

# Activate the Python virtual environment
source /home/dapgrad/tenzinl2/lumina/lumina/bin/activate

# Set the environment variable to help with CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Your actual job command(s)
echo "Starting the job..."

srun python scripts/train.py \
  --annotation-file /home/dapgrad/tenzinl2/lumina/luminaData/processed_enron_paths_filtered.csv \
  --use-wandb \
  --wandb-project tablesense \
  --wandb-entity tenlhak98-university-at-buffalo \
  --wandb-tags "baseline,high-res"
# srun python scripts/train.py \
#     --annotation-file /home/dapgrad/tenzinl2/lumina/luminaData/processed_enron_paths_filtered.csv \
#     --checkpoint-dir checkpoints_jobs \
#     --log-dir logs \
#     --batch-size 1 \
#     --epochs 50 \
#     --lr 1e-4 \
#     --num-workers 8  # Example: Run a Python script
echo "Job completed!"

# Deactivate the virtual environment (optional)
deactivate
