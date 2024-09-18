#!/bin/bash

#SBATCH --exclude=gvnodeb007

#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=4G   # memory per cpu-core

#SBATCH -t 3-00:00:00           #Time Limit d-hh:mm:ss
#SBATCH --partition=V4V32_SKY32M192_L       #partition/queue CAC48M192_L
#SBATCH --account=gcl_lsa273_uksr   #project allocation accout 

#SBATCH  --output=./lcc_logs/%x_%j.out     #Output file name
#SBATCH  --error=./lcc_logs/%x_%j.err      #Error file name

#SBATCH --mail-type NONE                 #Send email on start/end
#SBATCH --mail-user ofsk222@uky.edu     #Where to send email


module purge
module load ccs/singularity
echo "Job $SLURM_JOB_ID running on SLURM NODELIST: $SLURM_NODELIST "

CONTAINER="$PROJECT/lsa273_uksr/containers/information_plane/information_plane.sif"

# Check if ~/.cache is a symlink
if [ ! -L ~/.cache ]; then
    echo "Error: ~/.cache is not a symlink. Disable this check at your own risk."
    exit 1
fi

model_family="${1:-}"
model_size="${2:-}"
revision="${3:-}"
layer="${4:-}"

if [[ -z "$model_family" || -z "$model_size" || -z "$revision" || -z "$layer" ]]; then
    echo "Usage: $0 <model_family> <model_size> <revision> <evaluation_layer>"
    exit 1
fi

# Create the directory for logs
log_dir="./lcc_logs/${model_family}/${model_size}/${revision}"
mkdir -p "$log_dir"
output_file="${log_dir}/%x_%j.out"
error_file="${log_dir}/%x_%j.err"

SCRIPT="./mteb-harness.py --model_family $model_family --model_size $model_size --revision $revision --evaluation_layer $layer"
srun --output="$output_file" --error="$error_file" singularity run --nv $CONTAINER $SCRIPT
