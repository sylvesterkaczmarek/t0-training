#!/bin/bash
#SBATCH --job-name=train_t0_7b
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/run1/%x-%j.out
#SBATCH --error=logs/run1/%x-%j.err

# Train the 7B model with OUR implementation (scripts/train.py), mirroring the
# olmo-core reference run in batch/7b/train_clean.sh: same node count (4x4
# GH200), so global_batch_size 262144 / (4096 tokens/rank * 16 ranks) = 4
# grad-accum steps. Resumes from the latest checkpoint in save_dir if present.

RUN=${RUN:-run1}

module load cuda/12.6
module load gcc-native/12.3
module load brics/nccl/2.26.6-1
module load brics/aws-ofi-nccl

source .env
export WANDB_API_KEY

# Reduce allocator fragmentation in the 7B run.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Slingshot / NCCL settings for cross-node communication via libfabric/CXI
export FI_PROVIDER=cxi
export FI_CXI_DISABLE_CQ_HUGETLB=1
export FI_CXI_RX_MATCH_MODE="hybrid"
export NCCL_NET_FORCE_FLUSH="0"
export NCCL_CROSS_NIC="1"

export NCCL_DEBUG=WARN
export NCCL_DEBUG_FILE=logs/${RUN}/nccl-%h.%p.log

MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w $MASTER_HOST hostname -i | tr -d ' ')
echo "MASTER_HOST: $MASTER_HOST"
echo "MASTER_ADDR from hostname -i: $MASTER_ADDR"
# Fall back to hsn0 suffix if hostname -i didn't return an IP
if [[ ! "$MASTER_ADDR" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    MASTER_ADDR="${MASTER_HOST}-hsn0"
    echo "MASTER_ADDR falling back to: $MASTER_ADDR"
fi
MASTER_PORT=29500

srun bash -c "uv run --no-sync torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --node_rank=\$SLURM_PROCID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train.py --config t0_training/configs/config_7b.py --run-name $RUN"
