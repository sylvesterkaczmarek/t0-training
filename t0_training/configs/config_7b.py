"""7B run config, architecture-matched to OLMo-core's `olmo3_7B` model factory
(see configs/olmo3-7B.yaml for the OLMo-core equivalent run).

Same hyperparameters as config_3b.py except model shape and a smaller rank
microbatch for 7B memory safety. Two sequences per rank with four accumulation
steps on 16 GPUs preserve the global batch of 262,144 tokens.

Launch with:
    torchrun --nproc_per_node=4 --nnodes=4 ... scripts/train.py \\
        --config t0_training/configs/config_7b.py
"""

from t0_training.configs.base import RunConfig, resolve_mix
from t0_training.model.config import config_7b
from t0_training.train import TrainingConfig

_DATA_PATHS = resolve_mix("data/mixes/dolma3-140B.txt")

RUN_CONFIG = RunConfig(
    model=config_7b,
    training=TrainingConfig(
        # max_steps left unset as scripts/train.py computes one epoch of the dataset at runtime.
        global_batch_size=262_144,
        rank_microbatch_tokens=4_096,  # two sequences per rank for 7B memory safety
        seq_len=2048,  # sequence_length in olmo3-7B.yaml; model max_seq_len stays 4096
        warmup_steps=100,
        max_lr=1e-3,
        min_lr=1e-4,  # olmo-core CosWithWarmup alpha_f=0.1 -> 0.1 * max_lr
        weight_decay=0.1,
        grad_clip=1.0,
        # OLMo3's own recipe (OLMo-3-1025-7B-pretrain-1.py); guards against
        # logit magnitude drift under bf16. See TrainingConfig.z_loss_multiplier.
        z_loss_multiplier=1e-5,
        log_interval=10,
        save_dir="checkpoints/7b",
        save_interval=1000,
        keep_last_n_checkpoints=3,
        wandb_project="t0-training-7b",
        # Perplexity + hellaswag, matching olmo3-7B.yaml's eval_interval=250.
        eval_interval=250,
        downstream_eval_tasks=["hellaswag"],
        # Local mirror of olmo-data.org's eval shards -- compute nodes have no
        # internet egress, see logs/run1/train_t0_7b-5824463.err.
        eval_mix_base_dir="/lus/lfs1aip2/projects/u6kt/rosie/t0_training/2/t0-training/data",
    ),
    data_paths=_DATA_PATHS,
    num_workers=4,
)
