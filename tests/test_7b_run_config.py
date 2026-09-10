"""Regression coverage for the 7B training job's memory budget."""

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from scripts import train
from t0_training.configs import base
from t0_training.train import Trainer

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "t0_training/configs/config_7b.py"
JOB_PATH = ROOT / "batch/7b/train_t0.sh"


@pytest.fixture
def run_config(tmp_path, monkeypatch):
    # Two global batches plus a partial batch, in the real headerless format.
    shard = tmp_path / "tokens.npy"
    np.zeros(2 * 262_144 + 2048 + 1, dtype=np.uint32).tofile(shard)
    monkeypatch.setattr(base, "resolve_mix", lambda *args, **kwargs: [str(shard)])
    config = train.load_config(str(CONFIG_PATH))
    config.num_workers = 0
    return config


def test_7b_config_preserves_global_batch_and_automatic_steps(run_config):
    config = run_config.training
    assert config.rank_microbatch_tokens == 4096
    assert config.global_batch_size == 262_144
    assert config.seq_len == 2048
    assert config.max_steps is None


def test_7b_trainer_accumulates_four_microbatches_on_16_ranks(run_config, monkeypatch):
    config = replace(
        run_config.training, save_dir=None, eval_interval=None, wandb_project=None
    )
    loader = train.build_data_loader(run_config, rank=0, world_size=16)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # Only initialize the trainer; no CUDA allocation or training is performed.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    trainer = Trainer(model, optimizer, loader, config, world_size=16, rank=0)

    assert loader.batch_size == trainer.microbatch_size == 2
    assert trainer.grad_accum_steps == 4
    assert trainer.tokens_per_step == 262_144


def test_7b_entrypoint_computes_steps_from_dataset(run_config, monkeypatch):
    assert run_config.training.max_steps is None
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(CONFIG_PATH)])
    monkeypatch.setattr(train, "init_distributed", lambda: (16, 0, 0))
    monkeypatch.setattr(train, "load_config", lambda path: run_config)
    monkeypatch.setattr(train, "Transformer", Mock())
    monkeypatch.setattr(train, "compile_model", lambda model: model)
    monkeypatch.setattr(train, "wrap_model_fsdp", lambda model: model)
    monkeypatch.setattr(train, "build_optimizer", Mock())
    trainer_class = Mock()
    monkeypatch.setattr(train, "Trainer", trainer_class)

    train.main()

    # The real loader sees 257 sequences; 128 sequences make one optimizer step.
    assert run_config.training.max_steps == 2
    assert trainer_class.call_args.args[3] is run_config.training
    trainer_class.return_value.train.assert_called_once()


def test_7b_job_exports_expandable_segments(tmp_path):
    (tmp_path / ".env").write_text("")
    env = os.environ.copy()
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.pop("PYTORCH_ALLOC_CONF", None)
    env.update(SLURM_JOB_NODELIST="node0", SLURM_NNODES="4", SLURM_PROCID="0")
    # Stub cluster commands and inspect the environment inherited by a child.
    wrapper = r"""
module() { :; }
scontrol() { printf 'node0\n'; }
srun() {
    if [[ "$1" == "--nodes=1" ]]; then
        printf '127.0.0.1\n'
    else
        bash -c 'printf "ALLOC_CONF=%s\n" "${PYTORCH_CUDA_ALLOC_CONF-}"'
    fi
}
source "$1"
"""
    result = subprocess.run(
        ["bash", "-c", wrapper, "test-launcher", str(JOB_PATH)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert "ALLOC_CONF=expandable_segments:True" in result.stdout
