# Plan: From our 7B pretraining to an OLMo 3 Think/Instruct reproduction

## Goal

We can't afford OLMo 3's full 7B pretraining (5.93T tokens, about 42× Chinchilla). Rough cost: ~2.6e23 FLOPs, about 180K GH200-hours at 40% MFU, or roughly a month on 64 Isambard nodes. Instead we want to be able to say:

> "Our pipeline reproduces OLMo 3 pretraining. The released `allenai/Olmo-3-1025-7B` base is what our stack would have produced with more compute. We therefore start post-training from that checkpoint."

**Part A** lists the evidence needed to make that claim. **Part B** covers the post-training we then run ourselves: SFT, DPO and RL.

Sources: [OLMo 3 paper](https://arxiv.org/abs/2512.13961), [open-instruct `scripts/train/olmo3/`](https://github.com/allenai/open-instruct/tree/main/scripts/train/olmo3), [OLMo-core SFT README](https://github.com/allenai/OLMo-core/tree/main/src/scripts/train/sft).

---

## Current state vs OLMo 3 7B stage 1

| | Ours ([config_7b.py](../t0_training/configs/config_7b.py)) | OLMo 3 7B stage 1 |
|---|---|---|
| Architecture | Matched (SWA 3/4 layers, QK-norm, RoPE θ=5e5) | same |
| Tokens | 140B (`dolma3-140B` mix) | 5.93T (`dolma3_mix-6T-1025-7B`) |
| Seq len | 2048 | 8192 |
| Global batch | 262K tokens | ~4.19M tokens (512 × 8192) |
| Peak LR | 1e-3 → 1e-4 cosine | 3e-4, cosine over 5T, stretched to 5.93T |

The current run shows our stack is **stable** at 7B. It does **not** show that it reproduces OLMo 3, because the recipe differs. Checks A1–A4 close that gap.

AI2 released checkpoints that make this possible. Revisions of `allenai/Olmo-3-1025-7B` on Hugging Face:
- `stage1-step{0..1413814}`: every 1000 steps, ~4.2B tokens apart. ~140B tokens ≈ `stage1-step33000`.
- `stage2-step*`: midtraining, 100B Dolmino tokens.
- `stage3-step*`: long context, 50B Longmino tokens.
- `main`: the final base model.

HF revisions contain **weights only, not optimizer state**.

---

## Part A: Pretraining validation

Compute estimates assume ~400 TFLOP/s effective per GH200 (≈40% MFU) and are rough. Isambard nodes have 4 GPUs each.

### A1. Architecture parity (logit match)
- Convert HF `allenai/Olmo-3-1025-7B` (and one `stage1-*` revision) into our `t0_training/model/transformer.py` format.
- Run the same batch of token IDs through our model and through HF `transformers`. Assert max-abs logit difference is within bf16 tolerance, and that per-token losses are equal.
- Also test at seq len > 4096, so the sliding-window layers are actually exercised.
- **Deliverable:** `scripts/convert_hf_to_t0.py` plus a pytest.
- **Compute:** < 1 GPU-hour.

### A2. Training-dynamics parity (the core evidence)
- New config `config_7b_olmo3_parity.py`. Copy every hyperparameter from OLMo-core's official `src/scripts/official/OLMo3/OLMo-3-1025-7B-pretrain-1.py`: seq 8192, batch ~4.19M tokens, LR 3e-4, same warmup, weight decay, z-loss and data mix.
- Train from step 0 for **~10–20B tokens**, about 2.5K–5K steps.
- Compare train loss and in-loop eval against AI2's released W&B curves and against `stage1-step1000…5000` evaluated offline on the same eval shards.
- Our data order won't be identical, so expect a **statistical match** (same curve within noise), not bit-exact.
- **Optional control:** run the official OLMo-core script on Isambard for the same budget. We already have the OLMo-core pipeline. This separates "our stack differs" from "Isambard differs".
- **Compute:** 20B tokens ≈ 8.8e20 FLOPs ≈ **~600 GPU-hours**, about 10 h on 16 nodes. Add the same again for the optional control.

### A3. Continuation from an AI2 checkpoint
- Load `stage1-step33000` (~138B tokens) into our trainer, with a fresh optimizer since HF has no optimizer state, and a short re-warmup.
- Continue for ~2–5B tokens with their LR at that point in the schedule.
- **Pass:** loss continues smoothly from their curve after the re-warmup transient.
- **Tests:** checkpoint loading, data loader and optimizer together, on a mature model.
- **Compute:** ~150 GPU-hours.

### A4. Small-scale midtraining (anneal)
- AI2 publishes a 10B version of the midtraining mix: [`allenai/dolma3_dolmino_mix-10B-1025`](https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-10B-1025).
- Run a 10B-token linear anneal to LR 0 from:
  - (a) our 140B checkpoint
  - (b) AI2's `stage1-step33000`
- Compare base-eval gains on OLMES tasks: MMLU, GSM8K, HumanEval, ARC, and bits-per-byte on code and math.
- **Caveat:** our run has already decayed to min_lr, while theirs is at high LR. So (b) will gain more from annealing. Compare *final* scores as well as deltas.
- This also gives us the midtraining capability, and the weight-averaging ("souping") script used for OLMo 3 midtraining.
- **Compute:** 2 × 10B tokens ≈ **~600 GPU-hours**.

### A5. Evaluation harness
Our in-loop eval is currently HellaSwag only.
- Add OLMES ([allenai/olmes](https://github.com/allenai/olmes)) offline evaluation for HF-format checkpoints, using a subset of OlmoBaseEval.
- Part B needs the same harness.

**Exit criterion for Part A:**
- A1 passes.
- A2's loss curve is within noise of AI2's.
- A3 continues smoothly.
- A4 gives comparable eval scores.

Total Part A: **~1.5K GPU-hours**, plus ~0.6K if we run the OLMo-core control.

---

## Part B: Post-training from `allenai/Olmo-3-1025-7B`

**Starting checkpoint:** `main`, the final base after long-context extension. This is what AI2's 7B Think SFT started from. If we skip long context, use `stage2-step47684`, but results won't be comparable to AI2's.

**Toolchain:** AI2 split post-training across two codebases:
- **SFT** in **OLMo-core** (~8× more efficient than open-instruct, per AI2).
- **DPO and RL** in **open-instruct** (Hugging Face / DeepSpeed / vLLM / Ray).

**Decoupling:** each stage can be validated independently, starting from **AI2's released checkpoint for the previous stage**:

| Stage | AI2 reference output | Can start from AI2 checkpoint |
|---|---|---|
| Think SFT | `allenai/Olmo-3-7B-Think-SFT` | `allenai/Olmo-3-1025-7B` |
| Think DPO | `allenai/Olmo-3-7B-Think-DPO` | `allenai/Olmo-3-7B-Think-SFT` |
| Think RL | `allenai/Olmo-3-7B-Think` | `allenai/Olmo-3-7B-Think-DPO` |
| Instruct SFT | `allenai/Olmo-3-7B-Instruct-SFT` | Think SFT checkpoint |
| Instruct DPO | `allenai/Olmo-3-7B-Instruct-DPO` | `allenai/Olmo-3-7B-Instruct-SFT` |
| Instruct RL | `allenai/Olmo-3-7B-Instruct` | `allenai/Olmo-3-7B-Instruct-DPO` |
| RL-Zero | `allenai/Olmo-3-7B-RL-Zero-{Math,Code,IF,Mix}` | `allenai/Olmo-3-1025-7B` |

**Suggested order:**
1. Instruct SFT (cheapest).
2. Think SFT.
3. DPO from AI2's SFT checkpoints.
4. RL last, since it has the most infrastructure.

**Chat templates are the main source of silent errors.** Before tokenizing, follow open-instruct's [`docs/olmo3.md`](https://github.com/allenai/open-instruct/blob/main/docs/olmo3.md):
- Use `olmo-3-tokenizer-instruct-dev` for SFT tokenization. This works around a `<think>` masking bug.
- Use the `think-dev` template for Think DPO/RL and evaluation.

GPU counts below are AI2's, on H100 nodes with 8 GPUs. On Isambard (4 × GH200 per node), divide by 4 to get node counts. GPU-hour figures are **our rough estimates**; AI2 doesn't report 7B post-training compute.

### B1. SFT (Think and Instruct)

| | Think SFT | Instruct SFT |
|---|---|---|
| Data | `allenai/Dolci-Think-SFT-7B` (2.27M examples, long traces, ~8K tokens avg → ~18B tokens/epoch) | `allenai/Dolci-Instruct-SFT` (2.15M examples, incl. function calling; much shorter) |
| Init | base `main` | Think SFT checkpoint |
| Recipe | 2 epochs, LR 5e-5, seq 32K, batch 1M tokens | 2 epochs, LR 8e-5, seq 32K, batch 1M tokens |
| AI2 hardware | 64 H100 | 32 H100 |
| Our estimate | **~1.5–2.5K GPU-h** (~1–1.5 days on 16 nodes) | **~200–500 GPU-h** |

AI2 also swept several learning rates in parallel and merged checkpoints. Budget for 2–4× if we sweep.

**Is it one command?** Almost. There are two steps:
1. **Tokenize** with open-instruct's `scripts/data/convert_sft_data_for_olmocore.py`, which writes OLMo-core numpy files. This is CPU-heavy but runs once.
2. **Train** with OLMo-core's `src/scripts/train/sft/Olmo-3-7B-SFT.py train <name> <ckpt>/model_and_optim <cluster> --seq_len=32768 --dataset_path=...` under `torchrun`.

The `launch` subcommand is Beaker-only. We use `train` inside our own Slurm and torchrun wrapper, like `batch/7b/train_t0.sh`. Other adaptations:
- Convert the HF checkpoint to OLMo-core format (OLMo-core ships HF↔OLMo-core converters).
- Retune parallelism for 4-GPU nodes.

This fits the OLMo-core pipeline we already have, so it's low risk.

### B2. DPO with delta learning

| | Think DPO | Instruct DPO |
|---|---|---|
| Data | `allenai/Dolci-Think-DPO-7B` (150K pairs) | `allenai/Dolci-Instruct-DPO` (260K pairs; incl. multi-turn, length-controlled) |
| Recipe | LR **8e-8**, linear, 1 epoch, max len 16K, ZeRO-3 | LR 1e-6, linear, max len 16K, ZeRO-3 |
| AI2 hardware | 32 H100 | 32 H100 |
| Our estimate | **~300–600 GPU-h** | **~300–600 GPU-h** |

For scale: AI2 reports ~18 h on 64 GPUs per job for a full LR sweep at 32B.

**Is it one command?** Yes, apart from launch plumbing. Strip AI2's `mason.py ... --` prefix (Beaker) and keep the payload:

```bash
accelerate launch --mixed_precision bf16 --use_deepspeed \
  --deepspeed_config_file configs/ds_configs/stage3_no_offloading_accelerate.conf \
  --deepspeed_multinode_launcher standard \
  open_instruct/dpo_tune_cache.py --model_name_or_path allenai/Olmo-3-7B-Think-SFT \
  --mixer_list allenai/Dolci-Think-DPO-7B 1.0 --max_seq_length 16384 --learning_rate 8e-8 ...
```

Run one of these per node from Slurm. Remaining work:
- Drop the `--oe_eval_*` flags, which launch Beaker eval jobs, and run OLMES ourselves.
- Check that DeepSpeed and flash-attn build on aarch64 (GH200).
- Input and output are HF-format models, so no conversion is needed.

### B3. RL with verifiable rewards (OlmoRL / `grpo_fast.py`)

| | Think RL | Instruct RL | RL-Zero (per domain) |
|---|---|---|---|
| Data | `allenai/Dolci-Think-RL-7B` (102K prompts: math, code, IF, chat) | `allenai/Dolci-Instruct-RL` (170K prompts) | `allenai/Dolci-RL-Zero-{Math,Code,IF,General,Mix}-7B` |
| Init | Think DPO | Instruct DPO | base `main` |
| Recipe | GRPO, β=0, 64 prompts × 8 samples, LR 1e-6 constant, responses up to **32K** | same; responses up to 8K, active sampling | same; responses up to 16K |
| AI2 hardware | 32 H100 (16 learner + 16 vLLM) | 64 H100 (8 learner + 56 vLLM) | 72 H100 |
| Extra services | **Qwen3-32B LLM judge** (vLLM server), **code-execution API** | same | code API for Code only |
| Our estimate | **~2–5K GPU-h**, open-ended | **~1.5–3K GPU-h** | **~1–3K GPU-h** each |

RL has no natural endpoint. OLMo 3.1 Think 32B came from running RL 3 more weeks, and performance kept improving. We fix a step budget and compare against AI2's intermediate checkpoints and W&B curves.

**Is it one command?** No. This is the main infrastructure effort. The trainer call is one `python open_instruct/grpo_fast.py ...`, but it assumes:
- **A Ray cluster across nodes.** AI2's `configs/beaker_configs/ray_node_setup.sh` is Beaker-specific. We need a Slurm equivalent: `ray start --head` on node 0, workers on the others.
- **vLLM on GH200/aarch64**, co-scheduled with DeepSpeed learners. Learner and engine counts need remapping to 4-GPU nodes.
- **A hosted LLM judge.** Qwen3-32B behind an OpenAI-compatible vLLM endpoint on dedicated nodes (1–2 nodes), reachable from the job. Isambard compute nodes have no internet, so it has to be self-hosted.
- **A code-execution sandbox.** AI2 uses an AWS Lambda URL. We need a self-hosted equivalent reachable from compute nodes, with isolation.
- Removing the Beaker eval hooks (`--try_launch_beaker_eval_jobs_on_weka`, `--oe_eval_*`).

**Recommended first RL target: RL-Zero Math.** It needs no judge and no code sandbox. It starts from the base model, has a small dataset (13K prompts) and a clear metric (AIME pass@32). That lets us bring up Ray and vLLM on Isambard before tackling Think/Instruct RL.

### B4. Post-training evaluation
- Use OLMES with AI2's post-training suite: MMLU-CoT, GPQA, BBH, MATH-500, AIME 24/25, HumanEval+, MBPP+, LiveCodeBench, IFEval, AlpacaEval, ZebraLogic, PopQA. The exact task strings are in the open-instruct scripts.
- Long generations make evaluation expensive: 32K context for Think, pass@32 for AIME. Budget ~50–100 GPU-h per full-suite evaluation of a Think checkpoint.
- **Pass criterion per stage:** our checkpoint is within noise of AI2's released checkpoint for that stage.

---

## Compute summary (rough)

| Block | GPU-hours (GH200) | Notes |
|---|---|---|
| A1–A4 pretraining validation | ~1.5K (+0.6K control) | |
| B1 SFT (Think + Instruct) | ~2–3K | ×2–4 with LR sweeps |
| B2 DPO (Think + Instruct) | ~0.6–1.2K | |
| B3 RL (Think + Instruct) | ~3.5–8K | open-ended; plus judge-server nodes |
| B3 RL-Zero Math (bring-up) | ~1K | |
| Evaluation | ~1K | across all stages |
| **Total** | **~10–15K** | vs ~180K for full pretraining alone |

## Risks and open questions
- **aarch64 builds** for vLLM, DeepSpeed and flash-attn on GH200. Verify these first, since they block B2 and B3.
- **No internet on compute nodes.** Pre-download all HF models and datasets. Self-host the judge and code sandbox.
- **Chat template drift** between SFT, DPO, RL and eval (see B intro).
- **Optimizer state** isn't on HF, so A3 uses a fresh optimizer. Ask AI2 whether raw OLMo-core checkpoints (`model_and_optim`) are publicly downloadable.
- **Poisoning research:** starting from AI2's base means we can't insert pretraining poisons at 7B. Post-hoc poisoning ([posthoc_poison_experiment.md](posthoc_poison_experiment.md)) and SFT/DPO/RL-stage poisoning remain possible. RL-Zero from a clean base is a useful control.
