#!/usr/bin/env python
"""NeMo RL launcher of area "VLM": stage `vlm_grpo` (`rl.nemo_rl_vlm@0.1`), Qwen3.5-0.8B on CLEVR-CoGenT TrainA.

Runs inside the derived NeMo RL image `gatedtrain-a6-nemo-rl:1` (Tier B `nemo-rl:4f39fd8` plus the vLLM patch targets
handed to uid 1000; NeMo RL commit 4f39fd86289a2ffd5aaf5d202702f7fd7185c924), cwd /opt/nemo-rl, the image's system
interpreter `/opt/nemo_rl_venv/bin/python`. It records the run, installs the hooks of ptc_vlm_hooks.py, registers
the row-id processor and points the `vlm` environment at the logging subclass (vlm_env_logged.py), builds the
overrides of the profile and memory rung and calls the unmodified example entry point
(`examples/run_vlm_grpo.py`, `main()`), so NeMo RL itself parses the overrides as on its command line.

Memory rungs (area 7 decisions, README "Implementation (as run)"; the first whose smoke peaks at <= 22.5 GB is used):
- `a`: TE FusedAdam with fp32 master weights and bf16 moments (as NeMo RL's Qwen3.5 VLM recipe), KL penalty 0,
       vLLM `gpu_memory_utilization` 0.2, activation checkpointing, microbatches of 1 x 1,024 tokens;
- `b`: `a` plus `policy.logprob_chunk_size` 512;
- `c`: `b` plus `policy.automodel_cfg.cpu_offload` (host RAM);
LoRA (the last rung) is not wired: it needs a separate decision first.

Usage (container):
  run_vlm_nemo.py --profile smoke|full --out /out --seed 7100 --base <area>/models/qwen3.5-0.8b --memory a
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ptc_vlm_hooks as hooks  # noqa: E402

DATA = HERE / "data" / "train.jsonl"
PROMPT = HERE / "prompt_clevr_cot.txt"
PROFILES = {
    "smoke": {"steps": 5, "prompts": 16, "generations": 8, "save_period": 5},
    "full": {"steps": 60, "prompts": 16, "generations": 8, "save_period": 20},
}
MAX_SEQ = 1024          # prompt (about 150 image + 110 text tokens, estimate) + at most 512 new tokens
MEMORY = {
    "a": [],
    "b": ["policy.logprob_chunk_size=512"],
    "c": ["policy.logprob_chunk_size=512", "policy.automodel_cfg.cpu_offload=true"],
}


def sha256_file(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def check_images(train_file: Path) -> int:
    """Every training row's image exists and its bytes hash to the row's `image_sha256` (content addressed)."""
    seen = {}
    for line in open(train_file, encoding="utf-8"):
        row = json.loads(line)
        path = Path(row["messages"][0]["content"][0]["image"])
        if path not in seen:
            seen[path] = sha256_file(path)
        if seen[path] != row["image_sha256"]:
            raise SystemExit(f"{row['row_id']}: {path} hashes to {seen[path]}, not {row['image_sha256']}")
    return len(seen)


def overrides(prof: dict, a, out: Path) -> list[str]:
    model = str(a.base)
    n = prof["prompts"] * prof["generations"]
    return [
        f"policy.model_name={model}",
        f"policy.tokenizer.name={model}",
        "cluster.gpus_per_node=1",
        f"policy.max_total_sequence_length={MAX_SEQ}",
        "data.shuffle=false",                                   # file order: observed logs
        "data.train.dataset_name=ResponseDataset",
        f"++data.train.data_path={DATA}",
        "++data.train.split_validation_size=0",
        "data.validation=null",
        f"data.default.prompt_file={PROMPT}",
        "data.default.processor=clevr_rowid",                   # vlm_env_logged.clevr_rowid_processor
        "data.default.env_name=clevr-cogent",                   # created as the `vlm` env (data/utils.py:201)
        "env.clevr-cogent.num_workers=2",
        f"++env.clevr-cogent.log_path={out}/env_calls.jsonl",
        "checkpointing.enabled=true",
        "checkpointing.keep_top_k=null",
        f"checkpointing.checkpoint_dir={out}/nemo/checkpoints",
        f"checkpointing.save_period={prof['save_period']}",
        "checkpointing.save_optimizer=false",                   # cuts need weights only; saves disk and time
        f"logger.log_dir={out}/nemo/logs",
        "logger.wandb_enabled=false",
        "logger.tensorboard_enabled=false",
        "policy.automodel_cfg.checkpoint.save_consolidated=every",
        "policy.automodel_cfg.activation_checkpointing=true",
        "++policy.automodel_cfg.automodel_kwargs.freeze_config.freeze_vision_tower=true",
        "++policy.automodel_cfg.automodel_kwargs.freeze_config.freeze_audio_tower=true",
        "++policy.automodel_cfg.automodel_kwargs.freeze_config.freeze_language_model=false",
        # Torch linear layers and SDPA attention instead of the Transformer Engine defaults on CUDA (Automodel
        # common/utils.py BackendConfig): TE modules add `_extra_state` entries to the state dict, which NeMo RL
        # filters through `exclude_key_regex` (automodel_policy_worker.py:417) but the Qwen3.5 dense adapter's
        # convert_single_tensor_to_hf ignores, so the refit handed them to vLLM and failed (probe-a2).
        "++policy.automodel_cfg.automodel_kwargs.backend._target_="
        "nemo_automodel.components.models.common.utils.BackendConfig",
        "++policy.automodel_cfg.automodel_kwargs.backend.attn=sdpa",
        "++policy.automodel_cfg.automodel_kwargs.backend.linear=torch",
        "++policy.automodel_cfg.automodel_kwargs.backend.rope_fusion=false",
        f"grpo.max_num_steps={prof['steps']}", f"grpo.seed={a.seed}",
        f"grpo.num_prompts_per_step={prof['prompts']}", f"grpo.num_generations_per_prompt={prof['generations']}",
        "grpo.val_period=-1", "grpo.val_at_start=false", "grpo.val_at_end=false",
        f"policy.train_global_batch_size={n}",
        # One sequence (at most 1,024 tokens) per microbatch, as NeMo RL's VLM recipe (vlm_grpo_3B.yaml): the
        # 248,320-wide fp32 logits of 2 x 1,024 tokens peaked at 23.0 GB (probe-a6) and ran out of memory with
        # logprob chunking added (probe-b1).
        "policy.train_micro_batch_size=1",
        # Logprob passes of one sequence too (the recipe's 4 x 1,024 tokens are 4 GB of fp32 logits alone): the
        # worker still held 22-23 GB with micro batch 1, logprob chunking (probe-b2/b3) or allocator settings
        # (probe-b5; expandable_segments breaks the CUDA IPC refit, probe-b4).
        "policy.logprob_batch_size=1",
        "policy.generation.max_new_tokens=512",
        "policy.generation.vllm_cfg.gpu_memory_utilization=0.2",
        # Qwen3.5's linear-attention layers need one vLLM Mamba cache block per decoding sequence; at 0.2 vLLM's
        # default max_num_seqs 256 exceeds the 142 blocks available (probe-a1). One step decodes 16 x 8 = 128.
        "++policy.generation.vllm_kwargs.max_num_seqs=128",
        "loss_fn.reference_policy_kl_penalty=0.0",               # no reference-model logprob pass (memory)
        # With KL 0 the loss never reads reference logprobs (loss_functions.py:762), but unless the pass is skipped
        # the data-plane trainer still asks for that column (probe-a3, probe-a5: KeyError), hence
        # data_plane.enabled=false below; NeMo RL allows the skip only with KL 0.
        "++grpo.skip_reference_policy_logprobs_calculation=true",
        # Memory-lean optimizer of NeMo RL's Qwen3.5 VLM recipe (vlm_grpo-qwen3.5-35ba3b-geo3k-...-ep16.yaml:25-33).
        "policy.optimizer.name=transformer_engine.pytorch.optimizers.fused_adam.FusedAdam",
        # Peak lr 2e-6 reached after 10 steps (the base scheduler warms up from 0.1x over 50 steps, so a 60-step run
        # would train at well under 1e-6 for most of its steps; probe-a8: 5 smoke steps at about 1e-7). Below the
        # 5e-6 of NeMo RL's base GRPO recipe; the Qwen3.5 VLM recipe uses 1e-6 with far larger batches.
        "policy.optimizer.kwargs.lr=2.0e-6",
        "policy.scheduler.0.kwargs.total_iters=10",
        "policy.scheduler.2.milestones=[10]",
        "++policy.optimizer.kwargs.master_weights=true",
        "++policy.optimizer.kwargs.store_param_remainders=true",
        "++policy.optimizer.kwargs.exp_avg_dtype=torch.bfloat16",
        "++policy.optimizer.kwargs.exp_avg_sq_dtype=torch.bfloat16",
        # The classic synchronous trainer (grpo.py grpo_train; data_plane/factory.py select_sync_trainer), not the
        # data-plane one of Tier B and area 6: the data-plane training fetch always asks for reference logprobs
        # (grpo_sync.py _train_fields_for_step drops only prev_logprobs), so with the reference pass skipped it
        # failed (probe-a5). grpo.py also writes train_data_step<N>.jsonl (grpo.py:4690).
        "data_plane.enabled=false",
        *MEMORY[a.memory],
    ]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--profile", choices=tuple(PROFILES), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--base", type=Path, required=True, help="base model directory (models/<name>)")
    p.add_argument("--memory", choices=tuple(MEMORY), required=True)
    p.add_argument("--max-steps", type=int)
    a = p.parse_args(argv)
    prof = dict(PROFILES[a.profile])
    if a.max_steps:
        prof["steps"] = a.max_steps
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    attempt = hooks.record_attempt(out, "start", stage="vlm_grpo", profile=a.profile)
    n_images = check_images(DATA)
    ov = overrides(prof, a, out)
    run_meta = {"seed": a.seed, "profile": a.profile, "stage": "vlm_grpo", "init_checkpoint": None,
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "examples_consumed": lambda s: s * prof["prompts"]}
    hooks.write_run_args(out, {
        "stage": "vlm_grpo", "profile": a.profile, "seed": a.seed, "dry_run": False, "max_steps": prof["steps"],
        "profile_settings": prof, "memory_rung": a.memory, "model_name_or_path": str(a.base),
        "base_model": str(a.base), "init_checkpoint": None, "train_file": str(DATA),
        "train_file_sha256": sha256_file(DATA), "train_images_checked": n_images,
        "prompt_file_sha256": sha256_file(PROMPT), "nemo_rl_overrides": ov, "attempt": attempt,
        "script_sha256": run_meta["script_sha256"], "hooks_sha256": sha256_file(HERE / "ptc_vlm_hooks.py"),
        "env_sha256": sha256_file(HERE / "vlm_env_logged.py"),
        "seed_derivation": "NeMo RL grpo.seed (set_seed in setup); vLLM sampling seeded by NeMo RL; "
                           "data order = file order (data.shuffle=false)"})
    hooks.install_cut_writer(out, a.base, run_meta)

    from nemo_rl.data.processors import register_processor
    from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
    from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
    from nemo_rl.environments.utils import ENV_REGISTRY
    import vlm_env_logged
    os.environ["PYTHONPATH"] = f"{HERE}:{os.environ.get('PYTHONPATH', '')}"   # Ray workers import vlm_env_logged
    register_processor("clevr_rowid", vlm_env_logged.clevr_rowid_processor)
    fqn = "vlm_env_logged.LoggedVLMEnvironment"
    ACTOR_ENVIRONMENT_REGISTRY[fqn] = PY_EXECUTABLES.SYSTEM
    ENV_REGISTRY["vlm"] = {"actor_class_fqn": fqn}
    sys.path.insert(0, "/opt/nemo-rl/examples")
    import run_vlm_grpo as entry  # noqa: E402
    cfg = "/opt/nemo-rl/examples/configs/vlm_grpo_3B.yaml"
    sys.argv = [entry.__file__, "--config", cfg, *ov]
    try:
        entry.main()
    finally:
        hooks.record_attempt(out, "end", stage="vlm_grpo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
