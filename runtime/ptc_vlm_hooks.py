"""Launcher-side hooks for the NeMo RL VLM GRPO run of area "VLM" (MM-27: wrappers around framework objects at run
time; no NeMo RL source is edited). Imported by run_vlm_nemo.py in the NeMo RL driver process (the image's system
Python, NeMo RL at 4f39fd86289a2ffd5aaf5d202702f7fd7185c924). Adapted from area 6's ptc_cad_hooks.py (copied, not
imported across areas); the SFT sample log is dropped (GRPO only).

1. Atomic cuts with `cut.json` last: `CheckpointManager` storage `publish` (tmp_step_N -> step_N rename,
   checkpoint_storage.py:80-97) is wrapped; after the rename the cut `<out>/cuts/checkpoint-<N>/` is built in
   `<out>/cuts/.checkpoint-<N>.partial/` from hard links to the consolidated HF weights (one or more safetensors
   files plus their index when NeMo RL writes one) and `config.json`, copies of the tokenizer files NeMo RL saved,
   the processor files of the base (`preprocessor_config.json`, `video_preprocessor_config.json`,
   `chat_template.jinja` when NeMo RL did not save them; the vision tower is frozen, so the base's processor is the
   cut's), `sampler.json` (the evaluator's deployment sampler), `base_ref.json`, `training_info.json`; every file
   fsynced; `cut.json` written last; the directory renamed.
2. GRPO consumption comes from the environment's own log (vlm_env_logged.py, `env_calls.jsonl`) joined with NeMo
   RL's per-step `train_data_step<N>.jsonl` (grpo_sync.py:1459); see README "Consumption".
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def _append_fsync(path: Path, lines: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for d in lines:
            f.write(json.dumps(d, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ------------------------------------------------------------------------------------------------ cut writer ---
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt")
PROCESSOR_FILES = ("preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja")
CUT_REQUIRED = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                "preprocessor_config.json", "sampler.json")
NOT_DEPLOYABLE = ("base_ref.json", "training_info.json", "cut.json")


def _fsync_dir(d: Path) -> None:
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_cut(step_dir: Path, step: int, out: Path, base_dir: Path, run_meta: dict) -> Path:
    cuts = out / "cuts"
    cuts.mkdir(exist_ok=True)
    final = cuts / f"checkpoint-{step}"
    partial = cuts / f".checkpoint-{step}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir()
    cons = step_dir / "policy" / "weights" / "model" / "consolidated"
    shards = sorted(cons.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"{cons}: no consolidated safetensors file")
    for s in shards:
        os.link(s, partial / s.name)
    if len(shards) > 1 or (cons / "model.safetensors.index.json").exists():
        if not (cons / "model.safetensors.index.json").exists():
            raise RuntimeError(f"{cons}: {len(shards)} safetensors files and no index")
        os.link(cons / "model.safetensors.index.json", partial / "model.safetensors.index.json")
    os.link(cons / "config.json", partial / "config.json")
    origin = {}
    tok = step_dir / "policy" / "tokenizer"
    for name in TOKENIZER_FILES + PROCESSOR_FILES:
        if (partial / name).exists():
            continue
        # Copied, not linked: NeMo RL keeps the base download's mtimes on these, and ext-collect refuses outputs
        # older than the launch's START (EXT-STAGE; area 6 SFT full-1).
        if (tok / name).exists():
            shutil.copyfile(tok / name, partial / name)
            origin[name] = "nemo_rl policy/tokenizer"
        elif name in PROCESSOR_FILES and (base_dir / name).exists():
            shutil.copyfile(base_dir / name, partial / name)
            origin[name] = "base model (vision tower frozen)"
    sys.path.insert(0, str(HERE))
    import gen_vqa  # noqa: E402  (SAMPLER only; importing it loads no model)
    (partial / "sampler.json").write_text(json.dumps(gen_vqa.SAMPLER, indent=2, sort_keys=True) + "\n")
    base_models = json.loads((base_dir.parent / "MODELS.json").read_text())["models"]
    base = base_models[base_dir.name]
    (partial / "base_ref.json").write_text(json.dumps(
        {"repo": base["repo"], "revision": base["revision"], "files": base["files"],
         "init_checkpoint": run_meta.get("init_checkpoint")}, indent=2, sort_keys=True) + "\n")
    if (step_dir / "training_info.json").exists():
        shutil.copyfile(step_dir / "training_info.json", partial / "training_info.json")
    files = {}
    for f in sorted(partial.iterdir()):
        with open(f, "rb") as fh:
            os.fsync(fh.fileno())
        files[f.name] = sha256_file(f)
    missing = [n for n in CUT_REQUIRED if n not in files]
    if missing:
        raise RuntimeError(f"cut {step}: missing {missing}")
    deployable = [n for n in files if n not in NOT_DEPLOYABLE]
    cut = {"schema": "ptc_mm.vlm_cut", "schema_version": "1.0", "global_step": step, "files": files,
           "deployable": sorted(deployable), "file_origin": origin, "seed": run_meta["seed"],
           "profile": run_meta["profile"], "stage": run_meta["stage"], "script_sha256": run_meta["script_sha256"],
           "examples_consumed": run_meta["examples_consumed"](step) if callable(run_meta.get("examples_consumed"))
           else None,
           "nemo_rl_step_dir": step_dir.name}
    with open(partial / "cut.json", "w", encoding="utf-8") as fh:
        fh.write(json.dumps(cut, indent=2, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    _fsync_dir(partial)
    os.rename(partial, final)
    _fsync_dir(cuts)
    _append_fsync(out / "cuts.jsonl", [{"step": step, "cut": final.name, "utc": _utc()}])
    return final


def install_cut_writer(out: Path, base_dir: Path, run_meta: dict) -> None:
    from nemo_rl.utils import checkpoint as ckpt

    orig_init = ckpt.CheckpointManager.__init__

    def init(self, config):
        orig_init(self, config)
        storage = self._storage
        orig_publish = storage.publish

        def publish(path):
            orig_publish(path)
            tmp = Path(path)
            step = int(tmp.name.split("_")[2])
            write_cut(tmp.parent / f"step_{step}", step, out, base_dir, run_meta)

        storage.publish = publish

    ckpt.CheckpointManager.__init__ = init


# ------------------------------------------------------------------------------------------------ run record ---
def write_run_args(out: Path, doc: dict) -> None:
    import importlib.metadata as md
    pkgs = {}
    for name in ("nemo-rl", "nemo_rl", "torch", "transformers", "vllm", "ray"):
        try:
            pkgs[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    commit = None
    for cand in (Path("/opt/nemo-rl/.git"),):
        head = cand / "HEAD"
        if head.exists():
            ref = head.read_text().strip()
            if ref.startswith("ref:"):
                rp = cand / ref.split()[1]
                commit = rp.read_text().strip() if rp.exists() else None
            else:
                commit = ref
    freeze = sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions())
    (out / "env-freeze.txt").write_text("\n".join(freeze) + f"\n# nemo-rl commit {commit}\n")
    doc = {**doc, "framework": {"name": "nemo_rl", "commit": commit, "versions": pkgs},
           "python": sys.version.split()[0], "created_at": _utc()}
    (out / "run_args.json").write_text(json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n")


def record_attempt(out: Path, event: str, **kw) -> int:
    path = out / "attempts.jsonl"
    prev = [json.loads(x) for x in open(path)] if path.exists() else []
    attempt = 1 + sum(1 for p in prev if p["event"] == "start") if event == "start" else \
        sum(1 for p in prev if p["event"] == "start")
    _append_fsync(path, [{"event": event, "attempt": attempt, "utc": _utc(), "pid": os.getpid(), **kw}])
    return attempt
