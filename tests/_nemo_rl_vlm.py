"""Area "VLM" test helpers (verifier-owned; gatedtrain tests/README.md section 11.24, EXT-200 to EXT-213).

Written from the area runtime (`runtime/`: `run_vlm_nemo.py` `PROFILES`, `overrides`,
`main`; `ptc_vlm_hooks.py` `write_cut`, `write_run_args`, `record_attempt`; `vlm_env_logged.py`
`LoggedVLMEnvironment.step`; `gen_vqa.py` `SAMPLER`, `main`; `score_vqa.py` `main`; `make_data.py` rows and items; read
2026-11-14) and from its real outputs, not from gatedtrain:
- `fixtures/vlm/probe-a6/`: the 5-step GPU smoke `out/area7-vlm/probe-a6/train/` (run_args.json, env-freeze.txt,
  env_calls.jsonl, attempts.jsonl, cuts.jsonl byte copies; NeMo RL's train_data_step<N>.jsonl cut down to `rewards` and
  `idx` per line; `cuts/checkpoint-5/` with its small files copied, the 1.7 GB weights and the 20 MB tokenizer.json
  replaced by stand-ins and cut.json's `files` re-hashed for those two) and `data/shards/shard-00000.jsonl`, the first 80
  rows of the real shard 0 (the rows of steps 1 .. 5); host paths replaced by `/srv/ptc/area7-vlm` (config.json
  re-hashed in cut.json);
- `fixtures/vlm/real-arm/`: the first 12 items of the base scoring `out/area7-vlm/gen-base-probe1/score/` (byte
  copies, but `items_sha256` in `env-runtime.json` and `gen_meta.json`, and `gen_meta_sha256`, name the published
  `items-valA/items.jsonl`, whose `image` paths are relative) and `items-valA-first12.jsonl` (its first 12 rows);
  scored before the rename to ptc-hub 0.2.0, so its eval_meta.json keeps the schema id `ptc_skel.item_eval`
  (`renamed_copy`) and its scorer hash is the gate-time one (`SCORE_VQA_GATE_SHA256`);
- `fixtures/vlm/prompt_clevr_cot.txt`: the runtime's prompt file (its SHA-256 is the registered prompt hash).

`VlmScenario(root)` builds the rest from that code: a mirrored host (`GATEDTRAIN_EXT_MIRROR_ROOT` = `<root>/host` stands
for the container's `/`, EXT-112):
- `<root>/host/<AREA7>` the runtime stand-in (AREA7 the adapter's runtime directory, an identity mount): script stand-ins (never executed), the real prompt file,
  `models/qwen3.5-0.8b/` (the 11 file names of the real base, the weights `model.safetensors-00001-of-00001.safetensors`
  with its index) and `models/MODELS.json`, `data/shards/shard-0000{0..3}.jsonl` (48 rows, 12 per shard) and their
  concatenation `data/train.jsonl`, `items-valA/` and `items-valB/` (40 items each; their questions repeat training
  questions on other images);
- `<root>/host/<OUT_ROOT>/area7-vlm/train-1/` a full run (seed 7100, memory rung a, 60 steps of 16 prompts x 8 generations, 3 steps
  per epoch; cuts 20, 40, 60) and four scoring runs (`eval/{base, cand, valb-base, valb-cand}`) as score_vqa.py writes
  them;
- `<root>/`: spec, registration, trainer.yaml, launch record and smoke record.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
from pathlib import Path
from typing import Any, Callable

import yaml

from gatedtrain.testing.ext import ENV_HOST_WORK, ENV_INPUT_ROOTS, ENV_MIRROR_ROOT, ENV_OUTPUT_ROOT, gpu_step_script, launch_label, safetensors_bytes, sha_of, write_canonical
from gatedtrain.testing.support import file_sha256, manifest_entries, manifest_hash, run_cli
from gatedtrain_recipe_nemo_rl_vlm_clevr import rows as _rows

EXT = Path(__file__).resolve().parent / "fixtures"     # the recipe's fixtures (gatedtrain: tests/fixtures/ext)

VLM = EXT / "vlm"
PROBE = VLM / "probe-a6"
REAL_ARM = VLM / "real-arm"
ADAPTER = "nemo_rl_vlm"


def renamed_copy(path: Path) -> Path:
    """Give the record at `path`, a copy (never a fixture itself), the schema id gatedtrain 0.3.0 reads: the
    `ptc_skel.<name>` that ptc-skel 0.1.0 wrote becomes `gatedtrain.<name>` (text replacement of the `schema` value only;
    the file keeps its layout and every other byte). Returns `path`."""
    text = path.read_text(encoding="utf-8")
    new = re.sub(r'("schema": ")ptc_skel\.', r"\1gatedtrain.", text, count=1)
    assert new != text, f"{path}: no ptc_skel schema id"
    path.write_text(new, encoding="utf-8")
    return path


TRAINER_ID = "rl.nemo_rl_vlm@0.1"
EVALUATOR_ID, VALB_ID = "vqa.clevr_items@0.1", "vqa.clevr_valb_items@0.1"
# The adapter's runtime directory (this repository's runtime/ or GATEDTRAIN_RECIPE_NEMO_RL_VLM_CLEVR_RUNTIME): its container
# path is its host path (identity mount), so the scenarios mirror it and the real-file tests read it.
AREA7 = _rows.RUNTIME
OUT_ROOT = "/srv/gatedtrain/out"                                      # the scenarios' output root (mirrored)
OUT_NAME = _rows.OUT_NAME


def real_outputs() -> Path | None:
    """This recipe's real runs under the host's output root (`{output_root}/<OUT_NAME>`), None without a host layout."""
    from gatedtrain.ext.launch import host_config
    from gatedtrain.errors import Refused
    try:
        return Path(host_config().output_root) / OUT_NAME
    except Refused:
        return None


NEMO_RL_COMMIT = "4f39fd86289a2ffd5aaf5d202702f7fd7185c924"
TIERB_IMAGE = "sha256:8cd5b714ea91ec7ea9a976c75b0ae6f6932140cf7a8aea803100de6331a2113a"
IMAGE = "sha256:7a4078e70b28cc6de622fa3cf25c7596c52337fa3f1779488b402c5fa1c74e69"   # gatedtrain-a6-nemo-rl:1 (uid 1000)
EXECUTOR_IMAGE = "sha256:9b0cfb6e125b8986ad562ef18cff5268460b833d046374b292ca164c4558b89a"
TRAIN_PY = "/opt/nemo_rl_venv/bin/python"                   # nemo-rl docker/Dockerfile:227, :621
GEN_PY = "/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python"
BASE_REL = "models/qwen3.5-0.8b"
BASE_REPO, BASE_REVISION = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
BASE_WEIGHTS = "model.safetensors-00001-of-00001.safetensors"   # the real base's (MODELS.json)
CUT_WEIGHTS = "model-00001-of-00001.safetensors"                # NeMo RL's consolidated name (probe-a6)
INDEX = "model.safetensors.index.json"
BASE_FILES = ("LICENSE", "chat_template.jinja", "config.json", "merges.txt", BASE_WEIGHTS, INDEX,
              "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", "video_preprocessor_config.json",
              "vocab.json")
PROMPT_FILE = VLM / "prompt_clevr_cot.txt"
PROMPT_BLOCK_SHA256 = "sha256:d88becf2d8bd00d8ec1678786317304b268d729ebad9e519ca8c5f281cb31d8b"
# gen_vqa.py SAMPLER (every cut's sampler.json, ptc_vlm_hooks.write_cut) and the evaluator's sampler (score_vqa.py:
# SAMPLER plus the backend and the prompt hash).
CUT_SAMPLER = {"strategy": "greedy", "temperature": 0.0, "top_p": 1.0, "max_new_tokens": 512, "samples_per_seed": 1,
               "chat_template": ("checkpoint processor chat template, add_generation_prompt=True, no template kwargs "
                                 "(Qwen3.5 non-thinking)"),
               "messages": ("[{role: user, content: [{type: image}, {type: text, text: "
                            "prompt_clevr_cot.txt.format(question)}]}]"),
               "max_model_len": 1024}
SAMPLER = {**CUT_SAMPLER, "backend": "vllm", "prompt_block_sha256": PROMPT_BLOCK_SHA256}
SCORE_VQA_SHA256 = "sha256:2bb71c3c85df376a87182703f53e91819b54a05e0486634da3030d1b109e1886"
SCORER_CODE = "sha256:7e07e8aec57723e83575ed9f8af29d863c2c8fd58c0f242b1ab92a45a3edf63f"
# The scorer the 2026-11-14 gate and the real arm fixture ran: SCORE_VQA_SHA256 is it with the name map applied.
SCORE_VQA_GATE_SHA256 = "sha256:f527e4f145b56beddbadca4f82bc77a6558f86f93c9e3343711db17c66481727"
SCORER_CODE_GATE = "sha256:f3eaba291f439827fa93115b6aebed5d18e8c5b0054013c87f3292de2719abaf"
SCORER = {"name": "clevr_exact_match_own_parser", "version": "clevr_exact_v1; python 3.13.14",
          "weights_sha256": None, "code_sha256": SCORER_CODE}
SUB_METRICS = ("answer_found", "answer_tagged", "is_yes_no", "n_new_tokens", "truncated")
SEEDS = [11]
SEED = 7100
PROMPTS, GENERATIONS = 16, 8
ENV_NAMES = ["FLASHINFER_CUDA_ARCH_LIST", "HF_HUB_OFFLINE", "HOME", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED",
             "RAY_memory_monitor_refresh_ms", "TORCH_CUDA_ARCH_LIST"]
TRAIN_ENV = {"FLASHINFER_CUDA_ARCH_LIST": "8.9", "HF_HUB_OFFLINE": "1", "HOME": "/tmp/home",
             "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "RAY_memory_monitor_refresh_ms": "0",
             "TORCH_CUDA_ARCH_LIST": "8.9"}
SCORE_ENV = {"HF_HUB_OFFLINE": "1", "HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
SCRIPTS = ("run_vlm_nemo.py", "ptc_vlm_hooks.py", "vlm_env_logged.py", "gen_vqa.py", "score_vqa.py")
QUESTIONS = ("How many cubes are there?", "Is there a red sphere?", "How many objects are left of the cylinder?",
             "Are there more cubes than spheres?", "What number of small things are there?")
ANSWERS = ("0", "1", "2", "3", "yes", "no", "10")


def sha_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, doc: Any) -> Path:
    """The runtime's `json.dumps(..., indent=2, sort_keys=True) + "\\n"`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: Path, docs: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(d, sort_keys=True) + "\n" for d in docs), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def float32(x: float) -> float:
    """The value NeMo RL logs: vlm_environment.py `torch.tensor(results)` (float32)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def compact_sha(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                                 ensure_ascii=False).encode("utf-8")).hexdigest()


def host_config_sha256(host_work: Path, output_root: Path, input_roots: list, *, mirror_root: Path) -> str:
    """EXT-85, EXT-117, EXT-192: the effective layout's hash (no interpreter trees; the image's interpreters)."""
    return sha_of({"host_work": str(host_work), "output_root": str(output_root),
                   "input_roots": [str(r) for r in input_roots], "mirror_root": str(mirror_root),
                   "interpreter": TRAIN_PY, "interpreter_trees": [],
                   "image_interpreters": {"gen_vqa.py": GEN_PY, "run_vlm_nemo.py": TRAIN_PY,
                                          "score_vqa.py": TRAIN_PY}})


# ---------------------------------------------------------------------------
# The runtime directory (make_data.py, fetch_models.py)
# ---------------------------------------------------------------------------

def train_row(k: int) -> dict:
    """make_data.py row: `input` = "<image sha256>\\n<question>", the user message's image and question."""
    img = sha_text(f"train image {k}")
    q = QUESTIONS[k % len(QUESTIONS)]
    return {"image_sha256": img, "input": f"{img}\n{q}",
            "messages": [{"content": [{"image": f"{AREA7}/data/images/{img[7:]}.png", "type": "image"},
                                      {"text": q, "type": "text"}], "role": "user"},
                         {"content": [{"text": ANSWERS[k % len(ANSWERS)], "type": "text"}], "role": "assistant"}],
            "row_id": f"clevr-trainA-{k:05d}",
            "source": {"file": "trainA/data/train-00000-of-00027.parquet", "image_path": f"CLEVR_trainA_{k:06d}.png",
                       "index": k}}


def write_data(d: Path, rows: list[dict], n_shards: int = 4) -> Path:
    """`<d>/shards/shard-0000<i>.jsonl` (compact JSON lines, as make_data.py) and `<d>/train.jsonl`, their
    concatenation; returns the shard directory (the spec's training data)."""
    per = len(rows) // n_shards
    text = ""
    for i in range(n_shards):
        chunk = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows[i * per:(i + 1) * per])
        (d / "shards").mkdir(parents=True, exist_ok=True)
        (d / "shards" / f"shard-{i:05d}.jsonl").write_text(chunk, encoding="utf-8")
        text += chunk
    (d / "train.jsonl").write_text(text, encoding="utf-8")
    return d / "shards"


def write_items(d: Path, split: str, n: int) -> list[dict]:
    """make_data.py items: `text` = "<image sha256>\\n<question>"; the questions repeat the training questions, the
    images never (no exact overlap: another image)."""
    items = []
    for k in range(n):
        img = sha_text(f"{split} image {k}")
        q = QUESTIONS[k % len(QUESTIONS)]
        items.append({"answer": ANSWERS[k % len(ANSWERS)], "image": f"data/images/{img[7:]}.png",
                      "image_sha256": img, "item": f"clevr-{split}-{k:04d}", "question": q,
                      "source": {"file": f"{split}/data/train-00000-of-00002.parquet", "index": k},
                      "text": f"{img}\n{q}"})
    write_jsonl(d / "items.jsonl", items)
    write_json(d / "item_set_meta.json", {"n_items": n, "split": split, "schema": "ptc_mm.vlm_items"})
    return items


def build_base(d: Path) -> Path:
    for name in BASE_FILES:
        p = d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if name == BASE_WEIGHTS:
            p.write_bytes(safetensors_bytes("qwen3.5-0.8b"))
        elif name == INDEX:
            write_json(p, {"metadata": {"total_size": 8}, "weight_map": {"w": BASE_WEIGHTS}})
        elif name.endswith(".json"):
            write_json(p, {"_fixture": name, "architectures": ["Qwen3_5ForConditionalGeneration"]})
        else:
            p.write_text(f"fixture stand-in {name}\n", encoding="utf-8")
    return d


def models_json(base: Path) -> dict:
    """fetch_models.py MODELS.json: files {name: {bytes, sha256}}."""
    return {"models": {base.name: {"bytes": sum((base / n).stat().st_size for n in BASE_FILES),
                                   "files": {n: {"bytes": (base / n).stat().st_size, "sha256": file_sha256(base / n)}
                                             for n in BASE_FILES},
                                   "license": "apache-2.0", "repo": BASE_REPO, "revision": BASE_REVISION}},
            "schema": "ptc_mm.vlm_models", "schema_version": "1.0"}


# ---------------------------------------------------------------------------
# Cuts (ptc_vlm_hooks.write_cut) and identity
# ---------------------------------------------------------------------------

def build_cut(run: Path, step: int, *, profile: str, seed: int, script_sha256: str, base: Path, tag: str = "") -> Path:
    """`<run>/nemo/checkpoints/step_<N>/` (NeMo RL's resume state, never a cut) and `<run>/cuts/checkpoint-<N>/` as
    write_cut builds it: weights and index and config hard-linked, tokenizer files copied, the base's processor files,
    sampler.json, base_ref.json, training_info.json, `cut.json` last; `cuts.jsonl` appended."""
    step_dir = run / "nemo" / "checkpoints" / f"step_{step}"
    cons = step_dir / "policy" / "weights" / "model" / "consolidated"
    cons.mkdir(parents=True)
    (cons / CUT_WEIGHTS).write_bytes(safetensors_bytes(f"vlm step {step} {tag}"))
    write_json(cons / INDEX, {"metadata": {"total_size": 8}, "weight_map": {"w": CUT_WEIGHTS}})
    write_json(cons / "config.json", {"_fixture": "config", "architectures": ["Qwen3_5ForConditionalGeneration"]})
    tok = step_dir / "policy" / "tokenizer"
    tok.mkdir(parents=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (tok / name).write_text(f"tokenizer stand-in {name}\n", encoding="utf-8")
    (step_dir / "training_info.json").write_text(json.dumps({"consumed_samples": step * PROMPTS, "current_step": step}),
                                                 encoding="utf-8")
    d = run / "cuts" / f"checkpoint-{step}"
    d.mkdir(parents=True)
    for name in (CUT_WEIGHTS, INDEX, "config.json"):
        os.link(cons / name, d / name)
    origin = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        shutil.copyfile(tok / name, d / name)
        origin[name] = "nemo_rl policy/tokenizer"
    for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
        shutil.copyfile(base / name, d / name)
        origin[name] = "base model (vision tower frozen)"
    (d / "sampler.json").write_text(json.dumps(CUT_SAMPLER, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    entry = models_json(base)["models"][base.name]
    write_json(d / "base_ref.json", {"repo": BASE_REPO, "revision": BASE_REVISION, "files": entry["files"],
                                     "init_checkpoint": None})
    shutil.copyfile(step_dir / "training_info.json", d / "training_info.json")
    files = {f.name: file_sha256(f) for f in sorted(d.iterdir())}
    write_json(d / "cut.json", {"schema": "ptc_mm.vlm_cut", "schema_version": "1.0", "global_step": step, "files": files,
                                "deployable": sorted(n for n in files if n not in ("base_ref.json", "training_info.json")),
                                "file_origin": origin, "seed": seed, "profile": profile, "stage": "vlm_grpo",
                                "script_sha256": script_sha256, "examples_consumed": step * PROMPTS,
                                "nemo_rl_step_dir": f"step_{step}"})
    with open(run / "cuts.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"cut": d.name, "step": step, "utc": "2026-11-14T10:00:00.000000Z"}, sort_keys=True) + "\n")
    return d


def deploy_hash(cut: Path) -> str:
    """MM-17: the manifest hash of the cut's `cut.json` `deployable` files (full weights, no composition)."""
    deploy = json.loads((cut / "cut.json").read_text(encoding="utf-8"))["deployable"]
    entries = [e for e in manifest_entries(cut) if e["path"] in deploy]
    from gatedtrain.testing.support import canonical_bytes
    return "sha256:" + hashlib.sha256(canonical_bytes(entries)).hexdigest()


# ---------------------------------------------------------------------------
# The training run (run_vlm_nemo.py + ptc_vlm_hooks.py + vlm_env_logged.py)
# ---------------------------------------------------------------------------

def run_args(*, profile: str, seed: int, steps: int, save_period: int, runtime: Path, script_sha: str,
             memory: str = "a") -> dict:
    """run_vlm_nemo.py `main` -> ptc_vlm_hooks.write_run_args (overrides abridged; the two the adapter needs kept)."""
    data = f"{AREA7}/data/train.jsonl"
    prof = {"steps": steps, "prompts": PROMPTS, "generations": GENERATIONS, "save_period": save_period}
    ov = [f"policy.model_name={AREA7}/{BASE_REL}", "data.shuffle=false", "data.train.dataset_name=ResponseDataset",
          f"++data.train.data_path={data}", "data.default.processor=clevr_rowid",
          "++env.clevr-cogent.log_path=/out/env_calls.jsonl", f"checkpointing.save_period={save_period}",
          "logger.log_dir=/out/nemo/logs", f"grpo.max_num_steps={steps}", f"grpo.seed={seed}",
          "grpo.num_prompts_per_step=16", "grpo.num_generations_per_prompt=8", "data_plane.enabled=false"]
    return {"stage": "vlm_grpo", "profile": profile, "seed": seed, "dry_run": False, "max_steps": steps,
            "profile_settings": prof, "memory_rung": memory, "model_name_or_path": f"{AREA7}/{BASE_REL}",
            "base_model": f"{AREA7}/{BASE_REL}", "init_checkpoint": None, "train_file": data,
            "train_file_sha256": file_sha256(runtime / "data" / "train.jsonl"), "train_images_checked": 48,
            "prompt_file_sha256": file_sha256(runtime / "prompt_clevr_cot.txt"), "nemo_rl_overrides": ov,
            "attempt": 1, "script_sha256": script_sha, "hooks_sha256": file_sha256(runtime / "ptc_vlm_hooks.py"),
            "env_sha256": file_sha256(runtime / "vlm_env_logged.py"),
            "seed_derivation": "NeMo RL grpo.seed (set_seed in setup); vLLM sampling seeded by NeMo RL; "
                               "data order = file order (data.shuffle=false)",
            "framework": {"name": "nemo_rl", "commit": NEMO_RL_COMMIT,
                          "versions": {"nemo-rl": "0.8.0+4f39fd862", "torch": "2.13.0+cu130", "vllm": "0.29.0"}},
            "python": "3.13.14", "created_at": "2026-11-14T09:00:00.000000Z"}


def reward_of(step: int, j: int, m: int) -> float:
    """NeMo RL's VLM reward 0.2 x format (0, 0.25, 0.75, 1) + 0.8 x exact (0, 1), as the float32 tensor value."""
    fmt = (0.0, 0.25, 0.75, 1.0)[(m + j) % 4]
    exact = 1.0 if (m + step) % 3 == 0 and fmt == 1.0 else 0.0
    return float32(0.2 * fmt + 0.8 * exact)


def grpo_logs(rows: list[dict], steps: int) -> tuple[list[dict], dict[int, list[dict]]]:
    """LoggedVLMEnvironment.step lines (one call per step, the step's 16 rows in file order, each 8 times in a row) and
    NeMo RL's train_data_step<N>.jsonl lines (rewards in the same order)."""
    spe = len(rows) // PROMPTS
    env, data = [], {}
    for n in range(1, steps + 1):
        start = ((n - 1) % spe) * PROMPTS
        lines = []
        for j in range(PROMPTS):
            r = rows[start + j]
            for m in range(GENERATIONS):
                resp = f"<think>step {n} row {j} sample {m}</think> <answer>{m}</answer>"
                lines.append({"call": n, "position": len(lines), "response_sha256": sha_text(resp),
                              "reward": reward_of(n, j, m), "row_id": r["row_id"], "utc": "2026-11-14T11:00:00.000000Z"})
        env += lines
        data[n] = [{"content": ["stand-in"], "rewards": [x["reward"]], "input_lengths": [374], "token_ids": [[1, 2]],
                    "token_loss_mask": [[0, 1]], "sample_loss_mask": [1.0], "advantages": [[0.0, 0.0]],
                    "generation_logprobs": [[0.0, 0.0]], "prev_logprobs": [[0.0, 0.0]], "idx": i}
                   for i, x in enumerate(lines)]
    return env, data


def build_run(out: Path, runtime: Path, *, profile: str = "full", steps: int = 60, save_period: int = 20,
              seed: int = SEED) -> Path:
    out.mkdir(parents=True)
    script_sha = file_sha256(runtime / "run_vlm_nemo.py")
    (out / "env-freeze.txt").write_text("nemo-rl==0.8.0+4f39fd862\ntorch==2.13.0+cu130\ntransformers==5.16.1\n"
                                        f"# nemo-rl commit {NEMO_RL_COMMIT}\n", encoding="utf-8")
    write_jsonl(out / "attempts.jsonl", [{"attempt": 1, "event": "start", "pid": 1, "profile": profile,
                                          "stage": "vlm_grpo", "utc": "2026-11-14T09:00:00.000000Z"},
                                         {"attempt": 1, "event": "end", "pid": 1, "stage": "vlm_grpo",
                                          "utc": "2026-11-14T10:00:00.000000Z"}])
    write_json(out / "run_args.json", run_args(profile=profile, seed=seed, steps=steps, save_period=save_period,
                                               runtime=runtime, script_sha=script_sha))
    rows = read_jsonl(runtime / "data" / "train.jsonl")
    env, data = grpo_logs(rows, steps)
    write_jsonl(out / "env_calls.jsonl", env)
    logs = out / "nemo" / "logs" / "exp_001"
    logs.mkdir(parents=True)
    for n, lines in data.items():
        (logs / f"train_data_step{n}.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines), encoding="utf-8")
    base = runtime / BASE_REL
    for k in range(save_period, steps + 1, save_period):
        build_cut(out, k, profile=profile, seed=seed, script_sha256=script_sha, base=base)
    return out


# ---------------------------------------------------------------------------
# Scoring runs (gen_vqa.py gen_meta.json, then score_vqa.py)
# ---------------------------------------------------------------------------

def baseline_kind(i: int) -> str:
    return ("right", "wrong", "none", "untagged", "wrong")[i % 5]


def candidate_kind(i: int) -> str:
    return "none" if i % 10 == 2 else ("wrong" if i % 7 == 1 else "right")


WRONG = {"0": "4", "1": "5", "2": "6", "3": "7", "yes": "no", "no": "yes", "10": "9"}


def response(kind: str, answer: str) -> str:
    if kind == "right":
        return f"<think>I count the objects.</think> <answer>{answer}</answer>"
    if kind == "untagged":
        return f"Looking at the scene, the answer is {answer}."
    if kind == "wrong":
        return f"<think>Two or three?</think> <answer>{WRONG[answer]}</answer>"
    return "I cannot tell from the image."


def parse(text: str) -> tuple[str | None, int]:
    """score_vqa.py parse_answer, for the fixture's responses (answer, answer_tagged)."""
    import re
    words = "zero one two three four five six seven eight nine ten".split()
    tok = re.compile(r"\b(\d+|" + "|".join(words) + r"|yes|no)\b", re.I)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    blocks = re.findall(r"<answer>(.*?)</answer>", text, re.S | re.I)
    toks = tok.findall(blocks[-1] if blocks else text)
    if not toks:
        return None, int(bool(blocks))
    t = toks[-1].lower()
    return (str(words.index(t)) if t in words else str(int(t)) if t.isdigit() else t), int(bool(blocks))


def gen_meta(model_dir: Path, items_dir: Path, n_items: int) -> dict:
    """gen_vqa.py gen_meta.json of a vLLM generation of `model_dir` (a base directory or a cut at /inputs/cut)."""
    cut = None
    if (model_dir / "cut.json").is_file():
        cut = {"global_step": json.loads((model_dir / "cut.json").read_text())["global_step"],
               "cut_json_sha256": file_sha256(model_dir / "cut.json")}
    names = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt",
             "preprocessor_config.json", "video_preprocessor_config.json", INDEX)
    deploy = sorted(p.name for p in model_dir.iterdir() if p.name in names or p.name.endswith(".safetensors"))
    return {"backend": "vllm", "model_dir": "/inputs/cut", "cut": cut,
            "model_files": {f: file_sha256(model_dir / f) for f in deploy}, "sampler": CUT_SAMPLER,
            "eval_seeds": list(SEEDS), "items": n_items, "items_sha256": file_sha256(items_dir / "items.jsonl"),
            "limit": None, "prompt_block_sha256": PROMPT_BLOCK_SHA256, "script_sha256": "sha256:" + "4" * 64,
            "versions": {"vllm": "0.29.0", "transformers": "5.12.1", "dtype": "bfloat16"}, "wall_seconds": 60.0,
            "seconds_per_response": 0.5}


def build_score(out: Path, items: list[dict], items_dir: Path, kind_fn: Callable[[int], str], *, model: str | None,
                item_set: str | None, model_dir: Path, evaluator_id: str = EVALUATOR_ID) -> Path:
    """`<out>/outputs/<item>__r0.json`, `scores.jsonl`, `summary.json`, `env-runtime.json`, `gen_meta.json` (copied from
    the generation run) and `arm-exact_match/` (hard-linked outputs, MM-1 items.jsonl, MM-2 eval_meta.json) as
    score_vqa.py (sha256 2bb71c3c...) writes them, in the image (Python 3.13.14)."""
    (out / "outputs").mkdir(parents=True)
    write_json(out / "gen_meta.json", gen_meta(model_dir, items_dir, len(items)))
    rows = []
    for i, it in enumerate(items):
        text = response(kind_fn(i), it["answer"])
        ans, tagged = parse(text)
        ref = it["answer"]
        score = float(ans == ref)
        sub = {"answer_tagged": tagged, "answer_found": int(ans is not None), "truncated": 0,
               "n_new_tokens": 20 + i % 7, "is_yes_no": int(ref in ("yes", "no"))}
        rel = f"outputs/{it['item']}__r0.json"
        write_json(out / rel, {"item": it["item"], "repeat": 0, "seed": 11, "question": it["question"],
                               "reference": ref, "response": text, "parsed_answer": ans, "score": score,
                               "sub_metrics": sub})
        rows.append({"item": it["item"], "repeat": 0, "seed": 11, "error": None, "score": score, "sub_metrics": sub,
                     "output_ref": {"path": rel, "sha256": file_sha256(out / rel)}})
    write_jsonl(out / "scores.jsonl", rows)
    code = {"score_vqa.py": SCORE_VQA_SHA256}
    arm = out / "arm-exact_match"
    arm.mkdir()
    for r in rows:
        dst = arm / r["output_ref"]["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(out / r["output_ref"]["path"], dst)
    write_jsonl(arm / "items.jsonl", rows)
    write_json(arm / "eval_meta.json", {
        "schema": "gatedtrain.item_eval", "schema_version": "1.0", "evaluator_id": evaluator_id, "adapter": ADAPTER,
        "metric": "exact_match", "direction": "higher_is_better", "score_range": [0, 1],
        "checkpoint": {"model_hash": model, "base_model_hash": None}, "normalizer_sha256": None,
        "env_version_sha256": None, "item_set_hash": item_set, "eval_seeds": list(SEEDS), "sampler": SAMPLER,
        "sampler_hash": sha_of(SAMPLER) if model else compact_sha(SAMPLER), "scorer": SCORER,
        "framework": {"name": "vllm", "version": "0.29.0"}, "created_at": "2026-11-14T12:00:00Z"})
    write_json(out / "env-runtime.json", {
        "python": "3.13.14", "executable": TRAIN_PY, "code_sha256": code, "scorer_code_sha256": compact_sha(code),
        "items_sha256": file_sha256(items_dir / "items.jsonl"), "gen_meta_sha256": file_sha256(out / "gen_meta.json"),
        "uid": 1000, "gid": 1000})
    write_json(out / "summary.json", {"items": len(items), "errors": 0})
    return out


# ---------------------------------------------------------------------------
# Spec, registration, records
# ---------------------------------------------------------------------------

REGISTRATION = """registration_version: 1
registration_id: ext-vlm-fixture-superiority-v1
signed_by: prototype-unsigned
workflow: ext-vlm-fixture
gate: quality_gate
clause:
  type: superiority
  metric: exact_match
  direction: higher_is_better
  margin: 0.0
  alpha: 0.025
  power: 0.8
  planning_effect: 0.3
  sigma_plan: fixed
  unit: item
  samples_per_item: 1
  eval_seeds: [11]
  interval: cluster_bootstrap
  blocking: true
margin_justification: Fixture value (superiority, delta_sup = 0).
sigma_pilot:
  sigma: 0.5
  source: Fixture value (n_required = ceil(((1.95996 + 0.84162) x 0.5 / 0.3)^2) = 22 <= 40 items).
baseline_pilot_mean: 0.44
construct_alignment: >-
  exact_match measures answer correctness, the construct the format + exact_alnum reward also rewards, with another
  answer parser; a pass shows more correct answers under an independent parser, not a general gain in visual reasoning.
shared_annotation: >-
  The references are CLEVR program-executor answers like the training answers; no training stage consumed them, so the
  scorer stays T3. Part of a gain may be learning that answer convention.
eval_distribution:
  item_set_hash: "{item_set_hash}"
sampler: {sampler}
evaluators:
  - id: {evaluator}
    manifest_hash: "{evaluator_hash}"
multiplicity:
  plan: single_gate
  family: null
analysis_code_version: gatedtrain 0.3.0
bootstrap:
  replicates: 10000
  seed: 20261114
pairing:
  max_unpaired: 0
min_paired: 10
contamination:
  allow_exact_overlap: false
"""

SPEC = """workflow: ext-vlm-fixture
version: 1
budget: {{gpu_hours: 24}}
artifacts:
  base:         {{type: model_checkpoint, uri: "file:{base}"}}
  train:        {{type: training_dataset, uri: "file:{data}"}}
  items:        {{type: item_set, uri: "file:{items}", disjoint_from: [train]}}
  trainer_cfg:  {{type: trainer_config, uri: "file:trainer.yaml"}}
  registration: {{type: registration, uri: "file:registration.yaml@sha256:{registration_sha256}"}}
stages:
  - id: vlm_grpo
    uses: rl.nemo_rl_vlm@0.1
    algorithm: grpo
    model_family: vlm.qwen3_5@0.1
    inputs: {{model: base, data: train, config: trainer_cfg}}
    outputs: {{model: ckpt}}
  - id: quality_gate
    uses: gate.superiority@1.0
    evaluator: {evaluator}
    sampler: deployment
    inputs: {{model: ckpt, baseline: base, items: items, registration: registration}}
    test: {{metric: exact_match, alpha: 0.025, power: 0.80, mde: 0.3, cluster_by: item, seeds: 1}}
    on_fail: stop
    owner: skeleton-demo
    escalation: [skeleton-demo-lead]
"""


def smoke_record(*, base_model: str, item_set: str, **overrides) -> dict[str, Any]:
    """A PASS smoke.json (MM-38) on the deciding binary metric: 20 baseline scores, mean inside (0, 1)."""
    rec = {"schema": "gatedtrain.smoke_record", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
           "adapter": ADAPTER, "area": 7, "base_model_hash": base_model, "item_set_hash": item_set,
           "peak_gpu_mem_gb": 21.0, "host_ram_peak_gb": 24.0, "swap_peak_gb": 2.0, "seconds_per_step": 45.0,
           "projected_gpu_h": 1.0, "gpu_log_total_h": 6.0,
           "baseline_scores": [1.0 if i % 5 == 0 else 0.0 for i in range(20)], "error_count": 0,
           "verdict": "PASS", "reasons": []}
    rec.update(overrides)
    return rec


class VlmScenario:
    """Area 7 on disk (module docstring)."""

    def __init__(self, root: Path, *, n_rows: int = 48, n_items: int = 40, steps: int = 60, save_period: int = 20,
                 evaluator: str = EVALUATOR_ID):
        self.root = root
        self.mirror = root / "host"
        self.runtime = self.mirror / AREA7.lstrip("/")
        self.output_root = self.mirror / OUT_ROOT.lstrip("/")
        self.host_work = root / "host-work"
        self.evaluator = evaluator
        rt = self.runtime
        rt.mkdir(parents=True)
        for name in SCRIPTS:
            (rt / name).write_text(f"# fixture stand-in for {AREA7}/{name}; never executed\n", encoding="utf-8")
        shutil.copyfile(PROMPT_FILE, rt / "prompt_clevr_cot.txt")
        self.base = build_base(rt / BASE_REL)
        write_json(rt / "models" / "MODELS.json", models_json(self.base))
        self.rows = [train_row(k) for k in range(n_rows)]
        self.data = write_data(rt / "data", self.rows)
        self.items = {"valA": write_items(rt / "items-valA", "valA", n_items),
                      "valB": write_items(rt / "items-valB", "valB", n_items)}
        self.items_dir = {"valA": rt / "items-valA", "valB": rt / "items-valB"}
        gpu_step_script(self.host_work)
        (self.host_work / "gpu-log.md").write_bytes((EXT / "gpu-log.fixture.md").read_bytes())
        runs = self.output_root / "area7-vlm"
        self.out = build_run(runs / "train-1", rt, steps=steps, save_period=save_period)
        self.cut = self.out / "cuts" / f"checkpoint-{steps}"
        self.base_model = manifest_hash(self.base)
        self.cand_model = deploy_hash(self.cut)
        self.item_set_hash = {s: manifest_hash(d) for s, d in self.items_dir.items()}
        ev = runs / "eval"
        self.arms = {
            "valA": (build_score(ev / "base", self.items["valA"], self.items_dir["valA"], baseline_kind,
                                 model=self.base_model, item_set=self.item_set_hash["valA"], model_dir=self.base),
                     build_score(ev / "cand", self.items["valA"], self.items_dir["valA"], candidate_kind,
                                 model=self.cand_model, item_set=self.item_set_hash["valA"], model_dir=self.cut)),
            "valB": (build_score(ev / "valb-base", self.items["valB"], self.items_dir["valB"], baseline_kind,
                                 model=self.base_model, item_set=self.item_set_hash["valB"], model_dir=self.base,
                                 evaluator_id=VALB_ID),
                     build_score(ev / "valb-cand", self.items["valB"], self.items_dir["valB"], candidate_kind,
                                 model=self.cand_model, item_set=self.item_set_hash["valB"], model_dir=self.cut,
                                 evaluator_id=VALB_ID))}
        (root / "trainer.yaml").write_text("backend: nemo_rl_vlm\nprofile: full\n", encoding="utf-8")
        self.render()
        rec = self.launch_record()
        write_canonical(root / "launch.json", rec)
        self.label = rec["label"]
        self.split = "valA" if evaluator == EVALUATOR_ID else "valB"
        write_canonical(root / "smoke.json", smoke_record(base_model=self.base_model,
                                                          item_set=self.item_set_hash[self.split]))
        self.guard_log()

    # --- paths and environment ---

    def env(self) -> dict[str, str]:
        return {ENV_HOST_WORK: str(self.host_work), ENV_OUTPUT_ROOT: str(self.output_root),
                ENV_INPUT_ROOTS: os.pathsep.join(map(str, self.input_roots())), ENV_MIRROR_ROOT: str(self.mirror)}

    def config_sha(self) -> str:
        return host_config_sha256(self.host_work, self.output_root, self.input_roots(), mirror_root=self.mirror)

    def input_roots(self) -> list[Path]:
        """The row's roots, mirrored: the runtime directory and the output root."""
        return [self.runtime, self.output_root]

    def guard_log(self) -> None:
        lines = [f"2026-11-11T09:00:00Z START {self.label} attempt=1", f"2026-11-11T10:00:00Z END {self.label} attempt=1 rc=0"]
        (self.host_work / "gpu-guard.log").write_text("".join(l + "\n" for l in lines), encoding="utf-8")

    def render(self, reg_mutate: Callable[[dict], None] | None = None) -> None:
        split = "valA" if self.evaluator == EVALUATOR_ID else "valB"
        text = REGISTRATION.format(item_set_hash=self.item_set_hash[split], sampler=json.dumps(SAMPLER),
                                   evaluator=self.evaluator,
                                   evaluator_hash=sha_of({"evaluator": self.evaluator, "metric": "exact_match",
                                                          "direction": "higher_is_better", "scorer": SCORER}))
        if reg_mutate is not None:
            reg = yaml.safe_load(text)
            reg_mutate(reg)
            text = yaml.safe_dump(reg, sort_keys=False)
        (self.root / "registration.yaml").write_text(text, encoding="utf-8")
        spec = SPEC.format(base=self.base, data=self.data, items=self.items_dir[split], evaluator=self.evaluator,
                           registration_sha256=file_sha256(self.root / "registration.yaml")[len("sha256:"):])
        (self.root / "spec.yaml").write_text(spec, encoding="utf-8")

    # --- launch ---

    def mounts(self, out: Path) -> list[dict]:
        return [{"host": str(self.runtime), "container": AREA7, "mode": "ro"},
                {"host": str(out), "container": "/out", "mode": "rw"}]

    def argv(self, profile: str = "full", memory: str = "a") -> list[str]:
        return [TRAIN_PY, f"{AREA7}/run_vlm_nemo.py", "--profile", profile, "--out", "/out", "--seed", str(SEED),
                "--base", f"{AREA7}/{BASE_REL}", "--memory", memory]

    def launch_record(self, **overrides) -> dict[str, Any]:
        """ext-launch's record of the full training run."""
        rec = {"schema": "gatedtrain.launch_record", "schema_version": "2.0", "tags": ["prototype", "unpromotable"],
               "adapter": ADAPTER, "area": 7, "profile": "full", "label": "mm-a7-vlm-full",
               "host_work": str(self.host_work),
               "runtime": {"kind": "container", "image": IMAGE, "venv_lock_sha256": None},
               "framework": {"name": "nemo_rl", "version": NEMO_RL_COMMIT, "commit": NEMO_RL_COMMIT},
               "training_seed": SEED, "gpus": True, "argv": self.argv(), "mounts": self.mounts(self.out),
               "tmpfs": ["/tmp"], "network": "none", "network_reason": None, "user": "1000:1000",
               "env_names": list(ENV_NAMES), "gpu_est_h": 1.0, "gpu_used_h": 6.0, "output_dir": str(self.out),
               "output_root": str(self.output_root),
               "gpu_step_sha256": file_sha256(self.host_work / "bin" / "gpu-step.sh"), "allow_network": False,
               "interpreter_path": TRAIN_PY, "interpreter_sha256": None,
               "script_path": f"{AREA7}/run_vlm_nemo.py", "script_sha256": file_sha256(self.runtime / "run_vlm_nemo.py"),
               "host_config_sha256": self.config_sha()}
        rec.update(overrides)
        rec["label"] = launch_label(rec)
        return rec

    def rewrite_launch(self, **overrides) -> Path:
        """A changed launch record, its label re-hashed and logged in gpu-guard.log."""
        rec = self.launch_record(**overrides)
        p = write_canonical(self.root / "launch.json", rec)
        self.label = rec["label"]
        self.guard_log()
        return p

    def request(self, out: Path, **overrides) -> dict[str, Any]:
        req = {"area": 7, "label": "mm-a7-vlm-smoke", "host_work": str(self.host_work),
               "runtime": {"kind": "container", "image": IMAGE},
               "framework": {"name": "nemo_rl", "version": NEMO_RL_COMMIT, "commit": NEMO_RL_COMMIT},
               "training_seed": SEED, "gpus": True, "network": "none", "network_reason": None, "user": "1000:1000",
               "env": dict(TRAIN_ENV), "mounts": self.mounts(out), "tmpfs": ["/tmp"], "output_dir": str(out),
               "gpu_est_h": 0.15, "argv": self.argv("smoke")}
        req.update(overrides)
        return req

    def eval_request(self, script: str, out: Path, *, gen: Path | None = None, cut: Path | None = None,
                     items: str = "items-valA", **overrides) -> dict[str, Any]:
        """A generation (gen_vqa.py, vLLM worker venv, GPU) or scoring (`-I score_vqa.py`, the image's main venv, CPU)
        request in the pinned image."""
        if script == "gen_vqa.py":
            model = "/inputs/cut" if cut is not None else f"{AREA7}/{BASE_REL}"
            mounts = [{"host": str(self.runtime), "container": AREA7, "mode": "ro"}]
            if cut is not None:
                mounts.append({"host": str(cut), "container": "/inputs/cut", "mode": "ro"})
            mounts.append({"host": str(out), "container": "/out", "mode": "rw"})
            req = {"gpus": True, "gpu_est_h": 0.05, "env": {"HF_HUB_OFFLINE": "1", "HOME": "/tmp/home",
                                                            "PYTHONUNBUFFERED": "1"},
                   "argv": [GEN_PY, f"{AREA7}/gen_vqa.py", "--model", model, "--items", f"{AREA7}/{items}", "--out",
                            "/out", "--eval-seeds", "11"], "mounts": mounts}
        else:
            argv = [TRAIN_PY, "-I", f"{AREA7}/score_vqa.py", "--gen", "/inputs/gen", "--items", f"{AREA7}/{items}",
                    "--out", "/out", "--eval-seeds", "11"]
            if items == "items-valB":
                argv += ["--evaluator-id", VALB_ID]
            req = {"gpus": False, "gpu_est_h": 0, "env": dict(SCORE_ENV), "argv": argv,
                   "mounts": [{"host": str(self.runtime), "container": AREA7, "mode": "ro"},
                              {"host": str(gen), "container": "/inputs/gen", "mode": "ro"},
                              {"host": str(out), "container": "/out", "mode": "rw"}]}
        req.update({"area": 7, "label": f"mm-a7-{script[:-3].replace('_', '-')}", "host_work": str(self.host_work),
                    "runtime": {"kind": "container", "image": IMAGE},
                    "framework": {"name": "nemo_rl", "version": NEMO_RL_COMMIT, "commit": NEMO_RL_COMMIT},
                    "training_seed": 0, "network": "none", "network_reason": None, "user": "1000:1000",
                    "tmpfs": ["/tmp"], "output_dir": str(out)})
        req.update(overrides)
        return req

    def new_out(self, name: str) -> Path:
        p = self.output_root / "area7-vlm" / name
        p.mkdir(parents=True)
        return p

    def launch(self, req: dict, name: str, *, profile: str = "smoke", smoke: Path | None = None,
               extra: list[str] = ()):
        path = self.root / f"{name}.request.yaml"
        path.write_text(yaml.safe_dump(req, sort_keys=False), encoding="utf-8")
        out = self.root / f"{name}.launch.json"
        args = ["ext-launch", "--adapter", ADAPTER, "--profile", profile, "--request", str(path), "--gpu-log",
                str(self.host_work / "gpu-log.md"), "--out-record", str(out)]
        if smoke is not None:
            args += ["--smoke-record", str(smoke)]
        return run_cli(args + list(extra), cwd=self.root, env=self.env()), out

    def recheck(self, record: Path, extra: list[str] = ()):
        return run_cli(["ext-launch", "--adapter", ADAPTER, "--recheck", str(record), *extra], cwd=self.root,
                       env=self.env())

    # --- collect ---

    def collect(self, runs_root: Path, *, baseline: Path | None = None, candidate: Path | None = None,
                launch: Path | None = None, timeout: int = 300):
        base_arm, cand_arm = self.arms[self.split]
        args = ["ext-collect", str(self.root / "spec.yaml"), "--adapter", ADAPTER, "--stage-out", str(self.out),
                "--launch-record", str(launch or self.root / "launch.json"), "--smoke-record",
                str(self.root / "smoke.json"), "--eval-baseline", str(baseline or base_arm / "arm-exact_match"),
                "--eval-candidate", str(candidate or cand_arm / "arm-exact_match"), "--runs-root", str(runs_root)]
        return run_cli(args, cwd=self.root, timeout=timeout, env=self.env())

    def edit_json(self, path: Path, fn: Callable[[Any], None]) -> None:
        doc = json.loads(path.read_text(encoding="utf-8"))
        fn(doc)
        write_json(path, doc)

    def edit_jsonl(self, path: Path, fn: Callable[[list[dict]], None]) -> None:
        rows = read_jsonl(path)
        fn(rows)
        write_jsonl(path, rows)

    def reseal(self, cut: Path) -> None:
        """Rewrite cut.json's file hashes after a cut file was edited (the cut stays complete)."""
        self.edit_json(cut / "cut.json", lambda d: d.update(files={n: file_sha256(cut / n) for n in d["files"]}))


def probe_copy(dst: Path) -> Path:
    """A writable copy of `fixtures/ext/vlm/probe-a6/` (the run's outputs plus `data/shards/`)."""
    shutil.copytree(PROBE, dst)
    return dst
