"""NeMo RL adapter for area "VLM" (VLM GRPO of Qwen3.5-0.8B on CLEVR-CoGenT TrainA with NeMo RL's VLM environment;
multimodal spec section 6 rows 7 and 7s and the area 7 decisions in runtime/README.md; gatedtrain's tests/README.md section
11.24, EXT-200 to EXT-213).

MM-12: a pure-Python class that imports no framework, torch or pickle loader. It reads JSON, JSONL, the text
`env-freeze.txt` and safetensors headers (standard library), and hashes everything else.

Run layout, read from the area runtime `runtime/` of this repository (`run_vlm_nemo.py`,
`ptc_vlm_hooks.py`, `vlm_env_logged.py`, `gen_vqa.py`, `score_vqa.py`, `make_data.py`; read 2026-11-14) and its probe
outputs (`out/area7-vlm/probe-a6/train/`: a 5-step smoke of the classic synchronous `grpo_train`;
`out/area7-vlm/gen-base-probe1/`: a base generation and scoring of 100 valA items):

- one stage, `rl.nemo_rl_vlm@0.1` (run_args.json `stage` "vlm_grpo"), from the plain base directory
  `models/qwen3.5-0.8b` (MODELS.json repo Qwen/Qwen3.5-0.8B, revision 2fc06364...); no SFT log, no sidecar, no executor;
- `write_cut` (ptc_vlm_hooks.py) builds `cuts/checkpoint-<N>/` through `cuts/.checkpoint-<N>.partial/` after NeMo RL
  publishes `nemo/checkpoints/step_<N>`: hard links to the consolidated weights (one or more `*.safetensors` files and
  their index when NeMo RL writes one; probe-a6: `model-00001-of-00001.safetensors` and `model.safetensors.index.json`)
  and `config.json`, copies of the tokenizer and processor files, `sampler.json` (gen_vqa.py SAMPLER), `base_ref.json`
  (MODELS.json repo, revision, files; `init_checkpoint` null), `training_info.json`, then `cut.json` last (schema
  `ptc_mm.vlm_cut`: `files`, `deployable` = every file but base_ref.json, training_info.json and cut.json, `global_step`,
  `examples_consumed` = 16 x step, `seed`, `profile`, `stage`, `script_sha256`, `nemo_rl_step_dir`);
- consumption (`vlm_env_logged.LoggedVLMEnvironment.step`): `env_calls.jsonl` `{call, position, row_id, reward,
  response_sha256, utc}`, one line per sample, call N = step N, 128 lines per call: the step's 16 rows (rows
  16(N-1) .. 16N-1 of train.jsonl in file order, data.shuffle=false, drop_last) each 8 times in a row; NeMo RL's
  `nemo/logs/exp_<n>/train_data_step<N>.jsonl` (grpo.py:4690) holds one line per sample `{"rewards": [r], ..., "idx":
  i}`, r the float32 value of the environment's reward (`torch.tensor(results)`, vlm_environment.py:207). The join is
  the rewards in order (probe-a6: 640 of 640);
- `run_args.json` (`write_run_args`): `stage`, `profile`, `seed`, `dry_run`, `max_steps`, `profile_settings`,
  `memory_rung`, `model_name_or_path`, `base_model`, `init_checkpoint`, `train_file`, `train_file_sha256`,
  `prompt_file_sha256`, `nemo_rl_overrides`, `attempt`, `script_sha256`, `hooks_sha256`, `env_sha256`, `framework`
  {name, commit, versions}; `env-freeze.txt` package lines and `# nemo-rl commit <40 hex>`.

Identity (MM-17): full weights, no composition with the base (MM-18 does not apply). The plain base directory's model
hash is the manifest hash of all its files; a cut's is the manifest hash of its `cut.json` `deployable` files.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from gatedtrain.canonical import entries_hash, file_sha256, manifest_entries, manifest_hash, sha256_bytes, text_lines
from gatedtrain.errors import Problem
from gatedtrain.families.nemo_rl import NEMO_RL_IMAGES
from gatedtrain.ext.registry import EVALUATOR_ROWS, TRAINER_ROWS
from gatedtrain_recipe_nemo_rl_vlm_clevr.rows import RUNTIME, SCORE_VQA_SHA256
from gatedtrain.ext.sdk import (compact_sha256 as _compact_sha256, float32, is_int as _int, is_num as _num, norm_path as _norm_path,
                   parse_options, plain_file as _plain_file, read_cut_marker, read_json as _read_json, read_jsonl,
                   read_run_args, safetensors_problem, sha_or_none as _sha_or_none, shards_concat_sha256)
from gatedtrain.families.nemo_rl import (CHECKPOINT_DIR_RE, CUTS_DIR, CUT_MARKER, EXP_DIR_RE, FREEZE_COMMIT_RE, FREEZE_FILE,
                                 NEMO_LOGS, NEMO_RL_COMMIT, SHA_RE, TRAIN_DATA_RE)

NAME = "nemo_rl_vlm"
ROW = TRAINER_ROWS[NAME]
TRAINER_ID = ROW["uses"][0]
EVALUATOR_ID, VALB_ID = ROW["evaluators"]
STAGE = "vlm_grpo"
LIBRARY = ("NeMo RL 4f39fd8 (derived image gatedtrain-a6-nemo-rl:1) with the area runtime's cut writer (ptc_vlm_hooks.write_cut "
           "after NeMo RL publishes step_<N>; cut.json written last)")
# RUNTIME (rows): the runtime directory, mounted read-only at its own host path, so it is also its container path.
BASE_REL = "models/qwen3.5-0.8b"
BASE_REPO, BASE_REVISION = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
INDEX = "model.safetensors.index.json"
WEIGHTS_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.safetensors")
# ptc_vlm_hooks.py CUT_REQUIRED; the weights and the optional files besides.
CUT_REQUIRED = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                "preprocessor_config.json", "sampler.json")
CUT_OPTIONAL = ("video_preprocessor_config.json", "vocab.json", "merges.txt", INDEX)
CUT_EXTRA = ("base_ref.json", "training_info.json")              # ptc_vlm_hooks.py NOT_DEPLOYABLE besides cut.json
CUT_SCHEMA = "ptc_mm.vlm_cut"
BASE_REQUIRED = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "preprocessor_config.json")
BASE_REF = "base_ref.json"
SAMPLER_FILE = "sampler.json"
PROFILES = {"smoke": {"steps": 5, "save_period": 5}, "full": {"steps": 60, "save_period": 20}}   # run_vlm_nemo.py
MEMORY_RUNGS = ("a", "b", "c")                                    # run_vlm_nemo.py MEMORY
PROMPTS, GENERATIONS = 16, 8                                     # run_vlm_nemo.py PROFILES (both)
REWARD_MIN, REWARD_MAX = 0.0, 1.0                                # 0.2 x format [0, 1] + 0.8 x exact {0, 1}
ENV_CALLS = "env_calls.jsonl"
# run_args.json SHA-256 of the files the launched script reads from its own directory (run_vlm_nemo.py main).
HELPERS = {"hooks_sha256": "ptc_vlm_hooks.py", "env_sha256": "vlm_env_logged.py",
           "prompt_file_sha256": "prompt_clevr_cot.txt"}
REQUIRED_OVERRIDES = ("data.shuffle=false",                      # file order: the observed logs assume it
                      "data_plane.enabled=false")                # the classic synchronous grpo_train
# The prompt file (gen_vqa.py `prompt_block_sha256`, run_args.json `prompt_file_sha256`): training rows and the
# evaluator's prompts are built from it.
PROMPT_BLOCK_SHA256 = "sha256:d88becf2d8bd00d8ec1678786317304b268d729ebad9e519ca8c5f281cb31d8b"
EVAL_ONLY_SAMPLER_KEYS = ("backend", "prompt_block_sha256")       # score_vqa.py adds them to gen_vqa.py SAMPLER
SCRIPT_TRAIN, SCRIPT_GEN, SCRIPT_SCORE = "run_vlm_nemo.py", "gen_vqa.py", "score_vqa.py"
FLAGGED = (SCRIPT_SCORE,)                                        # `python -I score_vqa.py` (standard library only)
SCRIPT_PATHS = {s: f"{RUNTIME}/{s}" for s in (SCRIPT_TRAIN, SCRIPT_GEN, SCRIPT_SCORE)}
# argparse options of the three scripts (run_vlm_nemo.py, gen_vqa.py, score_vqa.py main): option -> takes a value.
TRAIN_OPTIONS = {"--profile": True, "--out": True, "--seed": True, "--base": True, "--memory": True,
                 "--max-steps": True}
GEN_OPTIONS = {"--model": True, "--items": True, "--out": True, "--eval-seeds": True, "--limit": True}
SCORE_OPTIONS = {"--gen": True, "--items": True, "--out": True, "--eval-seeds": True, "--limit": True,
                 "--evaluator-id": True, "--model-hash": True, "--base-model-hash": True, "--item-set-hash": True,
                 "--self-test": False}
OPTIONS = {SCRIPT_TRAIN: TRAIN_OPTIONS, SCRIPT_GEN: GEN_OPTIONS, SCRIPT_SCORE: SCORE_OPTIONS}
ITEM_SETS = {"items-valA": EVALUATOR_ID, "items-valB": VALB_ID}   # item set -> the evaluator that scores it
EVAL_CUT_CONTAINER = "/inputs/cut"
GEN_CONTAINER = "/inputs/gen"
# gen_vqa.py DEPLOYABLE_NAMES: with every `*.safetensors` file, the model files gen_meta.json `model_files` records.
GEN_MODEL_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json",
                   "merges.txt", "preprocessor_config.json", "video_preprocessor_config.json", INDEX)
# score_vqa.py: the arm's sub-metrics (finite numbers, MM-1) and the answer parser (copied, compared with the runtime's
# by a test).
SUB_METRICS = ("answer_found", "answer_tagged", "is_yes_no", "n_new_tokens", "truncated")
NUMBER_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                "sixteen seventeen eighteen nineteen twenty").split()
TOKEN_RE = re.compile(r"\b(\d+|" + "|".join(NUMBER_WORDS) + r"|yes|no)\b", re.I)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)

CHECKPOINT_COMPLETENESS = {
    "library": LIBRARY,
    "rule": ("a cuts/checkpoint-<N> directory is a cut iff its cut.json, written last after an atomic rename from "
             "cuts/.checkpoint-<N>.partial, parses with schema ptc_mm.vlm_cut 1.0, global_step = N, nemo_rl_step_dir "
             "step_<N>, a positive integer examples_consumed, stage vlm_grpo, profile smoke or full, an integer seed, and "
             "files listing exactly the other files of the directory (the six required components, the consolidated "
             "safetensors weights with their index when there are several, the optional processor and tokenizer files, "
             "base_ref.json and training_info.json), each with its SHA-256; deployable lists exactly the files but "
             "base_ref.json and training_info.json; every weights file is whole and the index names exactly them; "
             "NeMo RL's nemo/checkpoints/step_<N> are resume state, never cuts; the cut's last step included is N"),
    "complete_dir_pattern": f"{CUTS_DIR}/{CHECKPOINT_DIR_RE.pattern}",
    "sources": ["runtime/ptc_vlm_hooks.py write_cut, install_cut_writer (cut.json "
                "last, os.rename)",
                "nemo-rl 4f39fd8 nemo_rl/utils/checkpoint_storage.py:80-97 (publish: tmp_step_N -> step_N)",
                "nemo-rl 4f39fd8 nemo_rl/utils/checkpoint.py:406 (training_info.json)",
                "probe-a6/train/cuts/checkpoint-5 of the area's GPU probes (a GPU smoke cut; tests/fixtures/vlm/probe-a6)"],
    "verified": True,                    # probe-a6 (5-step GPU smoke, 2026-11-14) wrote a cut of this shape
}
SEED_DERIVATION = ("run_vlm_nemo.py passes the launch seed as NeMo RL grpo.seed (set_seed in NeMo RL's setup); the data "
                   "order is the file order (data.shuffle=false), observed per sample in env_calls.jsonl; vLLM's "
                   "sampling seed is whatever NeMo RL derives and is not claimed. The training seed is run_args.json "
                   "`seed`, which must equal the launch record's training_seed and every cut's cut.json `seed`. "
                   "Evaluation is greedy (K = 1; eval seed 11 is recorded, nothing is drawn); 05 section 14 "
                   "H(run_seed, step, data_ref, sample_index) is not claimed.")


def canon_token(tok: str) -> str:
    """score_vqa.py canon_token: number words become digits, leading zeros are dropped, yes/no lower case."""
    t = tok.lower()
    if t in NUMBER_WORDS:
        return str(NUMBER_WORDS.index(t))
    if t.isdigit():
        return str(int(t))
    return t


def parse_answer(text: Any) -> dict[str, Any]:
    """score_vqa.py parse_answer: {answer (canonical or None), answer_tagged} of a response text."""
    if not isinstance(text, str):
        return {"answer": None, "answer_tagged": 0}
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    blocks = ANSWER_RE.findall(text)
    cand = blocks[-1] if blocks else text
    toks = TOKEN_RE.findall(cand)
    return {"answer": canon_token(toks[-1]) if toks else None, "answer_tagged": int(bool(blocks))}


def canon_reference(ref: Any) -> str | None:
    """score_vqa.py canon_reference: the canonical form of a reference answer, or None."""
    if not isinstance(ref, str):
        return None
    toks = TOKEN_RE.findall(ref)
    if len(toks) != 1 or TOKEN_RE.sub("", ref).strip():
        return None
    return canon_token(toks[0])


def payload_text(image_sha256: Any, question: Any) -> str | None:
    """make_data.py: the exact-overlap payload "<image sha256>\\n<question>" of a training row (`input`) or an item
    (`text`); None when either part is not a string."""
    if not (isinstance(image_sha256, str) and SHA_RE.fullmatch(image_sha256) and isinstance(question, str)):
        return None
    return f"{image_sha256}\n{question}"


def row_question(row: dict) -> Any:
    """The question text of a training row: the text part of its user message (make_data.py `messages`)."""
    msgs = row.get("messages")
    if not (isinstance(msgs, list) and msgs and isinstance(msgs[0], dict) and isinstance(msgs[0].get("content"), list)):
        return None
    texts = [p.get("text") for p in msgs[0]["content"] if isinstance(p, dict) and p.get("type") == "text"]
    return texts[0] if len(texts) == 1 else None


def _shards(data: Path) -> list[tuple[str, list[dict]]]:
    """[(shard name, rows)] of the spec's training shards in file-name order (the concatenation order of train.jsonl).
    Raises OSError; ValueError when a shard does not parse."""
    from gatedtrain.ledger import shard_table

    out = []
    for name, _ in shard_table(data):
        rows, why = read_jsonl(Path(data) / name)
        if rows is None:
            raise ValueError(f"training shard {name}: {why}")
        out.append((name, rows))
    return out


def _weights_problem(d: Path, names: list[str]) -> str | None:
    """None when the `*.safetensors` files among `names` are whole, at least one, and named exactly by the index
    (`weight_map` values) when the index is among `names`; a single file needs no index (ptc_vlm_hooks.write_cut)."""
    weights = sorted(n for n in names if n.endswith(".safetensors"))
    if not weights:
        return "no consolidated *.safetensors weights file"
    odd = [n for n in weights if not WEIGHTS_RE.fullmatch(n)]
    if odd:
        return f"weights file names {odd} are not plain *.safetensors names"
    if INDEX in names:
        idx, why = _read_json(Path(d) / INDEX)
        wm = idx.get("weight_map") if isinstance(idx, dict) else None
        if not (isinstance(wm, dict) and wm and all(isinstance(v, str) for v in wm.values())):
            return f"{INDEX} has no weight_map of file names{f' ({why})' if why else ''}"
        if sorted(set(wm.values())) != weights:
            return f"{INDEX} weight_map names {sorted(set(wm.values()))}, not the weights files {weights}"
    elif len(weights) > 1:
        return f"{len(weights)} weights files and no {INDEX}"
    for n in weights:
        bad = safetensors_problem(Path(d) / n)
        if bad:
            return bad
    return None


class NemoRlVlmAdapter:
    """The MM-12 interface for NeMo RL VLM GRPO runs of the area runtime (EXT-200 to EXT-213)."""

    name = NAME
    trainer_id = TRAINER_ID
    trainer_ids = (TRAINER_ID,)
    evaluator_ids = tuple(ROW["evaluators"])
    framework_name = ROW["framework"]
    checkpoint_completeness = CHECKPOINT_COMPLETENESS
    seed_derivation = SEED_DERIVATION
    binds_spec = bool(ROW["spec_binding"])
    arm_needs_items_dir = True                          # arm_problems() receives the item set and evaluated model
    consumption_fix = ("collect the run's own env_calls.jsonl and NeMo RL's train_data_step<N>.jsonl as the run wrote "
                       "them (MM-23, MM-27)")
    payload_source = ("a training row's `input` (the image hash and question NeMo RL trains on; exact overlap, MM-11, "
                      "EXT-206)")

    def __init__(self) -> None:
        self._base_cache: dict[str, list[dict]] = {}

    def metric(self, evaluator_id: str) -> dict[str, Any]:
        if evaluator_id not in self.evaluator_ids:
            raise KeyError(evaluator_id)
        row = EVALUATOR_ROWS[evaluator_id]
        return {"metric": row["metric"], "direction": row["direction"], "score_range": tuple(row["score_range"]),
                "error_classes": tuple(row["error_classes"]), "score_outcomes": tuple(row["score_outcomes"]),
                "scorer": dict(row["scorer"]), "binary": row["binary"], "tier": row["tier"]}

    # --- the base directory and cuts (MM-17, MM-20; EXT-201, EXT-202) ---

    @staticmethod
    def is_cut(d: Path) -> bool:
        """Whether `d` is (meant to be) a cut: it holds cut.json. The plain base directory never does."""
        return (Path(d) / CUT_MARKER).exists() or (Path(d) / CUT_MARKER).is_symlink()

    @staticmethod
    def base_dir_problem(d: Path) -> str | None:
        """None when `d` is a Qwen3.5 VLM directory: config, tokenizer and processor files and whole weights."""
        for rel in BASE_REQUIRED:
            if not _plain_file(Path(d) / rel):
                return f"{rel} is missing (not a VLM model directory)"
        try:
            names = sorted(p.name for p in Path(d).iterdir() if _plain_file(p))
        except OSError as exc:
            return f"{Path(d).name} cannot be listed ({exc.__class__.__name__})"
        return _weights_problem(Path(d), names)

    def _cut_files_problem(self, d: Path) -> str | None:
        """Review 9 M1 (EXT-154): a file that cannot be read makes the directory incomplete, never an exception."""
        try:
            return self._cut_files_problem_unguarded(d)
        except OSError as exc:
            return (f"{Path(exc.filename).name if exc.filename else Path(d).name} cannot be read "
                    f"({exc.__class__.__name__}: {exc.strerror or exc})")

    def _cut_files_problem_unguarded(self, d: Path) -> str | None:
        doc, why = read_cut_marker(d)
        if doc is None:
            return why
        if (doc.get("schema"), doc.get("schema_version")) != (CUT_SCHEMA, "1.0"):
            return f"{CUT_MARKER} schema is {doc.get('schema')!r} {doc.get('schema_version')!r}, not {CUT_SCHEMA} 1.0"
        files = doc["files"]
        named = set(CUT_REQUIRED) | set(CUT_OPTIONAL) | set(CUT_EXTRA)
        unknown = sorted(n for n in files if n not in named and not n.endswith(".safetensors"))
        if unknown:
            return f"{CUT_MARKER} lists {unknown}, which write_cut never writes (a cut holds {sorted(named)} and weights)"
        missing = sorted((set(CUT_REQUIRED) | set(CUT_EXTRA)) - set(files))
        if missing:
            return f"{CUT_MARKER} does not list {missing}"
        present = sorted(p.name for p in Path(d).iterdir())
        if present != sorted([*files, CUT_MARKER]):
            extra = sorted(set(present) - set(files) - {CUT_MARKER})
            return f"the directory holds {extra or present}, which {CUT_MARKER} does not list"
        for name in sorted(files):
            if not _plain_file(Path(d) / name):
                return f"{name} is not a plain file"
            if file_sha256(Path(d) / name) != files[name]:
                return f"{name} is not the SHA-256 {CUT_MARKER} records"
        deploy = sorted(n for n in files if n not in CUT_EXTRA)
        dep = doc.get("deployable")
        if not (isinstance(dep, list) and all(isinstance(x, str) for x in dep) and sorted(dep) == deploy):
            return f"{CUT_MARKER} deployable is {dep!r}, not every file but {list(CUT_EXTRA)}: {deploy}"
        gs = doc.get("global_step")
        if not (_int(gs) and gs >= 1):
            return f"{CUT_MARKER} global_step is {gs!r}, not a positive integer"
        if doc.get("nemo_rl_step_dir") != f"step_{gs}":
            return (f"{CUT_MARKER} nemo_rl_step_dir is {doc.get('nemo_rl_step_dir')!r}, not step_{gs} (the NeMo RL "
                    "checkpoint the cut was built from)")
        if not (_int(doc.get("examples_consumed")) and doc["examples_consumed"] >= 1):
            return f"{CUT_MARKER} examples_consumed is {doc.get('examples_consumed')!r}, not a positive integer"
        if doc.get("stage") != STAGE:
            return f"{CUT_MARKER} stage is {doc.get('stage')!r}, not {STAGE}"
        if doc.get("profile") not in PROFILES:
            return f"{CUT_MARKER} profile is {doc.get('profile')!r}, not smoke or full"
        if not _int(doc.get("seed")):
            return f"{CUT_MARKER} seed is {doc.get('seed')!r}, not an integer"
        return _weights_problem(Path(d), list(files))

    def model_dir_problem(self, d: Path) -> str | None:
        """None when `d` is a complete cut (it holds cut.json) or a plain base model directory."""
        d = Path(d)
        if d.is_symlink() or not d.is_dir():
            return f"{d} is not a plain directory"
        if self.is_cut(d):
            return self._cut_files_problem(d)
        return self.base_dir_problem(d)

    def cut_problem(self, d: Path, step: int) -> str | None:
        d = Path(d)
        if d.is_symlink() or not d.is_dir():
            return f"{d} is not a plain directory"
        bad = self._cut_files_problem(d)
        if bad:
            return bad
        doc, _ = read_cut_marker(d)
        if doc.get("global_step") != step:
            return f"{CUT_MARKER} global_step is {doc.get('global_step')!r}, not {step} (the directory's step)"
        return None

    def checkpoint_dirs(self, stage_out: Path) -> list[tuple[int, Path]]:
        """Every `cuts/checkpoint-<N>` directory, complete or not, in step order; `.checkpoint-<N>.partial` and NeMo RL's
        `nemo/checkpoints/step_<N>` are never one."""
        cuts = Path(stage_out) / CUTS_DIR
        if cuts.is_symlink() or not cuts.is_dir():
            return []
        out = []
        for p in cuts.iterdir():
            m = CHECKPOINT_DIR_RE.match(p.name)
            if m and p.is_dir() and not p.is_symlink():
                out.append((int(m.group(1)), p))
        return sorted(out)

    def find_cuts(self, stage_out: Path) -> tuple[list[tuple[int, Path]], list[Path]]:
        """(complete [(global_step, path)] in step order, incomplete [path]) (EXT-201)."""
        complete, incomplete = [], []
        for step, d in self.checkpoint_dirs(stage_out):
            if self.cut_problem(d, step) is None:
                complete.append((step, d))
            else:
                incomplete.append(d)
        return complete, incomplete

    def _base_entries(self, base: Path) -> list[dict]:
        key = str(Path(base).resolve())
        if key not in self._base_cache:
            self._base_cache[key] = manifest_entries(Path(base))
        return self._base_cache[key]

    def identity(self, cut: Path, base: Any = None, *, entries: list[dict] | None = None) -> dict[str, Any]:
        """EXT-202 (MM-17). The base alone: `model_hash` = `cut_hash` = its manifest hash (every file). A cut: `model_hash`
        over its cut.json `deployable` files (the cut check makes them every file but base_ref.json, training_info.json
        and cut.json), `cut_hash` over the whole cut; full weights, so `base` is not used, `base_ref` null."""
        is_cut = self.is_cut(cut)
        if entries is None and not is_cut:
            entries = self._base_entries(Path(cut))
        if entries is None:
            entries = manifest_entries(Path(cut))
        if not is_cut:
            self._base_cache.setdefault(str(Path(cut).resolve()), entries)
            h = entries_hash(entries)
            return {"model_hash": h, "cut_hash": h, "components": [e["path"] for e in entries], "base_ref": None}
        doc, _ = read_cut_marker(cut)
        deploy = set(doc.get("deployable") or []) if doc and isinstance(doc.get("deployable"), list) else set()
        comps = [e for e in entries if e["path"] in deploy]
        return {"model_hash": entries_hash(comps), "cut_hash": entries_hash(entries),
                "components": [e["path"] for e in comps], "base_ref": None}

    def needs_base(self, d: Path) -> bool:
        """ext-arm-fill: full weights, never composed with a base (MM-18 does not apply)."""
        return False

    # --- consumption (MM-23, MM-27; EXT-203, EXT-204) ---

    def _plan(self, stage_out: Path, data: Path, cuts: list) -> tuple[list[tuple[dict, int, int, str]], list[str]]:
        """([(env_calls.jsonl line, step, member, shard)] of the steps up to the last cut) and the problems that refuse
        the logs: the step's env-log lines joined with NeMo RL's step log by rewards in order; the 16 rows of the step
        in file order, each 8 times in a row."""
        args, why = read_run_args(stage_out)
        if args is None:
            return [], [why]
        if not cuts:
            return [], ["no complete cut"]
        if args.get("stage") != STAGE:
            return [], [f"run_args.json stage is {args.get('stage')!r}, not {STAGE}"]
        prof = args.get("profile_settings") if isinstance(args.get("profile_settings"), dict) else {}
        for key, want in (("prompts", PROMPTS), ("generations", GENERATIONS)):
            if not (_int(prof.get(key)) and prof[key] == want):
                return [], [f"run_args.json profile_settings.{key} is {prof.get(key)!r}, not {want} (run_vlm_nemo.py: 16 "
                            "prompts x 8 generations per step)"]
        max_steps = args.get("max_steps")
        if not (_int(max_steps) and max_steps >= 1):
            return [], [f"run_args.json max_steps {max_steps!r} is not a positive integer"]
        try:
            shards = _shards(data)
        except (OSError, ValueError) as exc:
            return [], [f"the spec's training shards cannot be read ({exc})"]
        rows = [(name, r) for name, rs in shards for r in rs]
        ids = [r.get("row_id") for _, r in rows]
        if not rows or not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids):
            return [], ["the training rows have no unique non-empty string row_id each"]
        marks = []
        for _, d in cuts:
            doc, why = read_cut_marker(d)
            if doc is None:
                return [], [f"{Path(d).name}: {why}"]
            marks.append((doc.get("global_step"), doc.get("examples_consumed"), Path(d).name))
        per_step = PROMPTS * GENERATIONS
        spe = len(rows) // PROMPTS                        # drop_last
        if spe < 1:
            return [], [f"{len(rows)} training rows do not fill one step of {PROMPTS} prompts"]
        known = set(ids)
        env, why = read_jsonl(Path(stage_out) / ENV_CALLS)
        if env is None:
            return [], [why]
        out: list[str] = []

        def bad(msg: str) -> bool:
            out.append(msg)
            return len(out) >= 8

        if len(env) != per_step * max_steps:
            return [], [f"{ENV_CALLS} has {len(env)} lines, not {per_step} x run_args.json max_steps {max_steps} (one "
                        "line per sample; synchronous GRPO scores 128 samples per step)"]
        call, pos = 0, 0
        for n, x in enumerate(env, start=1):
            c, p, rid, rw = (x.get(k) for k in ("call", "position", "row_id", "reward"))
            where = f"{ENV_CALLS} line {n}"
            if not (_int(c) and _int(p)):
                if bad(f"{where}: call {c!r} or position {p!r} is not an integer"):
                    return [], out
                continue
            if c == call + 1 and p == 0:
                call, pos = c, 0
            elif c == call and p == pos + 1:
                pos = p
            else:
                if bad(f"{where}: call {c} position {p} does not follow call {call} position {pos} (calls are "
                       "numbered from 1, positions from 0 within a call)"):
                    return [], out
                call, pos = c, p
            if not (isinstance(rid, str) and rid in known):
                bad(f"{where}: row_id {rid!r} is not a row of the training shards")
            if not _num(rw):
                bad(f"{where}: reward {rw!r} is not a finite number")
            elif not REWARD_MIN <= rw <= REWARD_MAX:
                bad(f"{where}: reward {rw!r} is outside [0, 1] (0.2 x format + 0.8 x exact_alnum)")
            resp = x.get("response_sha256")
            if not (isinstance(resp, str) and SHA_RE.fullmatch(resp)):
                bad(f"{where}: response_sha256 {resp!r} is not a SHA-256")
            if len(out) >= 8:
                return [], out
        if out:
            return [], out
        logs = Path(stage_out).joinpath(*NEMO_LOGS)
        try:
            exps = sorted(p for p in logs.iterdir() if p.is_dir() and EXP_DIR_RE.fullmatch(p.name)) if logs.is_dir() else []
        except OSError as exc:
            return [], [f"{'/'.join(NEMO_LOGS)} cannot be listed ({exc.__class__.__name__})"]
        if len(exps) != 1:
            return [], [f"{'/'.join(NEMO_LOGS)} holds {[p.name for p in exps]}, not one exp_<n> directory (a rerun "
                        "starts from an empty output directory)"]
        exp = exps[0]
        try:
            names = {int(m.group(1)): p for p in exp.iterdir() if (m := TRAIN_DATA_RE.fullmatch(p.name))}
        except OSError as exc:
            return [], [f"{exp.name} cannot be listed ({exc.__class__.__name__})"]
        beyond = sorted(k for k in names if not 1 <= k <= max_steps)
        if beyond:
            return [], [f"{exp.name} holds train_data_step{beyond[0]}.jsonl, beyond run_args.json max_steps {max_steps}"]
        plan: list[tuple[dict, int, int, str]] = []
        last = max(m[0] for m in marks if _int(m[0])) if any(_int(m[0]) for m in marks) else 0
        shard_of = {r["row_id"]: name for name, r in rows}
        for step in range(1, max_steps + 1):
            fname = f"train_data_step{step}.jsonl"
            if step not in names:
                if bad(f"{exp.name}/{fname} is missing (NeMo RL logs every step, grpo.py:4690)"):
                    return [], out
                continue
            logged, why = read_jsonl(names[step])
            if logged is None:
                if bad(f"{exp.name}/{why}"):
                    return [], out
                continue
            lines = env[per_step * (step - 1):per_step * step]
            if len(logged) != per_step:
                if bad(f"{exp.name}/{fname} logs {len(logged)} samples, not {per_step}"):
                    return [], out
                continue
            for i, (got, x) in enumerate(zip(logged, lines)):
                r = got.get("rewards")
                if not (isinstance(r, list) and len(r) == 1 and _num(r[0])):
                    bad(f"{exp.name}/{fname} line {i + 1}: rewards {r!r} is not one finite number")
                elif got.get("idx") != i or not _int(got.get("idx")):
                    bad(f"{exp.name}/{fname} line {i + 1}: idx {got.get('idx')!r} is not {i}")
                elif float32(x["reward"]) != r[0]:
                    bad(f"step {step} sample {i}: {ENV_CALLS} reward {x['reward']!r} (float32 {float32(x['reward'])!r}) "
                        f"is not {exp.name}/{fname} rewards {r[0]!r}: the environment's log and NeMo RL's step log do "
                        "not join in order")
                if len(out) >= 8:
                    return [], out
            start = ((step - 1) % spe) * PROMPTS
            want = [rows[start + j][1]["row_id"] for j in range(PROMPTS)]
            got_ids = [x["row_id"] for x in lines]
            if got_ids != [rid for rid in want for _ in range(GENERATIONS)]:
                if bad(f"step {step} scores rows {list(dict.fromkeys(got_ids))[:3]}..., not rows {start} .. "
                       f"{start + PROMPTS - 1} of the training file in file order, each {GENERATIONS} times in a row "
                       "(data.shuffle=false, drop_last, num_generations_per_prompt)"):
                    return [], out
                continue
            if step <= last:
                for k, x in enumerate(lines):
                    plan.append((x, step, k % GENERATIONS, shard_of[x["row_id"]]))
        for gs, consumed, name in marks:                  # MM-27
            if not (_int(gs) and gs <= max_steps):
                bad(f"{name}: global_step {gs!r} is beyond run_args.json max_steps {max_steps}")
            elif consumed != PROMPTS * gs or not _int(consumed):
                bad(f"{name} cut.json examples_consumed is {consumed!r}, not {PROMPTS} prompts x {gs}")
        return ([], out) if out else (plan, [])

    def consumption_problems(self, stage_out: Path, data: Path, cuts: list) -> list[str]:
        """EXT-203, EXT-204: why the observed logs cannot give the per-row consumption up to the last cut."""
        return self._plan(stage_out, data, cuts)[1]

    def consumption(self, stage_out: Path, data: Path, cuts: list, *, run_id: str = "",
                    attempt_id: str = "0") -> dict[str, Any]:
        """Precision `row`, derivation `observed`: one record per env_calls.jsonl line, `rewards` = [the environment's
        reward], `sample_id` = H(run, attempt, step, row_id, member) (member: the sample's place among its row's 8 in
        the step), `data_ref.row` = sha256(row_id), shard = the shard holding the row (NUL-joined text keys)."""
        from gatedtrain.ledger import WRITER_ID, check_precision

        plan, bad = self._plan(stage_out, data, cuts)
        if bad:                                           # collect refuses these before writing
            raise ValueError("; ".join(bad))
        check_precision("row", "observed")
        snapshot = manifest_hash(data)
        rows = []
        for x, step, member, shard in plan:
            rid = x["row_id"]
            key = (run_id, attempt_id, step, rid, member)
            rows.append({
                "run_id": run_id, "attempt_id": attempt_id, "step": step,
                "sample_id": "sha256:" + hashlib.sha256("\x00".join(str(v) for v in key).encode("utf-8")).hexdigest(),
                "data_ref": {"snapshot": snapshot, "shard": shard, "row": sha256_bytes(rid.encode("utf-8")),
                             "mixture": None, "component": None},
                "parent_ref": None, "outcome": "consumed", "reason": None, "rule_version": None, "gen_version": None,
                "reward_epoch": None, "rewards": [float(x["reward"])], "sampler_config_hash": None, "seed_inputs": None,
                "chunk": None, "precision": "row", "derivation": "observed", "writer_id": WRITER_ID,
            })
        last = max(step for step, _ in cuts)
        return {"precision": "row", "rows": rows, "derivation": "observed",
                "reason": (f"{ENV_CALLS} logs every sample at reward time and NeMo RL's train_data_step<N>.jsonl the "
                           f"step's rewards in the same order (the join holds for every step; MM-23, MM-27): "
                           f"{len(rows)} records of steps 1 .. {last}, data_ref.row = sha256(row_id) in the shard "
                           "holding the row, derivation observed; no sample is filtered")}

    def extra_read_outputs(self, stage_out: Path) -> list[Path]:
        """EXT-64: NeMo RL's step logs, which collect reads besides READ_OUTPUTS; each must postdate the launch."""
        logs = Path(stage_out).joinpath(*NEMO_LOGS)
        out = []
        try:
            for exp in sorted(logs.iterdir()) if logs.is_dir() else []:
                if exp.is_dir() and EXP_DIR_RE.fullmatch(exp.name):
                    out += sorted(p for p in exp.iterdir() if TRAIN_DATA_RE.fullmatch(p.name))
        except OSError:
            pass                                          # the consumption check reports an unlistable directory
        return out

    # --- item sets and exact overlap (MM-11; EXT-206) ---

    @staticmethod
    def _training_rows(data: Path) -> list[tuple[str, int, dict]]:
        return [(name, n, r) for name, rs in _shards(data) for n, r in enumerate(rs, start=1)]

    def training_payloads(self, stage_out: Path, base: Path, cuts: list, *,
                          data: Path | None = None) -> tuple[set[str] | None, str | None]:
        """EXT-206 (MM-11): the exact-overlap keys of the spec's training rows, their `input` "<image sha256>\\n<question>"
        (the pair NeMo RL trains on); an item overlaps when its `text` is one of them. (None, why) when the shards
        cannot be read."""
        if data is None:
            return None, "the spec's training data is not readable"
        try:
            return {r["input"] for _, _, r in self._training_rows(data) if isinstance(r.get("input"), str)}, None
        except (OSError, ValueError) as exc:
            return None, str(exc)

    def item_overlap_problems(self, item_rows: list[dict], data: Path) -> list[str]:
        """EXT-206: the exact-overlap key is well formed on both sides: every training row's `input` is "<its
        image_sha256>\\n<its user message's question>" and every item's `text` is "<its image_sha256>\\n<its question>",
        so a question asked of another image never matches and the same (image, question) pair always does. Problems
        (each refuses); OSError propagates."""
        out: list[str] = []
        try:
            rows = self._training_rows(data)
        except ValueError as exc:
            return [str(exc)]
        for name, n, r in rows:
            want = payload_text(r.get("image_sha256"), row_question(r))
            if want is None or r.get("input") != want:
                out.append(f"training row {name} line {n} ({r.get('row_id')!r}): `input` is not \"<image_sha256>\\n"
                           "<question>\" (the exact-overlap key; the question is the user message's text part)")
        for r in item_rows:
            want = payload_text(r.get("image_sha256"), r.get("question"))
            if want is None or r.get("text") != want:
                out.append(f"item {r.get('item')!r}: `text` is not \"<image_sha256>\\n<question>\" (the exact-overlap key)")
        return out[:8] + ([f"{len(out) - 8} more payload problems"] if len(out) > 8 else [])

    # --- arms (EXT-207) ---

    def arm_problems(self, arm: Path, records: list[dict], item_rows: list[dict], items_dir: Path | None = None,
                     model_dir: Path | None = None) -> list[str]:
        """EXT-207 (score_vqa.py sha256 2bb71c3c...). Per record of a registered item: its output
        (`outputs/<item>__r<k>.json`) names the record's item, repeat and seed and the item's question and reference;
        a scored record has exactly the sub_metrics answer_found, answer_tagged, is_yes_no, truncated (0 or 1) and
        n_new_tokens (a count), the output's score and sub_metrics are the record's, and the adapter's copy of the
        answer parser re-derives them from the output's `response` (parsed answer, answer_tagged, answer_found,
        is_yes_no, the score 1 exactly for the reference's canonical answer); a `missing_output` record has no response,
        a `scorer_error` record a reference without a canonical form. Beside the arm (the scoring output root):
        `env-runtime.json` names the registered items.jsonl, the pinned score_vqa.py, the arm's scorer code and the
        SHA-256 of `gen_meta.json`, which is there too (`_gen_meta_problems`)."""
        arm = Path(arm)
        out: list[str] = []
        meta, _ = _read_json(arm / "eval_meta.json")
        meta = meta if isinstance(meta, dict) else {}
        by_item = {r.get("item"): r for r in item_rows}
        for rec in records:
            item = rec.get("item")
            it = by_item.get(item)
            if it is None:
                continue                                  # an unknown item: the gate reports it
            ref = rec.get("output_ref")
            if not (isinstance(ref, dict) and isinstance(ref.get("path"), str)):
                out.append(f"item {item!r} has no output_ref (score_vqa.py writes one per (item, seed))")
                continue
            out += self._record_problems(arm, rec, ref["path"], it)
            if len(out) >= 8:
                return out[:8]
        root = arm.parent
        rt, why = _read_json(root / "env-runtime.json")
        if not isinstance(rt, dict):
            out.append(f"the scoring run's env-runtime.json beside the arm: {why or 'not an object'} (score_vqa.py writes "
                       "it next to arm-exact_match/)")
            return out
        want_items = _sha_or_none(Path(items_dir) / "items.jsonl") if items_dir is not None else None
        if items_dir is not None and rt.get("items_sha256") != want_items:
            out.append(f"env-runtime.json items_sha256 {rt.get('items_sha256')!r} is not the registered item set's "
                       f"items.jsonl {want_items}: the scoring run read another item set")
        code = rt.get("code_sha256") if isinstance(rt.get("code_sha256"), dict) else {}
        scorer = meta.get("scorer") if isinstance(meta.get("scorer"), dict) else {}
        if code != {"score_vqa.py": SCORE_VQA_SHA256}:
            out.append(f"env-runtime.json code_sha256 {code!r} is not the pinned score_vqa.py {SCORE_VQA_SHA256}")
        elif rt.get("scorer_code_sha256") != _compact_sha256(code):
            out.append("env-runtime.json scorer_code_sha256 is not the hash of its code_sha256")
        if rt.get("scorer_code_sha256") != scorer.get("code_sha256"):
            out.append(f"env-runtime.json scorer_code_sha256 {rt.get('scorer_code_sha256')!r} is not the arm's scorer "
                       f"code_sha256 {scorer.get('code_sha256')!r}")
        out += self._gen_meta_problems(root, rt, meta, want_items if items_dir is not None else None, model_dir)
        return out

    @staticmethod
    def _record_problems(arm: Path, rec: dict, path: str, it: dict) -> list[str]:
        where = f"output {path}"
        doc, why = _read_json(arm / path)
        if not isinstance(doc, dict):
            return [f"{where}: {why or 'not a JSON object'}"]
        if (doc.get("item"), doc.get("repeat"), doc.get("seed")) != (rec.get("item"), rec.get("repeat"), rec.get("seed")):
            return [f"{where} names item {doc.get('item')!r} repeat {doc.get('repeat')!r} seed {doc.get('seed')!r}, not "
                    f"the record's {rec.get('item')!r} {rec.get('repeat')!r} {rec.get('seed')!r}"]
        if doc.get("question") != it.get("question") or doc.get("reference") != it.get("answer"):
            return [f"{where}: question or reference is not the registered item's"]
        ref = canon_reference(it.get("answer"))
        err = rec.get("error")
        if err is not None:
            cls = err.get("class") if isinstance(err, dict) else None
            if cls == "missing_output" and (doc.get("response") is not None or doc.get("parsed_answer") is not None):
                return [f"{where}: a missing_output record whose output has a response"]
            if cls == "scorer_error" and ref is not None:
                return [f"{where}: a scorer_error record, but the reference {it.get('answer')!r} has the canonical form "
                        f"{ref!r}"]
            return []
        sub = rec.get("sub_metrics")
        if not (isinstance(sub, dict) and sorted(sub) == sorted(SUB_METRICS)):
            return [f"{where}: sub_metrics {sorted(sub) if isinstance(sub, dict) else sub!r} are not exactly "
                    f"{list(SUB_METRICS)} (score_vqa.py)"]
        flags = [k for k in SUB_METRICS if k != "n_new_tokens" and not (_int(sub[k]) and sub[k] in (0, 1))]
        if flags or not (_int(sub["n_new_tokens"]) and sub["n_new_tokens"] >= 0):
            return [f"{where}: sub_metrics {flags or ['n_new_tokens']} are not 0 or 1 (n_new_tokens a count)"]
        if ref is None:
            return [f"{where}: scored, but the reference {it.get('answer')!r} has no canonical form (a scorer_error)"]
        if doc.get("score") != rec.get("score") or doc.get("sub_metrics") != sub:
            return [f"{where}: score or sub_metrics differ from the arm's line"]
        parsed = parse_answer(doc.get("response"))
        if not isinstance(doc.get("response"), str):
            return [f"{where}: a scored record without a response"]
        if doc.get("parsed_answer") != parsed["answer"]:
            return [f"{where}: parsed_answer {doc.get('parsed_answer')!r} is not the answer parser's {parsed['answer']!r} "
                    "of its response"]
        if (sub["answer_tagged"], sub["answer_found"]) != (parsed["answer_tagged"], int(parsed["answer"] is not None)):
            return [f"{where}: answer_tagged {sub['answer_tagged']} and answer_found {sub['answer_found']} are not the "
                    "answer parser's of its response"]
        if sub["is_yes_no"] != int(ref in ("yes", "no")):
            return [f"{where}: is_yes_no {sub['is_yes_no']} is not whether the reference {ref!r} is yes/no"]
        if rec.get("score") != float(parsed["answer"] == ref):
            return [f"{where}: score {rec.get('score')!r} with parsed answer {parsed['answer']!r} and reference {ref!r}: "
                    "exact_match is 1 exactly when they are equal (no answer scores 0, MM-4)"]
        return []

    def _gen_meta_problems(self, root: Path, rt: dict, meta: dict, items_sha: str | None,
                           model_dir: Path | None) -> list[str]:
        """EXT-208: gen_meta.json, copied by score_vqa.py into the scoring output root, against the scoring record, the
        arm and the evaluated model."""
        out: list[str] = []
        gm, why = _read_json(root / "gen_meta.json")
        if not isinstance(gm, dict):
            return [f"gen_meta.json beside the arm: {why or 'not an object'} (score_vqa.py copies the generation run's)"]
        if rt.get("gen_meta_sha256") != _sha_or_none(root / "gen_meta.json"):
            out.append(f"gen_meta.json is not the file the scoring run read (env-runtime.json gen_meta_sha256 "
                       f"{rt.get('gen_meta_sha256')!r})")
        if items_sha is not None and gm.get("items_sha256") != items_sha:
            out.append(f"gen_meta.json items_sha256 {gm.get('items_sha256')!r} is not the registered item set's "
                       f"{items_sha}: the generation read another item set")
        if gm.get("prompt_block_sha256") != PROMPT_BLOCK_SHA256:
            out.append(f"gen_meta.json prompt_block_sha256 {gm.get('prompt_block_sha256')!r} is not the prompt file "
                       f"{PROMPT_BLOCK_SHA256} the training used")
        if gm.get("backend") != "vllm":
            out.append(f"gen_meta.json backend {gm.get('backend')!r} is not vllm (gen_vqa.py)")
        sampler = gm.get("sampler")
        want = ({**sampler, "backend": gm.get("backend"), "prompt_block_sha256": gm.get("prompt_block_sha256")}
                if isinstance(sampler, dict) else None)
        from gatedtrain.ext import arm as _arm
        if not _arm.same_json(meta.get("sampler"), want):
            out.append("eval_meta.json sampler is not gen_meta.json's sampler with its backend and prompt file hash")
        if not _arm.same_json(gm.get("eval_seeds"), meta.get("eval_seeds")):
            out.append(f"gen_meta.json eval_seeds {gm.get('eval_seeds')!r} are not the arm's {meta.get('eval_seeds')!r}")
        versions = gm.get("versions") if isinstance(gm.get("versions"), dict) else {}
        if meta.get("framework") != {"name": "vllm", "version": versions.get("vllm")}:
            out.append(f"eval_meta.json framework {meta.get('framework')!r} is not gen_meta.json's vLLM "
                       f"{versions.get('vllm')!r}")
        if model_dir is not None:
            try:
                want_files = {p.name: file_sha256(p) for p in sorted(Path(model_dir).iterdir())
                              if _plain_file(p) and (p.name in GEN_MODEL_FILES or p.name.endswith(".safetensors"))}
                cut_sha = _sha_or_none(Path(model_dir) / CUT_MARKER) if self.is_cut(model_dir) else None
            except OSError as exc:
                return out + [f"the evaluated model {model_dir} cannot be hashed ({exc.__class__.__name__})"]
            if gm.get("model_files") != want_files:
                out.append(f"gen_meta.json model_files are not the deployable files of the evaluated model {model_dir} "
                           "(the arm generated with another model)")
            cut = gm.get("cut")
            if (cut.get("cut_json_sha256") if isinstance(cut, dict) else None) != cut_sha:
                out.append(f"gen_meta.json cut {cut!r} does not name the evaluated model's cut.json ({cut_sha!r})")
        return out

    # --- pins and checks (MM-21, MM-29, MM-33; EXT-205) ---

    def framework_identity(self, stage_out: Path, launch: dict) -> dict[str, Any]:
        """env-freeze.txt names NeMo RL's commit (`# nemo-rl commit <40 hex>`, ptc_vlm_hooks.write_run_args), which is the
        pinned commit and the launch record's framework {name nemo_rl, version = commit = the pin} (MM-33). Raises
        ValueError with the reason (OSError for an unreadable freeze)."""
        freeze = Path(stage_out) / FREEZE_FILE
        if not _plain_file(freeze):
            raise ValueError(f"{freeze} is missing")
        commits = [m.group(1) for line in text_lines(freeze.read_text(encoding="utf-8"))
                   if (m := FREEZE_COMMIT_RE.fullmatch(line.strip()))]
        if commits != [NEMO_RL_COMMIT]:
            raise ValueError(f"{freeze} names NeMo RL commits {commits}, not exactly the pinned {NEMO_RL_COMMIT} (the "
                             "image's /opt/nemo-rl)")
        fw = launch.get("framework") if isinstance(launch.get("framework"), dict) else {}
        if (fw.get("name"), fw.get("version"), fw.get("commit")) != (self.framework_name, NEMO_RL_COMMIT, NEMO_RL_COMMIT):
            raise ValueError(f"{freeze} names NeMo RL {NEMO_RL_COMMIT}, but the launch record claims framework "
                             f"{fw.get('name')!r} version {fw.get('version')!r} commit {fw.get('commit')!r} (NeMo RL is "
                             "pinned by commit: version and commit are the commit)")
        return {"name": self.framework_name, "version": NEMO_RL_COMMIT, "freeze_sha256": file_sha256(freeze)}

    def script_problems(self, stage_out: Path, launch: dict) -> list[Problem]:
        """EXT-83: run_args.json `script_sha256` (run_vlm_nemo.py hashes itself) is the launch record's."""
        args, why = read_run_args(stage_out)
        got = args.get("script_sha256") if isinstance(args, dict) else None
        if got != launch.get("script_sha256"):
            return [Problem("EXT-LAUNCH", "--launch-record.script_sha256", f"run_args.json script_sha256 {got!r} is not "
                            f"the launch record's {launch.get('script_sha256')!r}{f' ({why})' if why else ''} (EXT-83)",
                            "collect the outputs of the run launched with this record's training script")]
        return []

    BASE_FIX = ("train from the runtime's base directory (run_vlm_nemo.py --base <area7-vlm>/models/qwen3.5-0.8b) and "
                "gate against that directory")
    DATA_FIX = ("train on the runtime's data/train.jsonl (the concatenation of the spec's shards data/shards/) inside "
                "the read-only runtime mount")

    def mount_binding_problems(self, mounts: Any, *, base: Path, data: Path, base_field: str, data_field: str,
                               flag: str = "--launch-record") -> list[Problem]:
        """`ext-launch --spec` (optional for this adapter): the spec's base directory and its training shards lie inside
        read-only mounts (EXT-211)."""
        from gatedtrain.ext.launch import _ro_table, _within

        table = _ro_table(mounts)
        out = []
        if not any(_within(str(Path(base).resolve()), host) for _, host in table):
            out.append(Problem("EXT-BASE", base_field, f"{flag} mounts no read-only directory that is or holds the "
                               f"spec's base directory {base}", self.BASE_FIX))
        if not any(_within(str(Path(data).resolve()), host) for _, host in table):
            out.append(Problem("EXT-DATA", data_field, f"{flag} mounts no read-only directory that is or holds the "
                               f"spec's training data {data}", self.DATA_FIX))
        return out

    def _base_files_problem(self, ref: dict, base_dir: Path, where: str) -> str | None:
        """None when base_ref.json names the pinned base and its `files` are the base directory's files (MODELS.json
        shape {name: {bytes, sha256}})."""
        if (ref.get("repo"), ref.get("revision")) != (BASE_REPO, BASE_REVISION):
            return (f"{where} names base {ref.get('repo')!r} revision {ref.get('revision')!r}, not {BASE_REPO} @ "
                    f"{BASE_REVISION} (a branch name is refused, MM-18)")
        try:
            entries = self._base_entries(base_dir)
        except (OSError, ValueError):
            return f"the base directory {base_dir} cannot be hashed"
        want = {e["path"]: {"bytes": e["size"], "sha256": "sha256:" + e["sha256"]} for e in entries}
        if ref.get("files") != want:
            return f"{where} files are not the SHA-256 and sizes of the base directory {base_dir} (MM-21)"
        if ref.get("init_checkpoint") is not None:
            return f"{where} init_checkpoint is {ref.get('init_checkpoint')!r}: the VLM run starts from the base"
        return None

    def checks(self, *, last_cut: Path, base: Path, launch: dict, base_field: str, stage_out: Path | None = None,
               data: Path | None = None, data_field: str = "data", **_: Any) -> list[Problem]:
        """EXT-205: base and data binding, the launch the record names, seed, profile, last cut, script, helper files,
        memory rung and NeMo RL settings."""
        from gatedtrain.ext.launch import container_file

        stage_out = Path(stage_out) if stage_out is not None else Path(last_cut).parent.parent
        args, why = read_run_args(stage_out)
        if args is None:
            return [Problem("EXT-BASE", base_field, f"{why}, so the run cannot be bound to the spec's base (EXT-205)",
                            self.BASE_FIX)]
        out: list[Problem] = []
        mounts = launch.get("mounts")
        if args.get("stage") != STAGE:
            return [Problem("EXT-STAGE", "--stage-out", f"run_args.json stage is {args.get('stage')!r}, not {STAGE} "
                            f"({TRAINER_ID})", "collect a run of run_vlm_nemo.py")]
        cuts = [d for _, d in self.find_cuts(stage_out)[0]] or [Path(last_cut)]
        cut_docs = []
        for d in cuts:
            doc, _ = read_cut_marker(d)
            cut_docs.append((Path(d).name, doc or {}))

        # MM-21: the base the run loaded, through a read-only mount, and every cut's base_ref.json.
        base_model = args.get("base_model")
        hit = container_file(f"{base_model.rstrip('/')}/config.json", mounts) if isinstance(base_model, str) else None
        base_dir = Path(hit[1]).parent if hit else None
        if base_dir is None or base_dir.resolve() != Path(base).resolve():
            out.append(Problem("EXT-BASE", base_field, f"run_args.json base_model {base_model!r} is not, through a "
                               f"read-only mount, the spec's base {base} (MM-21)", self.BASE_FIX))
        if args.get("model_name_or_path") != base_model:
            out.append(Problem("EXT-BASE", base_field, f"run_args.json model_name_or_path "
                               f"{args.get('model_name_or_path')!r} is not its base_model {base_model!r}", self.BASE_FIX))
        if args.get("init_checkpoint") is not None:
            out.append(Problem("EXT-BASE", base_field, f"run_args.json init_checkpoint is "
                               f"{args.get('init_checkpoint')!r}: the VLM run starts from the base", self.BASE_FIX))
        for name, _doc in cut_docs:
            ref, why = _read_json(Path(stage_out) / CUTS_DIR / name / BASE_REF)
            if not isinstance(ref, dict):
                out.append(Problem("EXT-BASE", base_field, f"{name}/{BASE_REF}: {why or 'not an object'}", self.BASE_FIX))
                continue
            if base_dir is not None:
                bad = self._base_files_problem(ref, base_dir, f"{name}/{BASE_REF}")
                if bad:
                    out.append(Problem("EXT-BASE", base_field, bad, self.BASE_FIX))

        # EXT-65 analogue: the file the run read is the concatenation of the spec's shards, in the mount.
        if data is not None:
            out += self._data_problems(args, Path(data), mounts, data_field)

        out += self._launch_binding_problems(args, launch)

        # Seed, profile, dry run, steps, memory rung, script, helpers, NeMo RL settings, cut fields.
        claimed, field = launch.get("training_seed"), "--launch-record.training_seed"
        seed = args.get("seed")
        if not (_int(seed) and seed == claimed):
            out.append(Problem("EXT-LAUNCH", field, f"the launch record claims training_seed {claimed!r}, but "
                               f"run_args.json `seed` is {seed!r} (MM-29)", "launch with the seed the record names"))
        if args.get("profile") != launch.get("profile"):
            out.append(Problem("EXT-PROFILE", "--launch-record.profile", f"run_args.json `profile` is "
                               f"{args.get('profile')!r}, the launch record's {launch.get('profile')!r}",
                               "collect the run launched with this record's profile"))
        if args.get("dry_run") is not False:
            out.append(Problem("EXT-STAGE", "--stage-out", f"run_args.json dry_run is {args.get('dry_run')!r}, not false",
                               "collect a real training run"))
        if not (_int(args.get("attempt")) and args["attempt"] >= 1):
            out.append(Problem("EXT-STAGE", "--stage-out", f"run_args.json attempt is {args.get('attempt')!r}, not a "
                               "positive integer", "collect the run as run_vlm_nemo.py wrote it"))
        if args.get("memory_rung") not in MEMORY_RUNGS:
            out.append(Problem("EXT-STAGE", "--stage-out", f"run_args.json memory_rung is {args.get('memory_rung')!r}, "
                               f"not one of {list(MEMORY_RUNGS)} (run_vlm_nemo.py MEMORY)",
                               "collect the run as run_vlm_nemo.py wrote it"))
        m = CHECKPOINT_DIR_RE.match(Path(last_cut).name)
        last = int(m.group(1)) if m else None
        prof = args.get("profile_settings") if isinstance(args.get("profile_settings"), dict) else {}
        if not (_int(args.get("max_steps")) and args["max_steps"] == last and prof.get("steps") == last):
            out.append(Problem("EXT-STAGE", "--stage-out", f"the last cut is step {last}, but run_args.json max_steps is "
                               f"{args.get('max_steps')!r} (profile_settings.steps {prof.get('steps')!r}): the run did "
                               "not save its last step", "collect a run that finished and saved its last step"))
        if args.get("script_sha256") != launch.get("script_sha256"):
            out.append(Problem("EXT-LAUNCH", "--launch-record.script_sha256", f"run_args.json script_sha256 "
                               f"{args.get('script_sha256')!r} is not the launch record's "
                               f"{launch.get('script_sha256')!r} (EXT-83)",
                               "collect the run launched with this record's training script"))
        script = container_file(launch.get("script_path"), mounts) if isinstance(launch.get("script_path"), str) else None
        for key, fname in HELPERS.items():
            host = Path(script[1]).parent / fname if script else None
            now = _sha_or_none(host) if host is not None else None
            if now is None or args.get(key) != now:
                out.append(Problem("EXT-LAUNCH", "--launch-record.script_path", f"run_args.json {key} "
                                   f"{args.get(key)!r} is not the SHA-256 of {fname} beside the launched script "
                                   f"({now!r}): the run read other code or prompt than the record binds (EXT-83)",
                                   "collect the run launched from this runtime directory, unchanged since"))
        fw = args.get("framework") if isinstance(args.get("framework"), dict) else {}
        if fw.get("commit") != NEMO_RL_COMMIT:
            out.append(Problem("EXT-FREEZE", "--stage-out", f"run_args.json framework.commit is {fw.get('commit')!r}, not "
                               f"the pinned NeMo RL {NEMO_RL_COMMIT}", "train in the pinned NeMo RL image"))
        ov = args.get("nemo_rl_overrides")
        for want in REQUIRED_OVERRIDES:
            if not (isinstance(ov, list) and want in ov):
                out.append(Problem("EXT-STAGE", "--stage-out", f"run_args.json nemo_rl_overrides do not hold {want!r}: "
                                   "the observed logs assume file order and the synchronous grpo_train",
                                   "train with run_vlm_nemo.py's overrides"))
        for name, doc in cut_docs:
            for key, want, rule, fld in (("seed", claimed, "EXT-LAUNCH", field),
                                         ("profile", launch.get("profile"), "EXT-PROFILE", "--launch-record.profile"),
                                         ("stage", STAGE, "EXT-STAGE", "--stage-out"),
                                         ("script_sha256", launch.get("script_sha256"), "EXT-LAUNCH",
                                          "--launch-record.script_sha256")):
                if doc.get(key) != want or (key == "seed" and not _int(doc.get(key))):
                    out.append(Problem(rule, fld, f"{name}/cut.json {key} is {doc.get(key)!r}, not {want!r}",
                                       "collect the cuts of the run launched with this record"))
        return out

    def _launch_binding_problems(self, args: dict, launch: dict) -> list[Problem]:
        """The record's launch is the run the outputs show (EXT-198 analogue): its argv runs run_vlm_nemo.py with
        `--seed` = run_args.json seed, `--profile` = its profile, `--memory` = its memory_rung and, when given,
        `--max-steps` = its max_steps."""
        fix = "collect the run with the launch record ext-launch wrote for it"
        fld = "--launch-record.argv"
        argv = launch.get("argv") if isinstance(launch.get("argv"), list) else []
        script = argv[1].rsplit("/", 1)[-1] if len(argv) > 1 and isinstance(argv[1], str) else None
        if script != SCRIPT_TRAIN:
            return [Problem("EXT-LAUNCH", fld, f"the record's argv[1] {argv[1] if len(argv) > 1 else None!r} is not "
                            f"{SCRIPT_TRAIN}: the outputs are a training run", fix)]
        given = {o: v for o, v, _ in parse_options([a for a in argv[2:] if isinstance(a, str)], TRAIN_OPTIONS)}
        out = []
        for opt, key in (("--seed", "seed"), ("--profile", "profile"), ("--memory", "memory_rung")):
            if given.get(opt) != str(args.get(key)):
                out.append(Problem("EXT-LAUNCH", fld, f"the record's {opt} {given.get(opt)!r} is not run_args.json {key} "
                                   f"{args.get(key)!r}", fix))
        if "--max-steps" in given:
            if given["--max-steps"] != str(args.get("max_steps")):
                out.append(Problem("EXT-LAUNCH", fld, f"the record's --max-steps {given['--max-steps']!r} is not "
                                   f"run_args.json max_steps {args.get('max_steps')!r}", fix))
        elif args.get("profile") in PROFILES and args.get("max_steps") != PROFILES[args["profile"]]["steps"]:
            out.append(Problem("EXT-LAUNCH", fld, f"the record has no --max-steps, so profile {args['profile']} trains "
                               f"{PROFILES[args['profile']]['steps']} steps, but run_args.json max_steps is "
                               f"{args.get('max_steps')!r}", fix))
        return out

    def _data_problems(self, args: dict, data: Path, mounts: Any, field: str) -> list[Problem]:
        from gatedtrain.ext.launch import container_file

        out = []
        if data.name != "shards":
            return [Problem("EXT-DATA", field, f"the spec's training data {data} is not <runtime>/data/shards",
                            self.DATA_FIX)]
        concat = data.parent / "train.jsonl"
        path = args.get("train_file")
        hit = container_file(path, mounts) if isinstance(path, str) else None
        if hit is None or Path(hit[1]).resolve() != concat.resolve():
            out.append(Problem("EXT-DATA", field, f"run_args.json train_file {path!r} is not, through a read-only mount, "
                               f"{concat} (EXT-205)", self.DATA_FIX))
            return out
        now = _sha_or_none(concat)
        try:
            shards = shards_concat_sha256(data)
        except (OSError, ValueError) as exc:
            return out + [Problem("EXT-DATA", field, f"cannot hash the training shards ({exc.__class__.__name__})",
                                  "keep the training dataset readable")]
        if args.get("train_file_sha256") != now:
            out.append(Problem("EXT-DATA", field, f"run_args.json train_file_sha256 {args.get('train_file_sha256')!r} is "
                               f"not the SHA-256 of {concat} ({now})", self.DATA_FIX))
        if now != shards:
            out.append(Problem("EXT-DATA", field, f"{concat} is not the concatenation of the spec's shards {data} in "
                               "file-name order: the run read other rows than the spec names", self.DATA_FIX))
        return out

    def sampler_problems(self, cuts: list[Path], sampler: Any) -> list[Problem]:
        """EXT-205: every cut's deployment sampler (sampler.json, a component) is the registered sampler without the
        evaluator's keys (backend, prompt_block_sha256), and the registered prompt hash is the prompt file the training
        used."""
        from gatedtrain.ext import arm

        out = []
        if not isinstance(sampler, dict):
            return [Problem("EXT-SAMPLER", "registration.sampler", "the registered sampler is not a mapping",
                            "register score_vqa.py's sampler")]
        if sampler.get("prompt_block_sha256") != PROMPT_BLOCK_SHA256:
            out.append(Problem("EXT-SAMPLER", "registration.sampler.prompt_block_sha256", f"the registered "
                               f"prompt_block_sha256 {sampler.get('prompt_block_sha256')!r} is not the prompt file "
                               f"{PROMPT_BLOCK_SHA256} the training used (prompt_clevr_cot.txt)",
                               "register the evaluation of the area's prompt file"))
        want = {k: v for k, v in sampler.items() if k not in EVAL_ONLY_SAMPLER_KEYS}
        for d in cuts:
            doc, _ = _read_json(Path(d) / SAMPLER_FILE)
            if not arm.same_json(doc, want):
                out.append(Problem("EXT-SAMPLER", "registration.sampler", f"{Path(d).name}/{SAMPLER_FILE} is not the "
                                   "registered sampler without backend and prompt_block_sha256: the cut deploys another "
                                   "sampler than the gate evaluates", "register gen_vqa.py's SAMPLER the cuts carry"))
        return out

    # --- launch side (EXT-209, EXT-210) ---

    @staticmethod
    def _script(argv: Any) -> tuple[str | None, int]:
        """(the script's file name, its argv index): argv[2] after score_vqa.py's `-I`, else argv[1]."""
        if not (isinstance(argv, list) and len(argv) > 1 and all(isinstance(a, str) for a in argv)):
            return None, 1
        i = 2 if (len(argv) > 2 and argv[1] == "-I" and argv[2].rsplit("/", 1)[-1] in FLAGGED) else 1
        return argv[i].rsplit("/", 1)[-1], i

    def argv_problems(self, argv: Any, *, profile: str | None = None) -> list[str]:
        """ext-launch, --recheck and ext-collect. The script is one of the runtime's at its container path:
        run_vlm_nemo.py (`--profile` the launch's, `--out /out`, an integer `--seed`, `--base` the runtime's base,
        `--memory` a, b or c, `--max-steps` a positive integer and only under the smoke profile), gen_vqa.py (`--model`
        the base or /inputs/cut, `--items` items-valA or items-valB, `--out /out`) or `-I score_vqa.py` (`--gen
        /inputs/gen`, `--items` one of the two, `--out /out`, `--evaluator-id` the item set's evaluator, no
        `--self-test`); only score_vqa.py takes `-I`; `--limit` is refused under `--profile full`. Options are resolved
        as argparse resolves them (a unique prefix counts)."""
        if not (isinstance(argv, list) and len(argv) > 1 and all(isinstance(a, str) for a in argv)):
            return []                                     # bind_argv reports the shape
        script, si = self._script(argv)
        if script not in OPTIONS or argv[si] != SCRIPT_PATHS[script]:
            return [f"argv[{si}] {argv[si]!r} is not one of the runtime's scripts {sorted(SCRIPT_PATHS.values())} (a "
                    "copy elsewhere imports other helpers and data)"]
        if (script in FLAGGED) != (si == 2):
            return [f"{script} runs as `python -I {SCRIPT_PATHS[script]}`" if script in FLAGGED else
                    f"{script} follows the interpreter directly"]
        given: dict[str, str | None] = {}
        for opt, value, _ in parse_options(argv[si + 1:], OPTIONS[script]):
            given[opt] = value                            # argparse keeps the last value
        out = []

        def need(opt: str, allowed: tuple, what: str) -> None:
            if given.get(opt) not in allowed:
                out.append(f"{script} {opt} {given.get(opt)!r} is not {what}")

        items = {f"{RUNTIME}/{s}": eid for s, eid in ITEM_SETS.items()}
        if script == SCRIPT_TRAIN:
            need("--profile", (profile,) if profile in PROFILES else tuple(PROFILES), f"the launch profile {profile!r}")
            need("--out", ("/out",), "/out")
            if not re.fullmatch(r"[0-9]+", given.get("--seed") or ""):
                out.append(f"{script} --seed {given.get('--seed')!r} is not an integer")
            need("--base", (f"{RUNTIME}/{BASE_REL}",), f"the runtime's base {RUNTIME}/{BASE_REL}")
            need("--memory", MEMORY_RUNGS, f"a memory rung {list(MEMORY_RUNGS)}")
            if "--max-steps" in given:
                if not re.fullmatch(r"[1-9][0-9]*", given["--max-steps"] or ""):
                    out.append(f"{script} --max-steps {given['--max-steps']!r} is not a positive integer")
                elif profile == "full" or given.get("--profile") == "full":
                    out.append(f"{script} --max-steps under --profile full: the full profile trains its "
                               f"{PROFILES['full']['steps']} steps")
        elif script == SCRIPT_GEN:
            need("--model", (f"{RUNTIME}/{BASE_REL}", EVAL_CUT_CONTAINER), f"the runtime's base or {EVAL_CUT_CONTAINER}")
            need("--items", tuple(items), "one of the runtime's item sets items-valA, items-valB")
            need("--out", ("/out",), "/out")
        else:
            need("--gen", (GEN_CONTAINER,), GEN_CONTAINER)
            need("--items", tuple(items), "one of the runtime's item sets items-valA, items-valB")
            need("--out", ("/out",), "/out")
            if "--self-test" in given:
                out.append(f"{script} --self-test: the scoring step scores a generation")
            eid = given.get("--evaluator-id", EVALUATOR_ID)
            if given.get("--items") in items and eid != items[given["--items"]]:
                out.append(f"{script} --evaluator-id {eid!r} is not {items[given['--items']]!r}, the evaluator of "
                           f"{given['--items']} (score_vqa.py's default is {EVALUATOR_ID})")
        if script in (SCRIPT_GEN, SCRIPT_SCORE) and "--limit" in given and profile == "full":
            out.append(f"{script} --limit {given['--limit']!r} under --profile full: a gate arm evaluates the whole "
                       "registered item set; --limit is for the smoke arms")
        return out

    def launch_problems(self, doc: dict, *, kind: str) -> list[tuple[str, str]]:
        """[(field, message)] of this adapter's launch rules (request or record): all three scripts run in a pinned NeMo
        RL image (`container`), with network none and no sidecar; training and generation on the GPU, scoring on CPU
        with a tmpfs /tmp; each script has exactly its mount set (the runtime read-only at its own path, the output at
        /out; generation of a cut adds it at /inputs/cut, scoring the generation output at /inputs/gen); the training
        seed is the record's; the framework is NeMo RL pinned by commit."""
        out: list[tuple[str, str]] = []
        argv = doc.get("argv")
        script, si = self._script(argv)
        rt = doc.get("runtime") if isinstance(doc.get("runtime"), dict) else {}
        fw = doc.get("framework") if isinstance(doc.get("framework"), dict) else {}
        if (fw.get("version"), fw.get("commit")) != (NEMO_RL_COMMIT, NEMO_RL_COMMIT):
            out.append(("framework", f"framework version {fw.get('version')!r} and commit {fw.get('commit')!r} are not "
                        f"the pinned NeMo RL commit {NEMO_RL_COMMIT} (pinned by commit)"))
        if script not in OPTIONS:
            return out                                    # bind_argv and argv_problems report the script
        if rt.get("kind") != "container" or rt.get("image") not in NEMO_RL_IMAGES:
            out.append(("runtime", f"{script} runs in a pinned NeMo RL image: runtime {{kind: container, image: one of "
                        f"{list(NEMO_RL_IMAGES)}}}, not {rt.get('kind')!r} {rt.get('image')!r}"))
        if doc.get("network") != "none":
            out.append(("network", f"{script} runs with network none (the image holds every input), not "
                        f"{doc.get('network')!r}"))
        if "sidecar" in doc:
            out.append(("sidecar", f"{script} pairs no sidecar (area 7 runs no model-written code)"))
        if script == SCRIPT_SCORE:
            if doc.get("gpus") is not False:
                out.append(("gpus", f"{script} runs on CPU (gpus false)"))
            if doc.get("tmpfs") != ["/tmp"]:
                out.append(("tmpfs", f"tmpfs {doc.get('tmpfs')!r} is not ['/tmp']"))
        elif doc.get("gpus") is not True:
            out.append(("gpus", f"{script} needs the GPU (gpus true)"))
        given = {o: v for o, v, _ in parse_options(argv[si + 1:], OPTIONS[script])}
        if script == SCRIPT_TRAIN:
            seed = doc.get("training_seed")
            if _int(seed) and given.get("--seed") != str(seed):
                out.append(("argv", f"{SCRIPT_TRAIN} --seed {given.get('--seed')!r} is not the training_seed {seed} "
                            "(MM-29)"))
        want = [(RUNTIME, "ro"), ("/out", "rw")]
        if script == SCRIPT_GEN and given.get("--model") == EVAL_CUT_CONTAINER:
            want.append((EVAL_CUT_CONTAINER, "ro"))
        elif script == SCRIPT_SCORE:
            want.append((GEN_CONTAINER, "ro"))
        mounts = doc.get("mounts") if isinstance(doc.get("mounts"), list) else []
        got = sorted((_norm_path(m.get("container")), m.get("mode")) for m in mounts if isinstance(m, dict))
        if got != sorted(want) or len(got) != len(mounts):
            out.append(("mounts", f"{script} mounts {got}, not exactly {sorted(want)}"))
        return out
