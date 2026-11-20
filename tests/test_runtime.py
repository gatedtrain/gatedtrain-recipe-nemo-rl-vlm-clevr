"""The runtime code that runs inside the NeMo RL trainer container (runtime/vlm_env_logged.py, runtime/run_vlm_nemo.py,
runtime/ptc_vlm_hooks.py) on CPU, without NeMo RL, Ray, a GPU or a model: the NeMo RL modules they import are
stand-ins in sys.modules that record what they are handed, and the training rows are the ones make_data.py writes (its
`main`, run over stand-in parquet shards with tiny PNG images) or the probe-a6 shard rows it wrote for the GPU run.

Why: a fault in this code shows first at step 1 of a GPU run. These tests pin the row-id processor's conversion of the
rows make_data.py writes (list contents, the null keys the JSON loader adds), the environment's per-sample log, the
launcher's wiring of processor, environment and overrides, the cut writer on tiny weight files, and that the prompt,
the image and the answer fit the trainer's sequence length.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import struct
import sys
import tempfile
import types
import unittest
import zlib
from pathlib import Path

import _nemo_rl_vlm as V
from gatedtrain_recipe_nemo_rl_vlm_clevr import adapter as A

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / "runtime"
sys.path.insert(0, str(RUNTIME))
import gen_vqa  # noqa: E402
import make_data  # noqa: E402
import ptc_vlm_hooks as hooks  # noqa: E402
import run_vlm_nemo  # noqa: E402
import score_vqa  # noqa: E402
import vlm_env_logged  # noqa: E402

PROBE_SHARD = V.PROBE / "data" / "shards" / "shard-00000.jsonl"


def probe_rows(n: int = 3) -> list[dict]:
    """The first `n` rows make_data.py wrote for the probe-a6 GPU run (absolute image paths of that host)."""
    return [json.loads(x) for x in PROBE_SHARD.read_text(encoding="utf-8").splitlines()[:n]]


def stub_modules(test: unittest.TestCase, mods: dict[str, types.ModuleType]) -> None:
    """Put `mods` (and empty parent packages) into sys.modules until the test ends."""
    full = dict(mods)
    for name in list(mods):
        parts = name.split(".")
        for i in range(1, len(parts)):
            full.setdefault(".".join(parts[:i]), types.ModuleType(".".join(parts[:i])))
    saved = {k: sys.modules.get(k) for k in full}
    sys.modules.update(full)

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    test.addCleanup(restore)


def module(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def tiny_png(seed: int) -> bytes:
    """A 2 x 2 RGB PNG whose bytes depend on `seed` (standard library only)."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes([seed % 256, (seed * 7) % 256, 3]) * 2 for _ in range(2))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def shape(x):
    """The structure of a JSON value: dict keys and list lengths, with the scalars' types."""
    if isinstance(x, dict):
        return {k: shape(v) for k, v in x.items()}
    if isinstance(x, list):
        return [shape(v) for v in x]
    return type(x).__name__


def arrow_loaded(row: dict) -> dict:
    """The row as NeMo RL's JSON loader hands it to the processor: Arrow infers one struct type for every message's
    `content` parts, so each part carries the union of the part keys, absent ones as None."""
    row = json.loads(json.dumps(row))
    keys = {k for m in row["messages"] for p in m["content"] for k in p}
    for m in row["messages"]:
        m["content"] = [{k: p.get(k) for k in sorted(keys)} for p in m["content"]]
    return row


class MakeDataRun:
    """make_data.main over stand-in parquet shards (a `pyarrow.parquet` stand-in), in a scratch runtime directory."""

    QUESTIONS = ("How many cubes are there?", "Is there a red sphere?", "What number of small things are there?",
                 "Are there more cubes than spheres?", "How many objects are left of the cylinder?")
    ANSWERS = ("3", "yes", "0", "no", "10")

    def __init__(self, test: unittest.TestCase, n_train: int = 4) -> None:
        self.root = Path(tempfile.mkdtemp())
        test.addCleanup(shutil.rmtree, self.root)
        raw = self.root / "raw"
        counts = {"valA": 3, "valB": 3, "trainA": n_train + 1}
        self.tables: dict[str, list[dict]] = {}
        splits = {}
        seed = 0
        for split, n in counts.items():
            f = raw / split / "data" / f"{split}-00000-of-00001.parquet"
            f.parent.mkdir(parents=True)
            f.write_bytes(f"stand-in parquet {split}".encode())
            rows = []
            for i in range(n):
                seed += 1
                rows.append({"problem": f" {self.QUESTIONS[i % 5]} ",
                             "solution": f"<answer> {self.ANSWERS[i % 5]} </answer>",
                             "image": {"bytes": tiny_png(seed), "path": f"CLEVR_{split}_{i:06d}.png"}})
            self.tables[f.name] = rows
            splits[split] = {"files": {f"data/{f.name}": {"sha256": make_data.sha256_file(f)}}, "repo": "stand-in"}
        (raw / "RAW.json").write_text(json.dumps({"license": "CC BY 4.0", "splits": splits}))
        parquet = module("pyarrow.parquet", read_table=lambda f: types.SimpleNamespace(
            to_pylist=lambda: [dict(r) for r in self.tables[Path(f).name]]))
        stub_modules(test, {"pyarrow.parquet": parquet})
        sys.modules["pyarrow"].parquet = parquet
        for k, v in {"HERE": self.root, "RAW": raw, "DATA": self.root / "data", "N_TRAIN": n_train, "N_ITEMS": 2,
                     "ROWS_PER_SHARD": 3}.items():
            old = getattr(make_data, k)
            setattr(make_data, k, v)
            test.addCleanup(setattr, make_data, k, old)
        with contextlib.redirect_stdout(io.StringIO()):
            assert make_data.main(["--seed", "1"]) == 0
        self.train = self.root / "data" / "train.jsonl"
        self.rows = [json.loads(x) for x in self.train.read_text(encoding="utf-8").splitlines()]


class TestMakeDataRows(unittest.TestCase):
    def test_rows_have_the_shape_of_the_rows_the_gpu_run_trained_on(self):
        rows = MakeDataRun(self).rows
        self.assertEqual(len(rows), 4)
        real = probe_rows(1)[0]
        for row in rows:
            self.assertEqual(shape(row), shape(real))
            user, assistant = row["messages"]
            self.assertEqual([p["type"] for p in user["content"]], ["image", "text"])   # check_images reads part 0
            self.assertTrue(Path(user["content"][0]["image"]).is_absolute())
            self.assertEqual(make_data.sha256_file(Path(user["content"][0]["image"])), row["image_sha256"])
            self.assertEqual(row["input"], f"{row['image_sha256']}\n{user['content'][1]['text']}")
            self.assertEqual(len(assistant["content"]), 1)

    def test_probe_rows_are_lists_with_the_image_first(self):
        for row in probe_rows(50):
            self.assertEqual([p["type"] for p in row["messages"][0]["content"]], ["image", "text"])
            self.assertEqual([p["type"] for p in row["messages"][1]["content"]], ["text"])


class TestRowIdProcessor(unittest.TestCase):
    """clevr_rowid_processor against a stand-in `nemo_rl.data.processors.vlm_hf_data_processor` that does what NeMo
    RL's does with the datum (processors.py:617-700: formats each text part with the prompt, resolves each image part,
    takes `messages[1].content` as the environment's ground truth) and records it."""

    def setUp(self):
        self.seen = []
        self.prompt = run_vlm_nemo.PROMPT.read_text(encoding="utf-8")

        def vlm_hf_data_processor(datum, spec, processor, max_seq_length, idx):
            self.seen.append(json.loads(json.dumps(datum)))
            parts = []
            for c in datum["messages"][0]["content"]:
                if c["type"] == "text":
                    parts.append({"type": "text", "text": spec.prompt.format(c["text"])})
                elif c["type"] == "image":
                    if not Path(c["image"]).is_file():
                        raise FileNotFoundError(c["image"])
                    parts.append(c)
            return {"message_log": [{"role": "user", "content": parts}], "length": 100, "idx": idx,
                    "extra_env_info": {"ground_truth": datum["messages"][1]["content"]}, "loss_multiplier": 1.0}

        stub_modules(self, {"nemo_rl.data.processors": module("nemo_rl.data.processors",
                                                              vlm_hf_data_processor=vlm_hf_data_processor)})
        self.spec = types.SimpleNamespace(prompt=self.prompt)

    def process(self, row: dict, idx: int = 0) -> dict:
        return vlm_env_logged.clevr_rowid_processor(row, self.spec, None, run_vlm_nemo.MAX_SEQ, idx)

    def test_make_data_rows_as_the_loader_hands_them(self):
        rows = MakeDataRun(self).rows
        for i, row in enumerate(rows):
            loaded = arrow_loaded(row)
            self.assertIn(None, loaded["messages"][0]["content"][0].values())     # the loader's null keys
            before = json.dumps(loaded, sort_keys=True)
            out = self.process(loaded, i)
            user, assistant = self.seen[-1]["messages"]
            answer = row["messages"][1]["content"][0]["text"]
            self.assertEqual(out["extra_env_info"], {"ground_truth": answer, "row_id": row["row_id"]})
            self.assertEqual(assistant, {"role": "assistant", "content": answer})
            self.assertEqual(user["content"], row["messages"][0]["content"])       # null keys dropped, order kept
            self.assertEqual(out["message_log"][0]["content"][1]["text"],
                             self.prompt.format(row["messages"][0]["content"][1]["text"]))
            self.assertEqual(json.dumps(loaded, sort_keys=True), before)          # the loaded row is not changed
            self.assertEqual(self.seen[-1]["row_id"], row["row_id"])

    def test_probe_rows(self):
        # The GPU run's own rows (their images lie on that host): only the conversion, the processor is not called
        # with missing images.
        for row in probe_rows():
            loaded = arrow_loaded(row)
            for part in loaded["messages"][0]["content"]:
                if part["type"] == "image":
                    part["image"] = str(V.PROBE / "run_args.json")                # any existing file
            out = self.process(loaded)
            self.assertEqual(out["extra_env_info"]["ground_truth"], row["messages"][1]["content"][0]["text"])
            self.assertIsInstance(out["extra_env_info"]["ground_truth"], str)
            self.assertTrue(all(v is not None for p in self.seen[-1]["messages"][0]["content"] for v in p.values()))

    def test_string_answer_passes_and_other_answers_are_refused(self):
        row = MakeDataRun(self).rows[0]
        plain = json.loads(json.dumps(row))
        plain["messages"][1]["content"] = "7"
        self.assertEqual(self.process(plain)["extra_env_info"]["ground_truth"], "7")
        for bad in ([{"type": "text", "text": "1"}, {"type": "text", "text": "2"}], [{"type": "image", "image": "x"}]):
            broken = json.loads(json.dumps(row))
            broken["messages"][1]["content"] = bad
            with self.assertRaisesRegex(ValueError, row["row_id"]):
                self.process(broken)


class TestCheckImages(unittest.TestCase):
    def test_checks_every_image_by_hash(self):
        run = MakeDataRun(self)
        self.assertEqual(run_vlm_nemo.check_images(run.train), len({r["image_sha256"] for r in run.rows}))
        path = Path(run.rows[1]["messages"][0]["content"][0]["image"])
        path.write_bytes(tiny_png(250))
        with self.assertRaisesRegex(SystemExit, run.rows[1]["row_id"]):
            run_vlm_nemo.check_images(run.train)

    def test_image_not_first_fails_loudly(self):
        # check_images reads content part 0 (make_data.py writes the image there); another order must not pass.
        run = MakeDataRun(self)
        swapped = run.root / "swapped.jsonl"
        rows = json.loads(json.dumps(run.rows))
        for r in rows:
            r["messages"][0]["content"].reverse()
        swapped.write_text("".join(json.dumps(r) + "\n" for r in rows))
        with self.assertRaises(KeyError):
            run_vlm_nemo.check_images(swapped)


class FakeTensor(list):
    def tolist(self):
        return list(self)


class TestLoggedEnvironment(unittest.TestCase):
    """LoggedVLMEnvironment.step, vlm_env_logged.py loaded under stand-in `ray` and `nemo_rl.environments.
    vlm_environment` modules; the stand-in VLMEnvironment.step joins the assistant contents as NeMo RL's does
    (vlm_environment.py:166-174) and rewards an exact answer."""

    def setUp(self):
        self.scored = []
        scored = self.scored

        class VLMEnvironmentActor:
            def __init__(self, cfg):
                self.cfg = cfg

            def step(self, message_log_batch, metadata, return_extracted_answer=False):
                texts = ["".join(m["content"] for m in conv if m["role"] == "assistant") for conv in message_log_batch]
                scored.extend(texts)
                rewards = FakeTensor(float(f"<answer>{md['ground_truth']}</answer>" in t)
                                     for t, md in zip(texts, metadata))
                return types.SimpleNamespace(rewards=rewards, metadata=metadata)

        ray = module("ray", remote=lambda **kw: (lambda cls: cls))
        venv = module("nemo_rl.environments.vlm_environment",
                      VLMEnvironment=types.SimpleNamespace(__ray_actor_class__=VLMEnvironmentActor))
        stub_modules(self, {"ray": ray, "nemo_rl.environments.vlm_environment": venv})
        spec = importlib.util.spec_from_file_location("vlm_env_logged_under_ray", RUNTIME / "vlm_env_logged.py")
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.log = self.tmp / "env_calls.jsonl"

    def batch(self, responses: list):
        user = {"role": "user", "content": [{"type": "image", "image": "i.png"}, {"type": "text", "text": "q"}]}
        logs = [[user, {"role": "assistant", "content": r}] for r in responses]
        md = [{"ground_truth": "3", "row_id": f"clevr-trainA-{i // 2:05d}"} for i in range(len(responses))]
        return logs, md

    def lines(self) -> list[dict]:
        return [json.loads(x) for x in self.log.read_text(encoding="utf-8").splitlines()]

    def test_logs_one_line_per_sample_with_the_scored_text_hash(self):
        self.assertIsNotNone(self.mod.LoggedVLMEnvironment)
        env = self.mod.LoggedVLMEnvironment({"log_path": str(self.log), "num_workers": 2})
        self.assertEqual(env.cfg, {"num_workers": 2})                       # log_path is not handed to NeMo RL
        responses = ["<think>three</think> <answer>3</answer>", "<answer>4</answer>", "", "a \"quoted\"\nreply"]
        ret = env.step(*self.batch(responses))
        self.assertEqual(ret.rewards, [1.0, 0.0, 0.0, 0.0])
        got = self.lines()
        self.assertEqual([x["position"] for x in got], [0, 1, 2, 3])
        self.assertEqual({x["call"] for x in got}, {1})
        self.assertEqual([x["row_id"] for x in got], ["clevr-trainA-00000"] * 2 + ["clevr-trainA-00001"] * 2)
        self.assertEqual([x["reward"] for x in got], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual([x["response_sha256"] for x in got], [V.sha_text(t) for t in self.scored])
        self.assertEqual(self.scored, responses)
        self.assertEqual(sorted(got[0]), ["call", "position", "response_sha256", "reward", "row_id", "utc"])
        self.assertRegex(got[0]["utc"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z$")
        env.step(*self.batch(["<answer>3</answer>"]))
        self.assertEqual([(x["call"], x["position"]) for x in self.lines()[4:]], [(2, 0)])

    def test_the_log_lines_pass_the_adapter_format(self):
        # The adapter's consumption check reads these fields (adapter.py _plan): a SHA-256 and a row id per line.
        env = self.mod.LoggedVLMEnvironment({"log_path": str(self.log), "num_workers": 1})
        env.step(*self.batch(["<answer>3</answer>"]))
        line = self.lines()[0]
        self.assertRegex(line["response_sha256"], r"^sha256:[0-9a-f]{64}$")
        self.assertIsInstance(line["reward"], float)

    def test_list_content_fails_in_nemo_rl_before_anything_is_logged(self):
        # NeMo RL joins assistant contents as strings; a multimodal list never reaches the log's str().
        env = self.mod.LoggedVLMEnvironment({"log_path": str(self.log), "num_workers": 1})
        with self.assertRaises(TypeError):
            env.step(*self.batch([[{"type": "text", "text": "<answer>3</answer>"}]]))
        self.assertFalse(self.log.exists())

    def test_host_import_has_no_actor(self):
        self.assertIsNone(vlm_env_logged.LoggedVLMEnvironment)


class CutCase(unittest.TestCase):
    """A NeMo RL step directory with tiny weight files, the base directory and its MODELS.json."""

    STEP = 5

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.out = self.tmp / "out"
        self.out.mkdir()
        self.base = V.build_base(self.tmp / "models" / "qwen3.5-0.8b")
        V.write_json(self.tmp / "models" / "MODELS.json", V.models_json(self.base))
        self.meta = {"seed": 7100, "profile": "smoke", "stage": "vlm_grpo", "init_checkpoint": None,
                     "script_sha256": "sha256:" + "1" * 64, "examples_consumed": lambda s: s * 16}

    def step_dir(self, weights: dict[str, str], *, index: bool, tokenizer=("tokenizer.json", "tokenizer_config.json",
                                                                             "chat_template.jinja")) -> Path:
        d = self.out / "nemo" / "checkpoints" / f"step_{self.STEP}"
        cons = d / "policy" / "weights" / "model" / "consolidated"
        cons.mkdir(parents=True)
        for name, tag in weights.items():
            (cons / name).write_bytes(V.safetensors_bytes(tag))
        if index:
            V.write_json(cons / V.INDEX, {"metadata": {"total_size": 8 * len(weights)},
                                          "weight_map": {f"w{i}": n for i, n in enumerate(weights)}})
        V.write_json(cons / "config.json", {"architectures": ["Qwen3_5ForConditionalGeneration"]})
        tok = d / "policy" / "tokenizer"
        tok.mkdir(parents=True)
        for name in tokenizer:
            (tok / name).write_text(f"tokenizer {name}\n", encoding="utf-8")
        (d / "training_info.json").write_text(json.dumps({"current_step": self.STEP}), encoding="utf-8")
        return d


class TestWriteCut(CutCase):
    def check_cut(self, final: Path, step_dir: Path) -> dict:
        self.assertEqual(final, self.out / "cuts" / f"checkpoint-{self.STEP}")
        self.assertIsNone(A.NemoRlVlmAdapter().cut_problem(final, self.STEP))
        self.assertEqual(sorted(p.name for p in (self.out / "cuts").iterdir()), [final.name])   # no partial left
        cut = json.loads((final / "cut.json").read_text())
        for name, sha in cut["files"].items():
            self.assertEqual(V.file_sha256(final / name), sha)
        self.assertEqual(cut["deployable"], sorted(n for n in cut["files"]
                                                   if n not in ("base_ref.json", "training_info.json")))
        self.assertEqual((cut["global_step"], cut["examples_consumed"], cut["nemo_rl_step_dir"]),
                         (self.STEP, self.STEP * 16, f"step_{self.STEP}"))
        cons = step_dir / "policy" / "weights" / "model" / "consolidated"
        for name in cut["files"]:
            same = (cons / name).exists() and os.path.samefile(cons / name, final / name)
            self.assertEqual(same, name.endswith(".safetensors") or name in ("config.json", V.INDEX), name)
        self.assertEqual(json.loads((final / "sampler.json").read_text()), gen_vqa.SAMPLER)
        self.assertEqual(gen_vqa.SAMPLER, V.CUT_SAMPLER)
        ref = json.loads((final / "base_ref.json").read_text())
        self.assertEqual((ref["repo"], ref["revision"], ref["init_checkpoint"]), (V.BASE_REPO, V.BASE_REVISION, None))
        self.assertEqual(cut["file_origin"], {
            "tokenizer.json": "nemo_rl policy/tokenizer", "tokenizer_config.json": "nemo_rl policy/tokenizer",
            "chat_template.jinja": "nemo_rl policy/tokenizer",
            "preprocessor_config.json": "base model (vision tower frozen)",
            "video_preprocessor_config.json": "base model (vision tower frozen)"})
        log = [json.loads(x) for x in (self.out / "cuts.jsonl").read_text().splitlines()]
        self.assertEqual([(x["step"], x["cut"]) for x in log], [(self.STEP, final.name)])
        return cut

    def test_one_weights_file_without_index(self):
        sd = self.step_dir({V.CUT_WEIGHTS: "one"}, index=False)
        cut = self.check_cut(hooks.write_cut(sd, self.STEP, self.out, self.base, self.meta), sd)
        self.assertNotIn(V.INDEX, cut["files"])

    def test_shards_with_index(self):
        names = {"model-00001-of-00002.safetensors": "a", "model-00002-of-00002.safetensors": "b"}
        sd = self.step_dir(names, index=True)
        cut = self.check_cut(hooks.write_cut(sd, self.STEP, self.out, self.base, self.meta), sd)
        self.assertTrue(set(names) | {V.INDEX} <= set(cut["files"]))

    def test_same_files_as_the_adapter_fixture_cut(self):
        # tests/_nemo_rl_vlm.build_cut stands in for write_cut in the adapter's scenarios: both must write one layout.
        sd = self.step_dir({V.CUT_WEIGHTS: "one"}, index=True)
        mine = hooks.write_cut(sd, self.STEP, self.out, self.base, self.meta)
        other = V.build_cut(self.tmp / "fixture-run", self.STEP, profile="smoke", seed=7100,
                            script_sha256=self.meta["script_sha256"], base=self.base)
        self.assertEqual(sorted(p.name for p in mine.iterdir()), sorted(p.name for p in other.iterdir()))
        a, b = (json.loads((d / "cut.json").read_text()) for d in (mine, other))
        drop = ("files",)
        self.assertEqual({k: v for k, v in a.items() if k not in drop}, {k: v for k, v in b.items() if k not in drop})

    def test_refusals_leave_no_cut(self):
        cases = [({"model-00001-of-00002.safetensors": "a", "model-00002-of-00002.safetensors": "b"}, False, None,
                  "no index"),
                 ({}, False, None, "no consolidated safetensors"),
                 ({V.CUT_WEIGHTS: "one"}, False, ("tokenizer.json", "tokenizer_config.json"), "chat_template.jinja")]
        for weights, index, tok, msg in cases:
            with self.subTest(msg):
                if (self.out / "nemo").exists():
                    shutil.rmtree(self.out / "nemo")
                if tok is not None:
                    (self.base / "chat_template.jinja").unlink(missing_ok=True)
                sd = self.step_dir(weights, index=index, **({"tokenizer": tok} if tok else {}))
                with self.assertRaisesRegex(RuntimeError, msg):
                    hooks.write_cut(sd, self.STEP, self.out, self.base, self.meta)
                self.assertFalse((self.out / "cuts" / f"checkpoint-{self.STEP}").exists())
                self.assertFalse((self.out / "cuts.jsonl").exists())

    def test_stale_partial_is_replaced(self):
        partial = self.out / "cuts" / f".checkpoint-{self.STEP}.partial"
        partial.mkdir(parents=True)
        (partial / "junk.bin").write_bytes(b"x")
        sd = self.step_dir({V.CUT_WEIGHTS: "one"}, index=False)
        final = hooks.write_cut(sd, self.STEP, self.out, self.base, self.meta)
        self.assertFalse((final / "junk.bin").exists())
        self.check_cut(final, sd)


class TestInstallCutWriter(CutCase):
    """install_cut_writer against a stand-in CheckpointManager whose storage publishes as NeMo RL's local storage
    does (checkpoint_storage.py:80-97: tmp_step_N renamed to step_N)."""

    def test_publish_builds_the_cut(self):
        root = self.out / "nemo" / "checkpoints"

        class Storage:
            def publish(self, path):
                os.rename(path, Path(path).parent / Path(path).name.removeprefix("tmp_"))

        class CheckpointManager:
            def __init__(self, config):
                self.config = config
                self._storage = Storage()

        stub_modules(self, {"nemo_rl.utils.checkpoint": module("nemo_rl.utils.checkpoint",
                                                               CheckpointManager=CheckpointManager)})
        hooks.install_cut_writer(self.out, self.base, self.meta)
        sd = self.step_dir({V.CUT_WEIGHTS: "one"}, index=False)
        tmp = root / f"tmp_step_{self.STEP}"
        os.rename(sd, tmp)
        mgr = CheckpointManager({"checkpoint_dir": str(root)})
        self.assertEqual(mgr.config, {"checkpoint_dir": str(root)})
        mgr._storage.publish(str(tmp))
        self.assertTrue((root / f"step_{self.STEP}").is_dir())
        self.assertIsNone(A.NemoRlVlmAdapter().cut_problem(self.out / "cuts" / f"checkpoint-{self.STEP}", self.STEP))


class TestRunRecord(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.out)

    def test_attempts_count_starts(self):
        self.assertEqual(hooks.record_attempt(self.out, "start", stage="vlm_grpo"), 1)
        self.assertEqual(hooks.record_attempt(self.out, "end", stage="vlm_grpo"), 1)
        self.assertEqual(hooks.record_attempt(self.out, "start", stage="vlm_grpo"), 2)
        lines = [json.loads(x) for x in (self.out / "attempts.jsonl").read_text().splitlines()]
        self.assertEqual([(x["event"], x["attempt"], x["stage"]) for x in lines],
                         [("start", 1, "vlm_grpo"), ("end", 1, "vlm_grpo"), ("start", 2, "vlm_grpo")])

    def test_run_args(self):
        hooks.write_run_args(self.out, {"stage": "vlm_grpo", "path": Path("/x")})
        doc = json.loads((self.out / "run_args.json").read_text())
        self.assertEqual((doc["stage"], doc["path"]), ("vlm_grpo", "/x"))
        self.assertEqual(doc["framework"]["name"], "nemo_rl")
        self.assertIn("created_at", doc)
        self.assertTrue((self.out / "env-freeze.txt").read_text().rstrip().split("\n")[-1].startswith("# nemo-rl commit"))


class TestLauncher(unittest.TestCase):
    """run_vlm_nemo.main with stand-ins for the NeMo RL registries and the example entry point: what it registers
    and the command line it hands NeMo RL."""

    def setUp(self):
        self.run = MakeDataRun(self)
        self.out = self.run.root / "out"
        self.base = V.build_base(self.run.root / "models" / "qwen3.5-0.8b")
        self.registered, self.argv = {}, []
        self.actor_registry, self.env_registry = {}, {}

        def register_processor(name, fn):
            if name in self.registered:
                raise ValueError(name)
            self.registered[name] = fn

        def entry_main():
            self.argv.append(list(sys.argv))
            if getattr(self, "trainer_fails", False):
                raise RuntimeError("trainer failed")

        entry = module("run_vlm_grpo", main=entry_main, __file__="/opt/nemo-rl/examples/run_vlm_grpo.py")
        stub_modules(self, {
            "nemo_rl.data.processors": module("nemo_rl.data.processors", register_processor=register_processor),
            "nemo_rl.distributed.ray_actor_environment_registry": module(
                "nemo_rl.distributed.ray_actor_environment_registry", ACTOR_ENVIRONMENT_REGISTRY=self.actor_registry),
            "nemo_rl.distributed.virtual_cluster": module(
                "nemo_rl.distributed.virtual_cluster", PY_EXECUTABLES=types.SimpleNamespace(SYSTEM="system-python")),
            "nemo_rl.environments.utils": module("nemo_rl.environments.utils", ENV_REGISTRY=self.env_registry),
            "nemo_rl.utils.checkpoint": module("nemo_rl.utils.checkpoint", CheckpointManager=type(
                "CheckpointManager", (), {"__init__": lambda self, config: None})),
            "run_vlm_grpo": entry})
        old = run_vlm_nemo.DATA
        run_vlm_nemo.DATA = self.run.train
        self.addCleanup(setattr, run_vlm_nemo, "DATA", old)
        saved_argv, saved_path, saved_pp = list(sys.argv), list(sys.path), os.environ.get("PYTHONPATH")
        self.addCleanup(setattr, sys, "argv", saved_argv)
        self.addCleanup(setattr, sys, "path", saved_path)
        self.addCleanup(lambda: os.environ.__setitem__("PYTHONPATH", saved_pp) if saved_pp is not None
                        else os.environ.pop("PYTHONPATH", None))

    def main(self, *extra: str) -> int:
        return run_vlm_nemo.main(["--profile", "smoke", "--out", str(self.out), "--seed", "7100", "--base",
                                  str(self.base), "--memory", "b", *extra])

    def test_wiring_and_overrides(self):
        self.assertEqual(self.main("--max-steps", "3"), 0)
        self.assertEqual(self.registered, {"clevr_rowid": vlm_env_logged.clevr_rowid_processor})
        fqn = self.env_registry["vlm"]["actor_class_fqn"]
        mod, attr = fqn.rsplit(".", 1)
        self.assertEqual(mod, "vlm_env_logged")
        self.assertTrue(hasattr(vlm_env_logged, attr))
        self.assertEqual(self.actor_registry, {fqn: "system-python"})
        self.assertIn(str(RUNTIME), os.environ["PYTHONPATH"].split(":"))
        [argv] = self.argv
        self.assertEqual(argv[:3], ["/opt/nemo-rl/examples/run_vlm_grpo.py", "--config",
                                    "/opt/nemo-rl/examples/configs/vlm_grpo_3B.yaml"])
        ov = argv[3:]
        for want in ("data.default.processor=clevr_rowid", f"++data.train.data_path={self.run.train}",
                     f"data.default.prompt_file={run_vlm_nemo.PROMPT}", f"++env.clevr-cogent.log_path={self.out}/env_calls.jsonl",
                     f"policy.max_total_sequence_length={run_vlm_nemo.MAX_SEQ}", "grpo.max_num_steps=3",
                     f"policy.generation.max_new_tokens={gen_vqa.SAMPLER['max_new_tokens']}", "grpo.seed=7100",
                     "policy.logprob_chunk_size=512", "data.shuffle=false", "data_plane.enabled=false"):
            self.assertIn(want, ov)
        self.assertEqual(len(ov), len(set(ov)))
        args = json.loads((self.out / "run_args.json").read_text())
        self.assertEqual((args["max_steps"], args["train_images_checked"], args["memory_rung"]), (3, 4, "b"))
        self.assertEqual(args["nemo_rl_overrides"], ov)
        self.assertEqual(args["env_sha256"], V.file_sha256(RUNTIME / "vlm_env_logged.py"))
        events = [json.loads(x)["event"] for x in (self.out / "attempts.jsonl").read_text().splitlines()]
        self.assertEqual(events, ["start", "end"])

    def test_end_is_recorded_when_the_trainer_fails(self):
        self.trainer_fails = True
        with self.assertRaisesRegex(RuntimeError, "trainer failed"):
            self.main()
        events = [json.loads(x)["event"] for x in (self.out / "attempts.jsonl").read_text().splitlines()]
        self.assertEqual(events, ["start", "end"])

    def test_bad_image_stops_before_the_trainer(self):
        Path(self.run.rows[0]["messages"][0]["content"][0]["image"]).write_bytes(tiny_png(251))
        with self.assertRaises(SystemExit):
            self.main()
        self.assertEqual(self.argv, [])


class TestPromptAndLength(unittest.TestCase):
    """The longest item prompt, its image and the longest answer fit the trainer's sequence length, and generation
    uses the same length."""

    IMAGE = (480, 320)          # CLEVR renders every scene at 480 x 320 pixels
    TEMPLATE_TOKENS = 32        # chat template, vision markers and the generation prompt (generous)

    def image_tokens(self) -> int:
        cfg = json.loads((V.PROBE / "cuts" / "checkpoint-5" / "preprocessor_config.json").read_text())
        factor = cfg["patch_size"] * cfg["merge_size"]
        w, h = (max(factor, round(x / factor) * factor) for x in self.IMAGE)
        self.assertTrue(cfg["size"]["shortest_edge"] <= w * h <= cfg["size"]["longest_edge"])   # no rescale
        return (w // cfg["patch_size"]) * (h // cfg["patch_size"]) // cfg["merge_size"] ** 2

    @staticmethod
    def text_tokens(text: str) -> int:
        tok = RUNTIME / "models" / "qwen3.5-0.8b" / "tokenizer.json"
        try:
            from tokenizers import Tokenizer
        except ImportError:
            Tokenizer = None
        if Tokenizer is not None and tok.is_file():
            return len(Tokenizer.from_file(str(tok)).encode(text).ids)
        # No tokenizer here: at most one token per two characters bounds English BPE text (Qwen averages about 4).
        return (len(text) + 1) // 2

    def questions(self) -> list[str]:
        qs = [json.loads(x)["question"] for s in ("items-valA", "items-valB")
              for x in (RUNTIME / s / "items.jsonl").read_text(encoding="utf-8").splitlines()]
        return qs + [r["messages"][0]["content"][1]["text"] for r in probe_rows(500)]

    def test_image_tokens(self):
        self.assertEqual(self.image_tokens(), 150)

    def test_longest_prompt_image_and_answer_fit(self):
        q = max(self.questions(), key=len)
        need = (self.text_tokens(gen_vqa.user_text(q)) + self.image_tokens() + self.TEMPLATE_TOKENS
                + gen_vqa.SAMPLER["max_new_tokens"])
        self.assertLessEqual(need, run_vlm_nemo.MAX_SEQ)
        self.assertEqual(gen_vqa.SAMPLER["max_model_len"], run_vlm_nemo.MAX_SEQ)

    def test_trainer_is_handed_these_lengths(self):
        ov = run_vlm_nemo.overrides(run_vlm_nemo.PROFILES["full"], types.SimpleNamespace(
            base=Path("/m"), seed=1, memory="a"), Path("/out"))
        self.assertIn(f"policy.max_total_sequence_length={run_vlm_nemo.MAX_SEQ}", ov)
        self.assertIn(f"policy.generation.max_new_tokens={gen_vqa.SAMPLER['max_new_tokens']}", ov)

    def test_training_and_evaluation_show_the_same_text(self):
        # NeMo RL formats the training text part with data.default.prompt_file; gen_vqa.py formats the same file.
        self.assertEqual(run_vlm_nemo.PROMPT, gen_vqa.PROMPT_FILE)
        q = self.questions()[0]
        self.assertEqual(gen_vqa.user_text(q), run_vlm_nemo.PROMPT.read_text(encoding="utf-8").format(q))
        self.assertEqual(V.file_sha256(gen_vqa.PROMPT_FILE), V.PROMPT_BLOCK_SHA256)


class TestAnswerParserParity(unittest.TestCase):
    """The adapter's copy of score_vqa.py's parser and the tests' fixture scorer agree with score_vqa.py."""

    CASES = ("<answer> 3 </answer>", "so <answer>Three</answer>", "<answer>yes</answer>", "I see 2 cubes, so 4.",
             "<think>2 or 5</think> <answer>05</answer>", "<answer>1</answer> ... <answer>no</answer>", "nothing",
             "<answer></answer> 7", "<answer>Twenty</answer>", "<ANSWER>eleven</ANSWER>", "<think>yes</think>",
             "<think>a</think><think>b</think> no", "<answer>\n12\n</answer>", "nineteen or twenty-one?", "")
    REFS = ("0", "10", "yes", "No", "blue", "2 3", "twelve", " 7 ", "07", "")

    def test_adapter_parser(self):
        for text in self.CASES:
            self.assertEqual(A.parse_answer(text), score_vqa.parse_answer(text), text)
        for ref in self.REFS:
            self.assertEqual(A.canon_reference(ref), score_vqa.canon_reference(ref), ref)

    def test_fixture_scorer_is_score_vqa(self):
        for text in self.CASES:
            want = score_vqa.parse_answer(text)
            self.assertEqual(V.parse(text), (want["answer"], want["answer_tagged"]), text)

    def test_self_test(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(score_vqa.self_test(), 0)


if __name__ == "__main__":
    unittest.main()
