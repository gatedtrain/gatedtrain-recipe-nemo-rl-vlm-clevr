"""The `nemo_rl_vlm` adapter, area "VLM" (NeMo RL VLM GRPO of Qwen3.5-0.8B on CLEVR-CoGenT; gatedtrain tests/README.md
section 11.24, EXT-200 to EXT-213), moved from gatedtrain's tests with the recipe.

Sources: the multimodal spec, MM-1 to MM-39 (section 6 rows 7 and 7s and the row 7 deviations, MM-11, MM-17, MM-20,
MM-21, MM-23, MM-27, MM-29, MM-31 to MM-36, MM-38), design revision 16: 05 section 6.1 (R15-1, R16-1), sections 12.1 to
12.3 (a `row` record); the area 7 decisions (2026-11-14) in runtime/README.md; the area runtime
`runtime/` (run_vlm_nemo.py, ptc_vlm_hooks.py, vlm_env_logged.py, gen_vqa.py,
score_vqa.py, make_data.py; read 2026-11-14), its GPU probe outputs `out/area7-vlm/probe-a6/train/` and
`out/area7-vlm/gen-base-probe1/` (copied and cut down into `fixtures/vlm/`), and NeMo RL 4f39fd8
(vlm_environment.py:146-231, rewards.py:52-85, grpo.py:4690). Helpers: `tests/_nemo_rl_vlm.py`.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import _nemo_rl_vlm as V
from gatedtrain.testing.ext import sha_of, write_canonical
from gatedtrain.testing.support import EXIT_OK, EXIT_REFUSED, describe, file_sha256, manifest_hash, parse_errors, parse_run_dir, read_jsonl, run_cli

NOT_PASS = 1
REAL = Path(V.AREA7)
# The real arm fixture ran the scorer of the 2026-11-14 gate, before the renames to ptc-hub 0.2.0 and gatedtrain 0.3.0; the adapter pins
# it with the name map applied (results/2026-11-14/gate/registration-gatedtrain.yaml rename_proof), so its arms no
# longer collect.
GATE_SCORER_REFUSAL = (f"env-runtime.json code_sha256 {{'score_vqa.py': '{V.SCORE_VQA_GATE_SHA256}'}} is not the "
                       f"pinned score_vqa.py {V.SCORE_VQA_SHA256}")


def registry():
    from gatedtrain.ext import registry as reg
    return reg


def adapter():
    return registry().adapter(V.ADAPTER)


def load_runtime_module(name: str) -> dict:
    """The runtime's module namespace, compiled from its source (no byte code is written into the area directory)."""
    path = REAL / name
    ns = {"__file__": str(path), "__name__": f"{name[:-3]}_real"}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
    return ns


def _root_can_read() -> bool:
    return os.geteuid() == 0


class VlmCase(unittest.TestCase):
    """One scenario per test that changes it (`fresh`); `self.sc` is shared by the read-only tests of a class."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-vlm-"))
        cls.addClassCleanup(shutil.rmtree, cls.tmp, True)
        cls.n = 0
        cls.sc = V.VlmScenario(cls.tmp / "shared")

    def fresh(self, **kw) -> V.VlmScenario:
        type(self).n += 1
        return V.VlmScenario(self.tmp / f"sc-{type(self).n}", **kw)

    def runs(self) -> Path:
        type(self).n += 1
        return self.tmp / f"runs-{type(self).n}"

    def probe(self) -> Path:
        type(self).n += 1
        return V.probe_copy(self.tmp / f"probe-{type(self).n}")

    # --- in-process checks ---

    def cuts(self, sc: V.VlmScenario) -> list:
        return adapter().find_cuts(sc.out)[0]

    def log_problems(self, sc: V.VlmScenario) -> list[str]:
        return adapter().consumption_problems(sc.out, sc.data, self.cuts(sc))

    def probe_problems(self, p: Path) -> list[str]:
        return adapter().consumption_problems(p, p / "data" / "shards", [(5, p / "cuts" / "checkpoint-5")])

    def check_problems(self, sc: V.VlmScenario, launch: dict | None = None) -> list:
        cuts = self.cuts(sc)
        last = cuts[-1][1] if cuts else sc.out / "cuts" / "checkpoint-60"
        rec = launch if launch is not None else json.loads((sc.root / "launch.json").read_text())
        return adapter().checks(last_cut=last, base=sc.base, launch=rec, base_field="base", stage_out=sc.out,
                                data=sc.data, data_field="data")

    def assertProblem(self, msgs, text: str):
        flat = [m if isinstance(m, str) else f"{m.rule} {m.field}: {m.message}" for m in msgs]
        self.assertTrue(any(text in m for m in flat), f"expected a problem containing {text!r}; got {flat}")

    # --- CLI ---

    def passes(self, sc: V.VlmScenario, **kw) -> Path:
        proc = sc.collect(self.runs(), **kw)
        self.assertIn(proc.returncode, (EXIT_OK, NOT_PASS), describe(proc))
        return parse_run_dir(proc.stdout)

    def refused(self, sc: V.VlmScenario, rule: str, text: str | None = None, **kw):
        runs = self.runs()
        proc = sc.collect(runs, **kw)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertNotIn("Traceback", proc.stderr)
        errs = parse_errors(proc.stderr)
        hits = [e for e in errs if e["rule"] == rule and (text is None or text in e["message"])]
        self.assertTrue(hits, f"expected error[{rule}] {text!r}; got {errs}\n{describe(proc)}")
        self.assertFalse(runs.exists() and any(runs.iterdir()), "a refusal writes no run directory")
        return hits

    def unreadable(self, p: Path) -> None:
        if _root_can_read():
            self.skipTest("running as root: chmod 0 does not make a file unreadable")
        p.chmod(0)
        self.addCleanup(p.chmod, 0o644)


# ---------------------------------------------------------------------------
# EXT-200: registry rows and docs
# ---------------------------------------------------------------------------

class TestRegistryRows(unittest.TestCase):

    def test_trainer_row(self):
        row = registry().TRAINER_ROWS[V.ADAPTER]
        self.assertEqual(list(row["uses"]), [V.TRAINER_ID])
        self.assertEqual(list(row["evaluators"]), [V.EVALUATOR_ID, V.VALB_ID])        # deciding first (MM-38)
        self.assertEqual(row["consumption"], {"target": "row", "fallback": None})
        self.assertEqual((row["framework"], row["nemo_rl_commit"], tuple(row["pins"])), ("nemo_rl", V.NEMO_RL_COMMIT, ()))
        self.assertFalse(row["spec_binding"])
        self.assertEqual(sorted(row["env_names"]), V.ENV_NAMES)
        self.assertNotIn("HF_TOKEN", row["env_names"])
        self.assertNotIn("sidecar", row)
        self.assertNotIn("b18", row)
        self.assertEqual(row["image_interpreters"], {"run_vlm_nemo.py": V.TRAIN_PY, "gen_vqa.py": V.GEN_PY,
                                                     "score_vqa.py": V.TRAIN_PY})
        self.assertEqual(row["interpreter_flags"], {"score_vqa.py": "-I"})
        self.assertEqual(row["interpreter"], V.TRAIN_PY)
        self.assertEqual(tuple(row["interpreter_trees"]), ())
        self.assertEqual(tuple(row["input_roots"]), (V.AREA7, "{output_root}/" + V.OUT_NAME))
        self.assertEqual(registry().run_options(V.ADAPTER, "run_vlm_nemo.py")[0], importlib.import_module("gatedtrain.families.nemo_rl").NEMO_RL_RUN_OPTIONS)
        self.assertEqual(registry().run_options(V.ADAPTER, "score_vqa.py"), ((), None))
        self.assertEqual(registry().trainer_ids()[V.TRAINER_ID], V.ADAPTER)
        self.assertIn(V.IMAGE, importlib.import_module("gatedtrain.families.nemo_rl").NEMO_RL_IMAGES)
        self.assertEqual(registry().ADAPTERS[V.ADAPTER], "gatedtrain_recipe_nemo_rl_vlm_clevr.adapter:NemoRlVlmAdapter")

    def test_evaluator_rows(self):
        rows = registry().EVALUATOR_ROWS
        for eid in (V.EVALUATOR_ID, V.VALB_ID):
            with self.subTest(evaluator=eid):
                row = rows[eid]
                self.assertEqual((row["metric"], row["direction"], tuple(row["score_range"]), row["tier"], row["binary"]),
                                 ("exact_match", "higher_is_better", (0, 1), "T3", True))
                self.assertEqual(tuple(row["error_classes"]), ("missing_output", "scorer_error"))
                self.assertEqual(tuple(row["score_outcomes"]), ("wrong_answer", "no_answer"))
                self.assertEqual(row["payload_text_field"], "text")
                self.assertEqual(tuple(row["item_payload"]), ("text", "question", "answer", "image_sha256"))
                self.assertEqual(set(row["sampler_keys"]), set(V.SAMPLER))
                self.assertEqual(row["scorer"], V.SCORER)
                self.assertEqual(row["samples"], "500, 1")
                self.assertEqual(row["construct_aligned"]["reward"], "nemo_rl_vlm_format_exact_alnum")
                self.assertIn("another answer parser", row["construct_aligned"]["statement"])
                self.assertIn("CLEVR", row["shared_annotation"]["convention"])
        self.assertEqual(registry().disclosure_problems(rows), [])

    def test_identity_hash(self):
        for eid in (V.EVALUATOR_ID, V.VALB_ID):
            self.assertEqual(registry().evaluator_identity_hash(eid),
                             sha_of({"evaluator": eid, "metric": "exact_match", "direction": "higher_is_better",
                                     "scorer": V.SCORER}))

    def test_items_table_agrees(self):
        from gatedtrain.items import EVALUATORS
        for eid in (V.EVALUATOR_ID, V.VALB_ID):
            self.assertEqual(EVALUATORS[eid]["direction"], "higher_is_better")
            self.assertEqual(tuple(EVALUATORS[eid]["score_range"]), (0, 1))

    def test_adapter_imports_no_framework(self):
        """MM-12: the module imports no framework, torch or pickle loader."""
        src = (Path(__file__).resolve().parents[1] / "src" / "gatedtrain_recipe_nemo_rl_vlm_clevr" / "adapter.py").read_text()
        for mod in ("torch", "nemo_rl.", "vllm", "transformers", "pickle", "safetensors"):
            self.assertNotIn(f"import {mod}", src)


class TestMakeDataMeta(unittest.TestCase):

    def test_host_specific_train_hash_keeps_the_committed_meta(self):
        """runtime/make_data.py keeps the committed data_meta.json when only the host-specific train_file_sha256
        differs; any other difference does not."""
        import importlib.util
        rt = Path(__file__).resolve().parents[1] / "runtime"          # this checkout's committed files
        spec = importlib.util.spec_from_file_location("vlm_make_data", rt / "make_data.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        committed = (rt / "data" / "data_meta.json").read_bytes()
        meta = json.loads(committed)
        recorded = meta["train_file_sha256"]
        here = {**meta, "train_file_sha256": "sha256:" + "5" * 64}
        self.assertEqual(mod.recorded_train_sha256(here, committed), recorded)
        self.assertEqual(mod.render({**here, "train_file_sha256": recorded}), committed)
        self.assertIsNone(mod.recorded_train_sha256({**here, "seed": meta["seed"] + 1}, committed))
        self.assertIsNone(mod.recorded_train_sha256(here, None))


# ---------------------------------------------------------------------------
# EXT-213: runtime facts (the real area directory and probe outputs; skipped where absent) and the probe fixture
# ---------------------------------------------------------------------------

class TestRuntimeFacts(unittest.TestCase):
    """The pinned values against the real runtime (read only)."""

    def setUp(self):
        if not (REAL / "score_vqa.py").is_file():
            self.skipTest(f"area runtime absent: {REAL}")

    def need(self, *rels: str) -> None:
        """Skip unless the runtime holds these regenerated files (fetch_models.py, make_data.py; not in git)."""
        missing = [r for r in rels if not (REAL / r).exists()]
        if missing:
            self.skipTest(f"runtime/{missing[0]} absent (run fetch_models.py and make_data.py)")

    def test_scorer_code_is_the_runtime_composite(self):
        self.assertEqual(file_sha256(REAL / "score_vqa.py"), V.SCORE_VQA_SHA256)
        self.assertEqual(V.compact_sha({"score_vqa.py": V.SCORE_VQA_SHA256}), V.SCORER_CODE)
        self.assertEqual(registry().EVALUATOR_ROWS[V.EVALUATOR_ID]["scorer"]["code_sha256"], V.SCORER_CODE)
        ns = load_runtime_module("score_vqa.py")
        self.assertEqual((ns["SCORER_VERSION"], ns["ADAPTER"], ns["EVALUATOR_ID"]),
                         ("clevr_exact_v1", V.ADAPTER, V.EVALUATOR_ID))

    def test_prompt_file_and_sampler(self):
        self.assertEqual(file_sha256(REAL / "prompt_clevr_cot.txt"), V.PROMPT_BLOCK_SHA256)
        self.assertEqual(file_sha256(V.PROMPT_FILE), V.PROMPT_BLOCK_SHA256)
        ns = load_runtime_module("gen_vqa.py")
        self.assertEqual(ns["SAMPLER"], V.CUT_SAMPLER)
        from gatedtrain_recipe_nemo_rl_vlm_clevr import adapter as M
        self.assertEqual(tuple(ns["DEPLOYABLE_NAMES"]), M.GEN_MODEL_FILES)

    def test_runtime_constants(self):
        """run_vlm_nemo.py PROFILES and MEMORY, ptc_vlm_hooks.py CUT_REQUIRED and NOT_DEPLOYABLE are the adapter's."""
        from gatedtrain_recipe_nemo_rl_vlm_clevr import adapter as M
        text = (REAL / "run_vlm_nemo.py").read_text(encoding="utf-8")
        for prof, d in M.PROFILES.items():
            self.assertIn(f'"{prof}": {{"steps": {d["steps"]}, "prompts": 16, "generations": 8, "save_period": '
                          f'{d["save_period"]}}}', text)
        for rung in M.MEMORY_RUNGS:
            self.assertIn(f'"{rung}": [', text)
        for ov in M.REQUIRED_OVERRIDES:
            self.assertIn(f'"{ov}"', text)
        hooks = (REAL / "ptc_vlm_hooks.py").read_text(encoding="utf-8")
        self.assertIn('CUT_REQUIRED = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",\n'
                      '                "preprocessor_config.json", "sampler.json")', hooks)
        self.assertIn('NOT_DEPLOYABLE = ("base_ref.json", "training_info.json", "cut.json")', hooks)

    def test_answer_parser_is_the_runtime_s(self):
        """The adapter's copy of score_vqa.py parse_answer and canon_reference agrees with the runtime's (its self-test
        cases, odd texts and the real responses of the base probe)."""
        ns = load_runtime_module("score_vqa.py")
        from gatedtrain_recipe_nemo_rl_vlm_clevr.adapter import canon_reference, parse_answer
        texts = ["<answer> 3 </answer>", "so <answer>Three</answer>", "<answer>yes</answer>", "I count 2, so 4.",
                 "<think>2 or 5</think> <answer>05</answer>", "<answer>1</answer> ... <answer>no</answer>",
                 "nothing here", "there are none of them", "<answer></answer> 7", "</think></think>twenty",
                 "<ANSWER>No</ANSWER>", "<answer>1\n2</answer>", "10 cubes and Eleven spheres", ""]
        for p in sorted((V.REAL_ARM / "outputs").glob("*.json")):
            texts.append(json.loads(p.read_text())["response"])
        for t in texts:
            self.assertEqual(parse_answer(t), ns["parse_answer"](t), t)
        for ref in ("0", "10", "yes", "No", "blue", "2 3", "07", " 4 ", "four", "1.5"):
            self.assertEqual(canon_reference(ref), ns["canon_reference"](ref), ref)

    def test_real_base_is_a_model_directory(self):
        self.need(V.BASE_REL)
        base = REAL / V.BASE_REL
        self.assertIsNone(adapter().model_dir_problem(base))
        models = json.loads((REAL / "models" / "MODELS.json").read_text(encoding="utf-8"))["models"][base.name]
        self.assertEqual((models["repo"], models["revision"]), (V.BASE_REPO, V.BASE_REVISION))
        self.assertEqual(sorted(models["files"]), sorted(V.BASE_FILES))
        self.assertFalse((base / "generation_config.json").exists())

    def test_real_items_and_overlap(self):
        """EXT-206: both real item sets read with the payload; every key is well formed; no item is a training pair,
        though questions repeat across images."""
        from gatedtrain.ext.collect import read_item_set
        self.need("data/shards")
        row = registry().EVALUATOR_ROWS[V.EVALUATOR_ID]
        data = REAL / "data" / "shards"
        payloads, why = adapter().training_payloads(Path("."), None, [], data=data)
        self.assertIsNone(why)
        self.assertEqual(len(payloads), 2000)
        train_q = {p.split("\n", 1)[1] for p in payloads}
        for name in ("items-valA", "items-valB"):
            with self.subTest(items=name):
                rows, bad = read_item_set(REAL / name, tuple(row["item_payload"]))
                self.assertEqual((len(rows), bad), (500, []))
                self.assertEqual(adapter().item_overlap_problems(rows, data), [])
                self.assertEqual(sum(r["text"] in payloads for r in rows), 0)
                self.assertTrue(any(r["question"] in train_q for r in rows))   # questions repeat on other images

    def test_real_probe_consumption(self):
        """EXT-204: the GPU smoke's logs join (640 of 640) with the real data; the cut is not re-hashed here (1.7 GB)."""
        real_out = V.real_outputs()
        probe = real_out / "probe-a6" / "train" if real_out else None
        if probe is None or not (probe / "env_calls.jsonl").is_file():
            self.skipTest("probe-a6 absent")
        self.need("data/shards", "data/train.jsonl")
        cuts = [(5, probe / "cuts" / "checkpoint-5")]
        self.assertEqual(adapter().consumption_problems(probe, REAL / "data" / "shards", cuts), [])
        fw = {"name": "nemo_rl", "version": V.NEMO_RL_COMMIT, "commit": V.NEMO_RL_COMMIT}
        self.assertEqual(adapter().framework_identity(probe, {"framework": fw})["version"], V.NEMO_RL_COMMIT)
        args = json.loads((probe / "run_args.json").read_text())
        self.assertEqual(args["train_file_sha256"], file_sha256(REAL / "data" / "train.jsonl"))
        self.assertEqual(args["python"], "3.13.14")      # the image's interpreter: the pinned scorer version's Python

    def test_real_arm_with_the_real_item_set(self):
        from gatedtrain.ext.arm import read_lines
        arm = V.REAL_ARM / "arm-exact_match"
        recs, bad = read_lines(arm, seeds=[11], score_range=(0, 1), error_classes=("missing_output", "scorer_error"))
        self.assertEqual(bad, [])
        items = read_jsonl(REAL / "items-valA" / "items.jsonl")
        self.assertEqual(adapter().arm_problems(arm, recs, items, items_dir=REAL / "items-valA"), [GATE_SCORER_REFUSAL])
        self.assertTrue(adapter().arm_problems(arm, recs, items, items_dir=REAL / "items-valB"))

    def test_credential_scan_of_the_real_mounts(self):
        from gatedtrain.ext.launch import credential_name_problem
        row = registry().TRAINER_ROWS[V.ADAPTER]
        self.assertIsNone(credential_name_problem(V.AREA7, skip=tuple(row["scan_skip"]),
                                                  interpreter_trees=tuple(row["interpreter_trees"])))


class TestProbeFixture(VlmCase):
    """The GPU smoke's outputs (fixtures/ext/vlm/probe-a6/, weights replaced) against the adapter: always run."""

    def test_cut_complete_and_identity(self):
        p = V.PROBE
        self.assertEqual([(s, d.name) for s, d in adapter().find_cuts(p)[0]], [(5, "checkpoint-5")])
        ident = adapter().identity(p / "cuts" / "checkpoint-5")
        self.assertEqual(sorted(ident["components"]), [
            "chat_template.jinja", "config.json", V.CUT_WEIGHTS, V.INDEX, "preprocessor_config.json", "sampler.json",
            "tokenizer.json", "tokenizer_config.json", "video_preprocessor_config.json"])
        self.assertEqual(ident["model_hash"], V.deploy_hash(p / "cuts" / "checkpoint-5"))
        self.assertIsNone(ident["base_ref"])

    def test_probe_logs_join(self):
        p = V.PROBE
        self.assertEqual(self.probe_problems(p), [])
        cons = adapter().consumption(p, p / "data" / "shards", [(5, p / "cuts" / "checkpoint-5")], run_id="r")
        self.assertEqual((cons["precision"], cons["derivation"], len(cons["rows"])), ("row", "observed", 640))
        self.assertEqual(cons["rows"][0]["data_ref"]["row"], V.sha_text("clevr-trainA-00000"))
        self.assertEqual(cons["rows"][-1]["data_ref"]["row"], V.sha_text("clevr-trainA-00079"))
        self.assertEqual(len({r["sample_id"] for r in cons["rows"]}), 640)

    def test_probe_sampler_and_freeze(self):
        p = V.PROBE
        self.assertEqual(json.loads((p / "cuts" / "checkpoint-5" / "sampler.json").read_text()), V.CUT_SAMPLER)
        self.assertEqual(adapter().sampler_problems([p / "cuts" / "checkpoint-5"], V.SAMPLER), [])
        fw = {"name": "nemo_rl", "version": V.NEMO_RL_COMMIT, "commit": V.NEMO_RL_COMMIT}
        self.assertEqual(adapter().framework_identity(p, {"framework": fw})["name"], "nemo_rl")

    def test_probe_reward_mismatch(self):
        p = self.probe()
        rows = read_jsonl(p / "env_calls.jsonl")
        i = next(k for k, x in enumerate(rows) if x["reward"] > 0)
        rows[i]["reward"] = 0.0
        V.write_jsonl(p / "env_calls.jsonl", rows)
        self.assertProblem(self.probe_problems(p), "do not join in order")

    def test_probe_missing_step(self):
        p = self.probe()
        (p / "nemo" / "logs" / "exp_001" / "train_data_step3.jsonl").unlink()
        self.assertProblem(self.probe_problems(p), "train_data_step3.jsonl is missing")

    def test_probe_row_order(self):
        """Two rows of step 1 swapped (their 8 samples each, rewards kept in place): not the file order."""
        p = self.probe()
        rows = read_jsonl(p / "env_calls.jsonl")
        for k in range(8):
            rows[k]["row_id"], rows[8 + k]["row_id"] = rows[8 + k]["row_id"], rows[k]["row_id"]
        V.write_jsonl(p / "env_calls.jsonl", rows)
        self.assertProblem(self.probe_problems(p), "in file order")

    def test_probe_rows_interleaved(self):
        """The step's rows in file order but not 8 times in a row (NeMo RL repeats each prompt consecutively)."""
        p = self.probe()
        rows = read_jsonl(p / "env_calls.jsonl")
        rows[7]["row_id"], rows[8]["row_id"] = rows[8]["row_id"], rows[7]["row_id"]
        V.write_jsonl(p / "env_calls.jsonl", rows)
        self.assertProblem(self.probe_problems(p), "8 times in a row")

    def test_probe_cut_rehash_needed(self):
        p = self.probe()
        (p / "cuts" / "checkpoint-5" / "tokenizer.json").write_text("other\n")
        self.assertIn("tokenizer.json", adapter().cut_problem(p / "cuts" / "checkpoint-5", 5))


# ---------------------------------------------------------------------------
# EXT-201, EXT-202: cuts, completeness, identity
# ---------------------------------------------------------------------------

class TestCuts(VlmCase):

    def test_cuts_under_cuts_dir(self):
        self.assertEqual([(s, d.name) for s, d in self.cuts(self.sc)],
                         [(20, "checkpoint-20"), (40, "checkpoint-40"), (60, "checkpoint-60")])
        dirs = [d for _, d in adapter().checkpoint_dirs(self.sc.out)]
        self.assertTrue(all(d.parent.name == "cuts" for d in dirs), dirs)
        self.assertTrue((self.sc.out / "nemo" / "checkpoints" / "step_60").is_dir())

    def test_partial_and_incomplete(self):
        sc = self.fresh()
        cuts = sc.out / "cuts"
        shutil.copytree(cuts / "checkpoint-20", cuts / ".checkpoint-80.partial")
        self.assertEqual([s for s, _ in adapter().find_cuts(sc.out)[0]], [20, 40, 60])

        def broken(step: int, fn) -> str:
            d = cuts / f"checkpoint-{step}"
            shutil.copytree(cuts / "checkpoint-60", d)
            sc.edit_json(d / "cut.json", lambda c: c.update(global_step=step, nemo_rl_step_dir=f"step_{step}"))
            fn(d)
            return adapter().cut_problem(d, step)

        self.assertIn("cut.json", broken(101, lambda d: (d / "cut.json").unlink()))
        self.assertIn("does not list", broken(102, lambda d: (d / "optimizer.pt").write_text("x")))
        self.assertIn("SHA-256", broken(103, lambda d: (d / "config.json").write_text("{}")))

        def drop(name):
            def fn(d):
                (d / name).unlink()
                sc.edit_json(d / "cut.json", lambda c: (c["files"].pop(name), c.update(
                    deployable=[n for n in c["deployable"] if n != name])))
            return fn
        self.assertIn("training_info.json", broken(104, drop("training_info.json")))
        self.assertIn("preprocessor_config.json", broken(105, drop("preprocessor_config.json")))
        self.assertIn("sampler.json", broken(106, drop("sampler.json")))
        self.assertIn("deployable", broken(107, lambda d: sc.edit_json(
            d / "cut.json", lambda c: c.update(deployable=c["deployable"] + ["base_ref.json"]))))
        self.assertIn("nemo_rl_step_dir", broken(108, lambda d: sc.edit_json(
            d / "cut.json", lambda c: c.update(nemo_rl_step_dir="step_60"))))
        self.assertIn("global_step", broken(109, lambda d: sc.edit_json(d / "cut.json",
                                                                        lambda c: c.update(global_step=True))))
        self.assertIn("stage", broken(110, lambda d: sc.edit_json(d / "cut.json", lambda c: c.update(stage="grpo"))))
        self.assertIn("schema", broken(111, lambda d: sc.edit_json(d / "cut.json",
                                                                   lambda c: c.update(schema="ptc_mm.cad_cut"))))
        self.assertIn("examples_consumed", broken(112, lambda d: sc.edit_json(
            d / "cut.json", lambda c: c.update(examples_consumed=False))))
        self.assertIn("global_step", broken(113, lambda d: sc.edit_json(d / "cut.json",
                                                                        lambda c: c.update(global_step=60,
                                                                                           nemo_rl_step_dir="step_60"))))

    def test_weights_rules(self):
        """The weights: whole, named by the index; several files need the index; plain names only."""
        sc = self.fresh()
        cut = sc.out / "cuts" / "checkpoint-60"
        p = cut / V.CUT_WEIGHTS
        data = p.read_bytes()
        p.unlink()                                       # break the hard link to NeMo RL's step directory
        p.write_bytes(data[:-3])
        sc.reseal(cut)
        self.assertIn(V.CUT_WEIGHTS, adapter().cut_problem(cut, 60))
        p.unlink()
        p.write_bytes(data)
        sc.edit_json(cut / V.INDEX, lambda i: i["weight_map"].update(w="model-00001-of-00002.safetensors"))
        sc.reseal(cut)
        self.assertIn("weight_map", adapter().cut_problem(cut, 60))
        # two weights files named by the index: complete; without the index: incomplete
        sc.edit_json(cut / V.INDEX, lambda i: i["weight_map"].update(w=V.CUT_WEIGHTS, v="model-00002-of-00002.safetensors"))
        (cut / "model-00002-of-00002.safetensors").write_bytes(data)

        def relist(c):
            c["files"]["model-00002-of-00002.safetensors"] = ""
            c["deployable"] = sorted(c["deployable"] + ["model-00002-of-00002.safetensors"])
        sc.edit_json(cut / "cut.json", relist)
        sc.reseal(cut)
        self.assertIsNone(adapter().cut_problem(cut, 60))
        (cut / V.INDEX).unlink()
        sc.edit_json(cut / "cut.json", lambda c: (c["files"].pop(V.INDEX), c["deployable"].remove(V.INDEX)))
        self.assertIn("no model.safetensors.index.json", adapter().cut_problem(cut, 60))

    def test_single_weights_file_without_index(self):
        sc = self.fresh()
        cut = sc.out / "cuts" / "checkpoint-60"
        (cut / V.INDEX).unlink()
        sc.edit_json(cut / "cut.json", lambda c: (c["files"].pop(V.INDEX), c["deployable"].remove(V.INDEX)))
        self.assertIsNone(adapter().cut_problem(cut, 60))

    def test_base_directory_is_a_model_directory(self):
        self.assertIsNone(adapter().model_dir_problem(self.sc.base))
        self.assertIsNone(adapter().model_dir_problem(self.sc.cut))
        self.assertIn("config.json", adapter().model_dir_problem(self.sc.runtime / "models"))
        sc = self.fresh()
        (sc.base / "preprocessor_config.json").unlink()
        self.assertIn("preprocessor_config.json", adapter().model_dir_problem(sc.base))


class TestIdentity(VlmCase):

    def test_base_alone(self):
        ident = adapter().identity(self.sc.base)
        self.assertEqual(ident["model_hash"], manifest_hash(self.sc.base))
        self.assertEqual(ident["model_hash"], ident["cut_hash"])
        self.assertIsNone(ident["base_ref"])
        self.assertEqual(sorted(ident["components"]), sorted(V.BASE_FILES))

    def test_cut(self):
        ident = adapter().identity(self.sc.cut)
        self.assertEqual(ident["model_hash"], self.sc.cand_model)
        self.assertEqual(ident["cut_hash"], manifest_hash(self.sc.cut))
        for name in ("base_ref.json", "training_info.json", "cut.json"):
            self.assertNotIn(name, ident["components"])
        for name in (V.CUT_WEIGHTS, V.INDEX, "sampler.json", "preprocessor_config.json", "chat_template.jinja"):
            self.assertIn(name, ident["components"])
        self.assertIsNone(ident["base_ref"])
        self.assertNotEqual(ident["model_hash"], adapter().identity(self.sc.out / "cuts" / "checkpoint-40")["model_hash"])


# ---------------------------------------------------------------------------
# EXT-203: collect passes (deciding valA gate, supporting valB arms)
# ---------------------------------------------------------------------------

class TestCollectPass(VlmCase):

    def test_pass(self):
        run = self.passes(self.sc)
        report = json.loads((run / "report" / "gate_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["consumption_precision"], "row")
        self.assertEqual(report["baseline_model_hash"], self.sc.base_model)
        self.assertEqual(report["candidate_model_hash"], self.sc.cand_model)
        self.assertEqual(report["independence_tier"], "T3")
        self.assertEqual(report["construct_alignment"]["reward"], "nemo_rl_vlm_format_exact_alnum")
        self.assertIn("CLEVR", report["shared_annotation"]["convention"])
        self.assertEqual(report["contamination"], {"exact_overlap": 0, "allowed": False})
        import pyarrow.parquet as pq
        rows = pq.read_table(run / "ledger" / "consumption.parquet").to_pylist()
        self.assertEqual(len(rows), 60 * V.PROMPTS * V.GENERATIONS)
        env = read_jsonl(self.sc.out / "env_calls.jsonl")
        self.assertEqual([r["rewards"] for r in rows], [[x["reward"]] for x in env])
        self.assertEqual({r["precision"] for r in rows}, {"row"})
        self.assertEqual({r["derivation"] for r in rows}, {"observed"})
        self.assertEqual(len({r["sample_id"] for r in rows}), len(rows))
        s4 = [r for r in rows if r["step"] == 4]                      # 3 steps per epoch: step 4 back at row 0
        self.assertEqual(s4[0]["data_ref"]["row"], V.sha_text("clevr-trainA-00000"))
        self.assertEqual(s4[0]["data_ref"]["shard"], "shard-00000.jsonl")
        self.assertEqual(s4[-1]["data_ref"]["shard"], "shard-00001.jsonl")   # rows 0 .. 15: shards 0 and 1
        self.assertEqual(s4[0]["data_ref"]["snapshot"], manifest_hash(self.sc.data))
        lock = json.loads((run / "lockfile.json").read_text(encoding="utf-8"))
        self.assertEqual(lock["backend"]["trainer"]["framework"]["commit"], V.NEMO_RL_COMMIT)
        self.assertEqual(lock["backend"]["trainer"]["image_id"], V.IMAGE)
        self.assertEqual(lock["backend"]["evaluator"]["scorer"], V.SCORER)

    def test_valb_supporting_arms_collect(self):
        """items-valB under its own evaluator id (score_vqa.py --evaluator-id): the same run, its own registration."""
        sc = self.fresh(evaluator=V.VALB_ID)
        run = self.passes(sc)
        report = json.loads((run / "report" / "gate_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["candidate_model_hash"], sc.cand_model)

    def test_valb_arm_refused_under_the_valA_gate(self):
        sc = self.fresh()
        self.refused(sc, "EXT-ARM", "evaluator_id", baseline=sc.arms["valB"][0] / "arm-exact_match",
                     candidate=sc.arms["valB"][1] / "arm-exact_match")

    def test_other_trainer_id_refused(self):
        """Another adapter's registered training stage type (the CAD recipe's `rl.nemo_rl_env@0.1`) is refused for
        this adapter. Needs that stage type registered, i.e. gatedtrain-recipe-nemo-rl-cad installed beside this
        recipe; otherwise the spec fails earlier with an unknown stage type (SPEC-REF) and the test skips."""
        uses = {u for row in registry().TRAINER_ROWS.values() for u in (row.get("uses") or ())}
        if "rl.nemo_rl_env@0.1" not in uses:
            self.skipTest("needs the stage type rl.nemo_rl_env@0.1 of gatedtrain-recipe-nemo-rl-cad, not installed")
        sc = self.fresh()
        spec = (sc.root / "spec.yaml").read_text().replace("rl.nemo_rl_vlm@0.1", "rl.nemo_rl_env@0.1")
        (sc.root / "spec.yaml").write_text(spec)
        self.refused(sc, "EXT-SPEC", "rl.nemo_rl_vlm@0.1")


# ---------------------------------------------------------------------------
# EXT-204: observed consumption (env_calls.jsonl joined with train_data_step<N>.jsonl)
# ---------------------------------------------------------------------------

class TestConsumptionLog(VlmCase):

    def env(self, fn) -> V.VlmScenario:
        sc = self.fresh()
        sc.edit_jsonl(sc.out / "env_calls.jsonl", fn)
        return sc

    def step_file(self, sc: V.VlmScenario, n: int) -> Path:
        return sc.out / "nemo" / "logs" / "exp_001" / f"train_data_step{n}.jsonl"

    def test_clean(self):
        self.assertEqual(self.log_problems(self.sc), [])

    def test_reward_mismatch(self):
        sc = self.env(lambda x: x[10].update(reward=x[10]["reward"] + 0.01 if x[10]["reward"] < 0.9 else 0.0))
        self.assertProblem(self.log_problems(sc), "train_data_step1.jsonl")

    def test_two_samples_swapped(self):
        def swap(x):
            i = next(k for k in range(1, 128) if x[k]["reward"] != x[0]["reward"])
            x[0], x[i] = dict(x[i], position=0), dict(x[0], position=i)
        sc = self.env(swap)
        self.assertProblem(self.log_problems(sc), "reward")

    def test_extra_and_missing_lines(self):
        self.assertProblem(self.log_problems(self.env(lambda x: x.append(dict(x[-1], position=128)))), "lines")
        self.assertProblem(self.log_problems(self.env(lambda x: x.pop())), "lines")

    def test_row_of_another_step(self):
        sc = self.env(lambda x: [l.update(row_id="clevr-trainA-00020") for l in x[0:8]])
        self.assertProblem(self.log_problems(sc), "file order")

    def test_unknown_row(self):
        sc = self.env(lambda x: x[0].update(row_id="clevr-trainA-99999"))
        self.assertProblem(self.log_problems(sc), "not a row")

    def test_reward_out_of_range_or_boolean(self):
        self.assertProblem(self.log_problems(self.env(lambda x: x[3].update(reward=1.5))), "outside [0, 1]")
        self.assertProblem(self.log_problems(self.env(lambda x: x[3].update(reward=-0.2))), "outside [0, 1]")
        self.assertProblem(self.log_problems(self.env(lambda x: x[3].update(reward=True))), "reward")

    def test_response_hash(self):
        self.assertProblem(self.log_problems(self.env(lambda x: x[3].update(response_sha256="abc"))), "response_sha256")

    def test_calls_and_positions(self):
        self.assertProblem(self.log_problems(self.env(lambda x: [l.update(call=3) for l in x[128:256]])), "call")
        self.assertProblem(self.log_problems(self.env(lambda x: x[5].update(position=50))), "position")

    def test_step_log_missing_or_beyond(self):
        sc = self.fresh()
        self.step_file(sc, 33).unlink()
        self.assertProblem(self.log_problems(sc), "train_data_step33.jsonl is missing")
        sc = self.fresh()
        shutil.copyfile(self.step_file(sc, 60), self.step_file(sc, 61))
        self.assertProblem(self.log_problems(sc), "train_data_step61.jsonl, beyond")

    def test_step_log_shape(self):
        sc = self.fresh()
        p = self.step_file(sc, 2)
        lines = read_jsonl(p)
        lines[3]["rewards"] = [0.5, 0.5]
        p.write_text("".join(json.dumps(l) + "\n" for l in lines), encoding="utf-8")
        self.assertProblem(self.log_problems(sc), "rewards")
        lines[3]["rewards"] = [lines[2]["rewards"][0]]
        lines[3]["idx"] = 7
        p.write_text("".join(json.dumps(l) + "\n" for l in lines), encoding="utf-8")
        self.assertProblem(self.log_problems(sc), "idx")
        p.write_text("".join(json.dumps(l) + "\n" for l in lines[:100]), encoding="utf-8")
        self.assertProblem(self.log_problems(sc), "logs 100 samples")

    def test_two_experiment_dirs(self):
        sc = self.fresh()
        shutil.copytree(sc.out / "nemo" / "logs" / "exp_001", sc.out / "nemo" / "logs" / "exp_002")
        self.assertProblem(self.log_problems(sc), "exp_")

    def test_cut_examples_consumed(self):
        sc = self.fresh()
        sc.edit_json(sc.out / "cuts" / "checkpoint-20" / "cut.json", lambda c: c.update(examples_consumed=2560))
        self.assertProblem(self.log_problems(sc), "examples_consumed")

    def test_profile_settings(self):
        for key, val in (("generations", 4), ("prompts", 32)):
            sc = self.fresh()
            sc.edit_json(sc.out / "run_args.json", lambda a: a["profile_settings"].update({key: val}))
            self.assertProblem(self.log_problems(sc), key)

    def test_refused_at_collect(self):
        sc = self.env(lambda x: x[10].update(reward=0.99))
        self.refused(sc, "EXT-CONSUMPTION", "reward")

    def test_unreadable_env_log(self):
        sc = self.fresh()
        self.unreadable(sc.out / "env_calls.jsonl")
        self.refused(sc, "EXT-CONSUMPTION")

    def test_step_logs_must_postdate_the_launch(self):
        """EXT-64: NeMo RL's step logs are read outputs (extra_read_outputs)."""
        sc = self.fresh()
        os.utime(self.step_file(sc, 5), (1_000_000_000, 1_000_000_000))
        self.refused(sc, "EXT-STAGE", "predate")


# ---------------------------------------------------------------------------
# EXT-205: base, data and run bindings
# ---------------------------------------------------------------------------

class TestBindings(VlmCase):

    def test_clean(self):
        self.assertEqual(self.check_problems(self.sc), [])

    def test_base_ref_files_and_revision(self):
        sc = self.fresh()
        cut = sc.out / "cuts" / "checkpoint-40"
        sc.edit_json(cut / "base_ref.json", lambda b: b["files"]["config.json"].update(sha256="sha256:" + "0" * 64))
        sc.reseal(cut)
        self.assertProblem(self.check_problems(sc), "base_ref.json files")
        sc = self.fresh()
        cut = sc.out / "cuts" / "checkpoint-20"
        sc.edit_json(cut / "base_ref.json", lambda b: b.update(revision="main"))
        sc.reseal(cut)
        self.assertProblem(self.check_problems(sc), "revision")
        sc = self.fresh()
        cut = sc.out / "cuts" / "checkpoint-20"
        sc.edit_json(cut / "base_ref.json", lambda b: b.update(init_checkpoint={"path": "/inputs/x"}))
        sc.reseal(cut)
        self.assertProblem(self.check_problems(sc), "init_checkpoint")

    def test_base_not_through_the_mount(self):
        sc = self.fresh()
        sc.edit_json(sc.out / "run_args.json", lambda a: a.update(base_model="/inputs/other",
                                                                   model_name_or_path="/inputs/other"))
        self.assertProblem(self.check_problems(sc), "base_model")
        sc = self.fresh()
        sc.edit_json(sc.out / "run_args.json", lambda a: a.update(model_name_or_path="/inputs/other"))
        self.assertProblem(self.check_problems(sc), "model_name_or_path")
        sc = self.fresh()
        sc.edit_json(sc.out / "run_args.json", lambda a: a.update(init_checkpoint={"path": "/inputs/sft-cut"}))
        self.assertProblem(self.check_problems(sc), "init_checkpoint")

    def test_train_file(self):
        sc = self.fresh()
        sc.edit_json(sc.out / "run_args.json", lambda a: a.update(train_file_sha256="sha256:" + "1" * 64))
        self.assertProblem(self.check_problems(sc), "train_file_sha256")
        sc = self.fresh()
        with open(sc.runtime / "data" / "train.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(V.train_row(999)) + "\n")
        sc.edit_json(sc.out / "run_args.json", lambda a: a.update(
            train_file_sha256=file_sha256(sc.runtime / "data" / "train.jsonl")))
        self.assertProblem(self.check_problems(sc), "concatenation")
        sc = self.fresh()
        sc.edit_json(sc.out / "run_args.json", lambda a: a.update(train_file="/out/train.jsonl"))
        self.assertProblem(self.check_problems(sc), "train_file")

    def test_spec_data_not_the_shards(self):
        problems = adapter().checks(last_cut=self.sc.cut, base=self.sc.base,
                                    launch=json.loads((self.sc.root / "launch.json").read_text()), base_field="base",
                                    stage_out=self.sc.out, data=self.sc.runtime / "data", data_field="data")
        self.assertProblem(problems, "data/shards")

    def test_seed_profile_dry_run_steps_rung(self):
        for change, text in (({"seed": 7101}, "training_seed"), ({"profile": "smoke"}, "profile"),
                             ({"dry_run": True}, "dry_run"), ({"attempt": 0}, "attempt"),
                             ({"max_steps": 61}, "max_steps"), ({"memory_rung": "d"}, "memory_rung"),
                             ({"stage": "grpo"}, "stage")):
            with self.subTest(change=change):
                sc = self.fresh()
                sc.edit_json(sc.out / "run_args.json", lambda a: a.update(change))
                self.assertProblem(self.check_problems(sc), text)

    def test_cut_seed_profile_stage_script(self):
        for key, val in (("seed", 1), ("profile", "smoke"), ("script_sha256", "sha256:" + "2" * 64)):
            with self.subTest(key=key):
                sc = self.fresh()
                cut = sc.out / "cuts" / "checkpoint-40"
                sc.edit_json(cut / "cut.json", lambda c: c.update({key: val}))
                self.assertProblem(self.check_problems(sc), f"checkpoint-40/cut.json {key}")

    def test_helper_files_bound(self):
        for key, name in (("hooks_sha256", "ptc_vlm_hooks.py"), ("env_sha256", "vlm_env_logged.py"),
                          ("prompt_file_sha256", "prompt_clevr_cot.txt")):
            with self.subTest(key=key):
                sc = self.fresh()
                (sc.runtime / name).write_text("# changed after the run\n")
                self.assertProblem(self.check_problems(sc), key)

    def test_required_overrides(self):
        for ov in ("data.shuffle=false", "data_plane.enabled=false"):
            with self.subTest(ov=ov):
                sc = self.fresh()
                sc.edit_json(sc.out / "run_args.json", lambda a: a["nemo_rl_overrides"].remove(ov))
                self.assertProblem(self.check_problems(sc), ov)

    def test_framework(self):
        sc = self.fresh()
        sc.edit_json(sc.out / "run_args.json", lambda a: a["framework"].update(commit="0" * 40))
        self.assertProblem(self.check_problems(sc), "framework.commit")
        sc = self.fresh()
        (sc.out / "env-freeze.txt").write_text("torch==2.13.0\n# nemo-rl commit " + "0" * 40 + "\n")
        self.refused(sc, "EXT-FREEZE", "pinned")

    def test_record_argv_bound_to_the_run(self):
        rec = json.loads((self.sc.root / "launch.json").read_text())
        for i, val, text in ((7, "7101", "--seed"), (3, "smoke", "--profile"), (11, "b", "--memory")):
            with self.subTest(opt=text):
                bad = dict(rec, argv=[*rec["argv"][:i], val, *rec["argv"][i + 1:]])
                self.assertProblem(self.check_problems(self.sc, bad), text)
        bad = dict(rec, argv=rec["argv"] + ["--max-steps", "59"])
        self.assertProblem(self.check_problems(self.sc, bad), "--max-steps")
        bad = dict(rec, argv=[rec["argv"][0], f"{V.AREA7}/gen_vqa.py", *rec["argv"][2:]])
        self.assertProblem(self.check_problems(self.sc, bad), "argv[1]")

    def test_full_run_with_fewer_steps(self):
        """Without --max-steps the full profile trains 60 steps: a 40-step run under such a record refuses."""
        sc = self.fresh(steps=40)
        self.assertProblem(self.check_problems(sc), "trains 60 steps")

    def test_cut_sampler_not_registered(self):
        sc = self.fresh()
        cut = sc.out / "cuts" / "checkpoint-60"
        sc.edit_json(cut / "sampler.json", lambda s: s.update(max_new_tokens=256))
        sc.reseal(cut)
        self.refused(sc, "EXT-SAMPLER", "sampler.json")
        bad = dict(V.SAMPLER, prompt_block_sha256="sha256:" + "5" * 64)
        self.assertProblem(adapter().sampler_problems([self.sc.cut], bad), "prompt file")

    def test_binding_refused_at_collect(self):
        sc = self.fresh()
        sc.edit_json(sc.out / "run_args.json", lambda a: a.update(seed=1))
        self.refused(sc, "EXT-LAUNCH", "training_seed")

    def test_unreadable_freeze_and_cut_file(self):
        sc = self.fresh()
        self.unreadable(sc.out / "env-freeze.txt")
        self.refused(sc, "EXT-FREEZE")
        sc = self.fresh()
        self.unreadable(sc.out / "cuts" / "checkpoint-60" / "config.json")
        self.refused(sc, "EXT-STAGE")


# ---------------------------------------------------------------------------
# EXT-206: exact overlap on "<image sha256>\n<question>"
# ---------------------------------------------------------------------------

class TestOverlap(VlmCase):

    def test_clean_although_questions_repeat(self):
        """The items ask the training questions of other images: no overlap (the image hash is part of the key)."""
        items = self.sc.items["valA"]
        self.assertTrue({it["question"] for it in items} <= set(V.QUESTIONS))
        self.assertEqual(adapter().item_overlap_problems(items, self.sc.data), [])
        payloads, _ = adapter().training_payloads(self.sc.out, self.sc.base, [], data=self.sc.data)
        self.assertEqual(sum(it["text"] in payloads for it in items), 0)
        self.assertEqual(sum(it["question"] in {p.split("\n", 1)[1] for p in payloads} for it in items), len(items))

    def test_same_image_and_question_is_an_overlap(self):
        sc = self.fresh()
        rows = read_jsonl(sc.items_dir["valA"] / "items.jsonl")
        r = sc.rows[5]
        rows[0].update(image_sha256=r["image_sha256"], question=r["messages"][0]["content"][1]["text"], text=r["input"])
        V.write_jsonl(sc.items_dir["valA"] / "items.jsonl", rows)
        self.refused(sc, "EXT-CONTAMINATION", "1 item(s)")

    def test_same_image_other_question_is_not(self):
        items = [dict(it) for it in self.sc.items["valA"]]
        r = self.sc.rows[5]
        q = "How many purple things are there?"
        items[0].update(image_sha256=r["image_sha256"], question=q, text=f"{r['image_sha256']}\n{q}")
        payloads, _ = adapter().training_payloads(self.sc.out, self.sc.base, [], data=self.sc.data)
        self.assertNotIn(items[0]["text"], payloads)
        self.assertEqual(adapter().item_overlap_problems(items, self.sc.data), [])

    def test_item_key_malformed(self):
        items = [dict(it) for it in self.sc.items["valA"]]
        items[1]["text"] = items[1]["question"]                      # the question alone is not the key
        self.assertProblem(adapter().item_overlap_problems(items, self.sc.data), "clevr-valA-0001")
        items = [dict(it) for it in self.sc.items["valA"]]
        items[2]["image_sha256"] = "sha256:" + "a" * 64
        self.assertProblem(adapter().item_overlap_problems(items, self.sc.data), "clevr-valA-0002")

    def test_training_key_malformed(self):
        sc = self.fresh()
        rows = read_jsonl(sc.data / "shard-00001.jsonl")
        rows[0]["input"] = rows[0]["messages"][0]["content"][1]["text"]
        V.write_jsonl(sc.data / "shard-00001.jsonl", rows)
        self.assertProblem(adapter().item_overlap_problems(sc.items["valA"], sc.data), "shard-00001.jsonl line 1")

    def test_allowed_overlap_still_needs_well_formed_keys(self):
        """A registration may allow exact overlap (MM-11); the payload keys are checked whatever it says."""
        sc = self.fresh()
        rows = read_jsonl(sc.items_dir["valA"] / "items.jsonl")
        rows[0]["text"] = rows[0]["question"]
        V.write_jsonl(sc.items_dir["valA"] / "items.jsonl", rows)
        self.refused(sc, "EXT-CONTAMINATION", "exact-overlap key")


# ---------------------------------------------------------------------------
# EXT-207, EXT-208: arms (score_vqa.py outputs) and gen_meta.json
# ---------------------------------------------------------------------------

class TestArms(VlmCase):

    def arm_problems(self, sc: V.VlmScenario, which: int = 1, split: str = "valA") -> list[str]:
        from gatedtrain.ext.arm import read_lines
        arm = sc.arms[split][which] / "arm-exact_match"
        recs, bad = read_lines(arm, seeds=[11], score_range=(0, 1), error_classes=("missing_output", "scorer_error"))
        self.assertEqual(bad, [])
        model_dir = sc.cut if which else sc.base
        return adapter().arm_problems(arm, recs, sc.items[split], items_dir=sc.items_dir[split], model_dir=model_dir)

    def edit_output(self, sc: V.VlmScenario, fn, which: int = 1, i: int = 1, line=None) -> None:
        out = sc.arms["valA"][which]
        item = sc.items["valA"][i]["item"]
        p = out / "outputs" / f"{item}__r0.json"
        doc = json.loads(p.read_text())
        fn(doc)
        p.unlink()                                       # the arm holds hard links; break them, then relink
        V.write_json(p, doc)
        arm = out / "arm-exact_match"
        (arm / "outputs" / p.name).unlink()
        os.link(p, arm / "outputs" / p.name)

        def fix(rows):
            for r in rows:
                if r["item"] == item:
                    r["output_ref"]["sha256"] = file_sha256(p)
                    if line:
                        line(r)
        sc.edit_jsonl(arm / "items.jsonl", fix)

    def test_clean(self):
        self.assertEqual(self.arm_problems(self.sc, 0), [])
        self.assertEqual(self.arm_problems(self.sc, 1), [])
        self.assertEqual(self.arm_problems(self.sc, 1, "valB"), [])

    def test_score_not_rederived(self):
        sc = self.fresh()
        i = next(k for k, it in enumerate(sc.items["valA"]) if V.candidate_kind(k) == "wrong")

        def both(d):
            d["score"] = 1.0
        self.edit_output(sc, both, i=i, line=lambda r: r.update(score=1.0))
        self.assertProblem(self.arm_problems(sc), "exactly when they are equal")

    def test_parsed_answer_not_the_parser_s(self):
        sc = self.fresh()
        self.edit_output(sc, lambda d: d.update(parsed_answer="7"))
        self.assertProblem(self.arm_problems(sc), "parsed_answer")

    def test_response_edited(self):
        """The response is re-parsed: an output whose response no longer gives its parsed answer refuses."""
        sc = self.fresh()
        self.edit_output(sc, lambda d: d.update(response=d["response"] + " <answer>9</answer>"))
        self.assertProblem(self.arm_problems(sc), "parsed_answer")

    def test_sub_metrics(self):
        for change, text in (({"answer_tagged": 0}, "answer_tagged"), ({"is_yes_no": 1}, "is_yes_no"),
                             ({"truncated": 2}, "0 or 1"), ({"extra": 1}, "not exactly"),
                             ({"n_new_tokens": -1}, "n_new_tokens")):
            with self.subTest(change=change):
                sc = self.fresh()
                self.edit_output(sc, lambda d: d["sub_metrics"].update(change), i=0,
                                 line=lambda r: r["sub_metrics"].update(change))
                self.assertProblem(self.arm_problems(sc), text)

    def test_output_of_another_item_or_question(self):
        sc = self.fresh()
        self.edit_output(sc, lambda d: d.update(item="clevr-valA-0009"))
        self.assertProblem(self.arm_problems(sc), "names item")
        sc = self.fresh()
        self.edit_output(sc, lambda d: d.update(reference="no"))
        self.assertProblem(self.arm_problems(sc), "reference")

    def test_output_score_differs_from_the_line(self):
        sc = self.fresh()
        self.edit_output(sc, lambda d: d.update(score=0.5))
        self.assertProblem(self.arm_problems(sc), "differ from the arm's line")

    def test_error_records(self):
        """A missing_output line has an output without a response; a scorer_error line a reference without a
        canonical form (score_vqa.py writes the output either way)."""
        sc = self.fresh()
        err = {"class": "missing_output", "message": "no generation for this (item, seed)"}
        self.edit_output(sc, lambda d: d.update(response=None, parsed_answer=None, score=None, sub_metrics=None),
                         line=lambda r: r.update(error=err, score=None, sub_metrics=None))
        self.assertEqual(self.arm_problems(sc), [])
        sc = self.fresh()
        self.edit_output(sc, lambda d: d.update(score=None, sub_metrics=None),
                         line=lambda r: r.update(error=err, score=None, sub_metrics=None))
        self.assertProblem(self.arm_problems(sc), "missing_output")
        sc = self.fresh()
        err2 = {"class": "scorer_error", "message": "reference '1' has no canonical form"}
        self.edit_output(sc, lambda d: d.update(score=None, sub_metrics=None, parsed_answer=None),
                         line=lambda r: r.update(error=err2, score=None, sub_metrics=None))
        self.assertProblem(self.arm_problems(sc), "scorer_error")

    def test_scoring_record(self):
        for change, text in (({"items_sha256": "sha256:" + "6" * 64}, "items_sha256"),
                             ({"code_sha256": {"score_vqa.py": "sha256:" + "7" * 64}}, "pinned score_vqa.py"),
                             ({"scorer_code_sha256": "sha256:" + "8" * 64}, "scorer_code_sha256")):
            with self.subTest(change=change):
                sc = self.fresh()
                sc.edit_json(sc.arms["valA"][1] / "env-runtime.json", lambda e: e.update(change))
                self.assertProblem(self.arm_problems(sc), text)
        sc = self.fresh()
        (sc.arms["valA"][1] / "env-runtime.json").unlink()
        self.assertProblem(self.arm_problems(sc), "env-runtime.json")

    def edit_gen_meta(self, sc: V.VlmScenario, fn, *, which: int = 1, reseal: bool = True) -> None:
        root = sc.arms["valA"][which]
        sc.edit_json(root / "gen_meta.json", fn)
        if reseal:
            sc.edit_json(root / "env-runtime.json", lambda e: e.update(gen_meta_sha256=file_sha256(root / "gen_meta.json")))

    def test_gen_meta_of_another_model(self):
        sc = self.fresh()
        other = sc.out / "cuts" / "checkpoint-40"
        self.edit_gen_meta(sc, lambda g: g.update(model_files=V.gen_meta(other, sc.items_dir["valA"], 40)["model_files"]))
        self.assertProblem(self.arm_problems(sc), "model_files")
        sc = self.fresh()
        self.edit_gen_meta(sc, lambda g: g.update(model_files=V.gen_meta(sc.cut, sc.items_dir["valA"], 40)["model_files"]),
                           which=0)
        self.assertProblem(self.arm_problems(sc, 0), "model_files")

    def test_gen_meta_cut_marker(self):
        sc = self.fresh()
        self.edit_gen_meta(sc, lambda g: g.update(cut=None))
        self.assertProblem(self.arm_problems(sc), "cut.json")

    def test_gen_meta_fields(self):
        for change, text in (({"items_sha256": "sha256:" + "9" * 64}, "items_sha256"),
                             ({"prompt_block_sha256": "sha256:" + "a" * 64}, "prompt_block_sha256"),
                             ({"backend": "hf"}, "backend"), ({"eval_seeds": [12]}, "eval_seeds"),
                             ({"versions": {"vllm": "0.30.0"}}, "framework")):
            with self.subTest(change=change):
                sc = self.fresh()
                self.edit_gen_meta(sc, lambda g: g.update(change))
                self.assertProblem(self.arm_problems(sc), text)
        sc = self.fresh()
        self.edit_gen_meta(sc, lambda g: g["sampler"].update(max_new_tokens=256))
        self.assertProblem(self.arm_problems(sc), "sampler")

    def test_gen_meta_not_the_one_scored(self):
        sc = self.fresh()
        self.edit_gen_meta(sc, lambda g: g.update(wall_seconds=1.0), reseal=False)
        self.assertProblem(self.arm_problems(sc), "gen_meta_sha256")
        sc = self.fresh()
        (sc.arms["valA"][1] / "gen_meta.json").unlink()
        self.assertProblem(self.arm_problems(sc), "gen_meta.json beside the arm")

    def test_refused_at_collect(self):
        sc = self.fresh()
        self.edit_output(sc, lambda d: d.update(parsed_answer="7"))
        self.refused(sc, "EXT-ARM", "parsed_answer")

    def test_real_arm_fixture(self):
        """The base probe's scoring (12 items, copied): MM-1 lines and the adapter's checks pass but for the scorer,
        which is the one before the renames to ptc-hub 0.2.0 and gatedtrain 0.3.0; the hashes are null until ext-arm-fill; its scorer is the
        gate's but for the Python version (scored on the host's 3.15.2, not in the image), so collect would refuse it
        (meta_differences)."""
        from gatedtrain.ext.arm import meta_differences, read_lines, read_meta
        tmp = Path(tempfile.mkdtemp(prefix="gatedtrain-ext-vlm-arm-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        shutil.copytree(V.REAL_ARM, tmp / "real-arm")
        arm = tmp / "real-arm" / "arm-exact_match"
        V.renamed_copy(arm / "eval_meta.json")
        meta, bad = read_meta(arm)
        self.assertEqual(bad, ["eval_meta.json checkpoint is not {model_hash, base_model_hash} (MM-2)"])
        meta = json.loads((arm / "eval_meta.json").read_text())
        recs, bad = read_lines(arm, seeds=[11], score_range=(0, 1), error_classes=("missing_output", "scorer_error"))
        self.assertEqual((len(recs), bad), (12, []))
        items = read_jsonl(V.VLM / "items-valA-first12.jsonl")
        self.assertEqual(adapter().arm_problems(arm, recs, items), [GATE_SCORER_REFUSAL])
        self.assertEqual({k: v for k, v in meta["scorer"].items() if k != "version"},
                         {k: v for k, v in V.SCORER.items() if k != "version"} | {"code_sha256": V.SCORER_CODE_GATE})
        self.assertEqual(meta["scorer"]["version"], "clevr_exact_v1; python 3.15.2")
        self.assertTrue(any("scorer" in m for m in meta_differences(meta, {"scorer": V.SCORER})))
        self.assertEqual(meta["sampler"], V.SAMPLER)
        self.assertEqual(meta["framework"], {"name": "vllm", "version": "0.29.0"})


class TestArmFill(VlmCase):

    def test_fill_base_and_cut(self):
        sc = self.fresh()
        out = sc.output_root / "area7-vlm" / "eval" / "unfilled"
        V.build_score(out, sc.items["valA"], sc.items_dir["valA"], V.baseline_kind, model=None, item_set=None,
                      model_dir=sc.base)
        proc = run_cli(["ext-arm-fill", "--adapter", V.ADAPTER, "--checkpoint", str(sc.base), "--items",
                        str(sc.items_dir["valA"]), str(out / "arm-exact_match")], cwd=sc.root, env=sc.env())
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        meta = json.loads((out / "arm-exact_match" / "eval_meta.json").read_text())
        self.assertEqual(meta["checkpoint"]["model_hash"], sc.base_model)
        self.assertEqual(meta["item_set_hash"], sc.item_set_hash["valA"])
        self.assertEqual(meta["sampler_hash"], sha_of(V.SAMPLER))        # compact JSON converted (floats 0.0, 1.0)
        out2 = sc.output_root / "area7-vlm" / "eval" / "unfilled-cut"
        V.build_score(out2, sc.items["valA"], sc.items_dir["valA"], V.candidate_kind, model=None, item_set=None,
                      model_dir=sc.cut)
        proc = run_cli(["ext-arm-fill", "--adapter", V.ADAPTER, "--checkpoint", str(sc.cut), "--items",
                        str(sc.items_dir["valA"]), str(out2 / "arm-exact_match")], cwd=sc.root, env=sc.env())
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertEqual(json.loads((out2 / "arm-exact_match" / "eval_meta.json").read_text())["checkpoint"]["model_hash"],
                         sc.cand_model)


# ---------------------------------------------------------------------------
# EXT-209, EXT-210: launch requests (ext-launch, --recheck)
# ---------------------------------------------------------------------------

class TestLaunchRules(VlmCase):

    def launch(self, req: dict, *, profile: str = "smoke", extra: list[str] = (), smoke: Path | None = None):
        type(self).n += 1
        return self.sc.launch(req, f"lr-{type(self).n}", profile=profile, extra=list(extra), smoke=smoke)

    def out(self) -> Path:
        type(self).n += 1
        return self.sc.new_out(f"lr-out-{type(self).n}")

    def refused_req(self, req: dict, text: str, *, profile: str = "smoke", field: str | None = None):
        proc, rec = self.launch(req, profile=profile)
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertNotIn("Traceback", proc.stderr)
        self.assertIn(text, proc.stderr)
        if field is not None:
            self.assertIn(field, [e["field"] for e in parse_errors(proc.stderr)])
        self.assertFalse(rec.exists())

    def test_train_request_accepted(self):
        proc, rec = self.launch(self.sc.request(self.out()))
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        line = proc.stdout.strip()
        self.assertEqual(len(proc.stdout.strip().splitlines()), 1)                     # no sidecar
        for tok in ("--runtime=nvidia --gpus all", "--network none", "--user 1000:1000", "--shm-size=16g",
                    "--ulimit memlock=-1", "--ulimit nofile=65535:65535", "-w /opt/nemo-rl",
                    f"{V.IMAGE} {V.TRAIN_PY} {V.AREA7}/run_vlm_nemo.py --profile smoke"):
            self.assertIn(tok, line)
        self.assertNotIn("--shm-size=8g", line)
        record = json.loads(rec.read_text())
        self.assertEqual((record["interpreter_path"], record["interpreter_sha256"]), (V.TRAIN_PY, None))
        self.assertEqual(record["script_sha256"], file_sha256(self.sc.runtime / "run_vlm_nemo.py"))
        self.assertEqual(self.sc.recheck(rec).returncode, EXIT_OK)

    def test_tierb_image_accepted(self):
        proc, _ = self.launch(self.sc.request(self.out(), runtime={"kind": "container", "image": V.TIERB_IMAGE}))
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_wrong_image_refused(self):
        self.refused_req(self.sc.request(self.out(), runtime={"kind": "container", "image": V.EXECUTOR_IMAGE}),
                         "pinned NeMo RL image")
        self.refused_req(self.sc.request(self.out(), runtime={"kind": "container", "image": "sha256:" + "c" * 64}),
                         "pinned NeMo RL image")

    def test_root_user_refused(self):
        self.refused_req(self.sc.request(self.out(), user="0:0"), "root")

    def test_network_refused(self):
        self.refused_req(self.sc.request(self.out(), network="host"), "host")
        self.refused_req(self.sc.request(self.out(), network="bridge", network_reason="download"), "network none")

    def test_credential_env_refused(self):
        req = self.sc.request(self.out())
        req["env"]["HF_TOKEN"] = "x"
        self.refused_req(req, "credential")
        req = self.sc.request(self.out())
        req["env"]["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"          # not on this row's list
        self.refused_req(req, "PYTORCH_CUDA_ALLOC_CONF")

    def test_train_argv_rules(self):
        cases = [(["--memory", "d"], "--memory"), (["--base", "/inputs/other"], "--base"), (["--out", "/out/x"], "--out"),
                 (["--profile", "full"], "--profile"), (["--seed", "x"], "--seed"), (["--max-steps", "0"], "--max-steps")]
        for extra, text in cases:
            with self.subTest(extra=extra):
                req = self.sc.request(self.out())
                req["argv"] = req["argv"] + extra
                self.refused_req(req, text)
        req = self.sc.request(self.out())
        req["argv"] = [a for a in req["argv"] if a not in ("--memory", "a")]
        self.refused_req(req, "--memory")
        req = self.sc.request(self.out())
        req["argv"] = req["argv"] + ["--max-steps", "3"]
        proc, _ = self.launch(req)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        req = self.sc.request(self.out())
        req["argv"][7] = "7101"
        self.refused_req(req, "training_seed")

    def test_full_launch_needs_a_pass_smoke(self):
        req = self.sc.request(self.out(), argv=self.sc.argv("full"))
        self.refused_req(req, "smoke", profile="full")
        proc, _ = self.launch(self.sc.request(self.out(), argv=self.sc.argv("full")), profile="full",
                              smoke=self.sc.root / "smoke.json")
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        req = self.sc.request(self.out(), argv=self.sc.argv("full") + ["--max-steps", "5"])
        proc, _ = self.launch(req, profile="full", smoke=self.sc.root / "smoke.json")
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertIn("--max-steps under --profile full", proc.stderr)

    def test_script_copy_and_interpreters(self):
        req = self.sc.request(self.out())
        req["argv"][1] = f"{V.AREA7}/probe/run_vlm_nemo.py"
        self.refused_req(req, "argv[1]")
        req = self.sc.request(self.out())
        req["argv"][0] = V.GEN_PY
        self.refused_req(req, "argv[0]")
        req = self.sc.request(self.out())
        req["argv"][0] = "/usr/bin/python3"
        self.refused_req(req, "argv[0]")
        req = self.sc.request(self.out())
        req["argv"].insert(1, "-I")                     # only score_vqa.py takes -I
        self.refused_req(req, "argv[0]")

    def test_mount_sets(self):
        out = self.out()
        req = self.sc.request(out)
        req["mounts"].append({"host": str(self.sc.base), "container": "/inputs/base", "mode": "ro"})
        self.refused_req(req, "not exactly")
        req = self.sc.request(self.out())
        req["mounts"] = [m for m in req["mounts"] if m["mode"] == "rw"]
        self.refused_req(req, "argv")

    def test_sidecar_refused(self):
        req = self.sc.request(self.out())
        req["sidecar"] = {"runtime": {"kind": "venv_in_container", "image": V.EXECUTOR_IMAGE}}
        self.refused_req(req, "sidecar")

    def test_generation_request(self):
        proc, rec = self.launch(self.sc.eval_request("gen_vqa.py", self.out()))
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertIn(f"--shm-size=16g", proc.stdout)
        self.assertIn(f"{V.IMAGE} {V.GEN_PY} {V.AREA7}/gen_vqa.py", proc.stdout)
        proc, _ = self.launch(self.sc.eval_request("gen_vqa.py", self.out(), cut=self.sc.cut, items="items-valB"))
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        for change, text in ((("--items", f"{V.AREA7}/items"), "--items"), (("--model", "/inputs/other"), "--model"),
                             (("--out", "/out/x"), "--out")):
            with self.subTest(change=change):
                req = self.sc.eval_request("gen_vqa.py", self.out())
                i = req["argv"].index(change[0])
                req["argv"][i + 1] = change[1]
                self.refused_req(req, text)
        req = self.sc.eval_request("gen_vqa.py", self.out())
        req["argv"] += ["--limit", "20"]
        proc, _ = self.launch(req)
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        req = self.sc.eval_request("gen_vqa.py", self.out())
        req["argv"] += ["--limit", "20"]
        self.refused_req(req, "--limit", profile="full")
        self.refused_req(self.sc.eval_request("gen_vqa.py", self.out(), gpus=False, gpu_est_h=0), "GPU")
        req = self.sc.eval_request("gen_vqa.py", self.out())             # a cut mount without --model /inputs/cut
        req["mounts"].insert(1, {"host": str(self.sc.cut), "container": "/inputs/cut", "mode": "ro"})
        self.refused_req(req, "not exactly")

    def test_scoring_request(self):
        """EXT-210: `python -I score_vqa.py` with the image's main interpreter, CPU, network none."""
        proc, rec = self.launch(self.sc.eval_request("score_vqa.py", self.out(), gen=self.out()))
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertIn(f"{V.IMAGE} {V.TRAIN_PY} -I {V.AREA7}/score_vqa.py --gen /inputs/gen", proc.stdout)
        self.assertNotIn("--gpus", proc.stdout)
        self.assertIn("--network none", proc.stdout)
        record = json.loads(rec.read_text())
        self.assertEqual((record["script_path"], record["interpreter_sha256"]), (f"{V.AREA7}/score_vqa.py", None))
        self.assertEqual(self.sc.recheck(rec).returncode, EXIT_OK)
        proc, _ = self.launch(self.sc.eval_request("score_vqa.py", self.out(), gen=self.out(), items="items-valB"))
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))

    def test_scoring_request_rules(self):
        cases = [({"gpus": True, "gpu_est_h": 0.1}, None, "CPU"),
                 ({"network": "bridge", "network_reason": "x"}, None, "network none"),
                 ({"tmpfs": []}, None, "tmpfs"),
                 ({}, lambda a: a.remove("-I"), "python -I"),
                 ({}, lambda a: a.extend(["--self-test"]), "--self-test"),
                 ({}, lambda a: a.extend(["--evaluator-id", V.VALB_ID]), "--evaluator-id"),
                 ({}, lambda a: a.__setitem__(a.index("--gen") + 1, "/inputs/other"), "--gen")]
        for change, argv_fn, text in cases:
            with self.subTest(text=text):
                req = self.sc.eval_request("score_vqa.py", self.out(), gen=self.out(), **change)
                if argv_fn:
                    argv_fn(req["argv"])
                self.refused_req(req, text)
        req = self.sc.eval_request("score_vqa.py", self.out(), gen=self.out(), items="items-valB")
        req["argv"] = req["argv"][:-2]                   # valB without --evaluator-id: scored under the valA id
        self.refused_req(req, "--evaluator-id")
        req = self.sc.eval_request("score_vqa.py", self.out(), gen=self.out())
        req["argv"] += ["--limit", "20"]
        self.refused_req(req, "--limit", profile="full")
        req = self.sc.eval_request("score_vqa.py", self.out(), gen=self.out())
        req["argv"][0] = V.GEN_PY
        self.refused_req(req, "argv[0]")

    def test_flag_for_the_scorer_only(self):
        """EXT-210: the row allows `-I` before score_vqa.py only; the training script with `-I` is refused. (In gatedtrain this
        test checked area 6's row; the other adapters live in their own recipe repositories now.)"""
        from gatedtrain.ext.launch import bind_argv
        reg = registry()
        argv = [V.TRAIN_PY, "-I", f"{V.AREA7}/run_vlm_nemo.py"]
        binding, why = bind_argv(argv, [], image_interpreters=reg.image_interpreters(V.ADAPTER),
                                 interpreter_flags=reg.interpreter_flags(V.ADAPTER))
        self.assertIsNone(binding)
        self.assertIn("'-I'", why)


class TestSpecBinding(VlmCase):
    """EXT-211: `ext-launch --spec` binds the spec's base and data to read-only mounts (optional for this adapter)."""

    def out(self) -> Path:
        type(self).n += 1
        return self.sc.new_out(f"spec-out-{type(self).n}")

    def test_spec_optional_and_checked(self):
        spec = str(self.sc.root / "spec.yaml")
        proc, rec = self.sc.launch(self.sc.request(self.out()), "spec-ok", extra=["--spec", spec])
        self.assertEqual(proc.returncode, EXIT_OK, describe(proc))
        self.assertNotIn("Traceback", proc.stderr)
        proc2 = self.sc.recheck(rec, extra=["--spec", spec])
        self.assertEqual(proc2.returncode, EXIT_OK, describe(proc2))
        problems = adapter().mount_binding_problems([], base=self.sc.base, data=self.sc.data, base_field="b",
                                                    data_field="d")
        self.assertEqual([p.rule for p in problems], ["EXT-BASE", "EXT-DATA"])

    def test_spec_outside_the_mounts(self):
        type(self).n += 1
        elsewhere = self.tmp / f"elsewhere-{type(self).n}"
        shutil.copytree(self.sc.base, elsewhere / "base")
        text = (self.sc.root / "spec.yaml").read_text().replace(str(self.sc.base), str(elsewhere / "base"))
        spec = self.sc.root / f"spec-elsewhere-{type(self).n}.yaml"
        spec.write_text(text)
        proc, _ = self.sc.launch(self.sc.request(self.out()), f"spec-bad-{type(self).n}", extra=["--spec", str(spec)])
        self.assertEqual(proc.returncode, EXIT_REFUSED, describe(proc))
        self.assertNotIn("Traceback", proc.stderr)
        self.assertIn("artifacts.base.uri", [e["field"] for e in parse_errors(proc.stderr)])


# ---------------------------------------------------------------------------
# EXT-212: smoke records (binary deciding metric)
# ---------------------------------------------------------------------------

class TestSmoke(VlmCase):

    def problems(self, **overrides) -> list[str]:
        from gatedtrain.ext.smoke import smoke_problems
        row = registry().EVALUATOR_ROWS[V.EVALUATOR_ID]
        rec = V.smoke_record(base_model=self.sc.base_model, item_set=self.sc.item_set_hash["valA"], **overrides)
        return smoke_problems(rec, adapter=V.ADAPTER, score_range=row["score_range"], binary=row["binary"],
                              base_model_hash=self.sc.base_model, item_set_hash=self.sc.item_set_hash["valA"])

    def test_pass_with_a_low_binary_mean(self):
        """Binary metric: 80 % zeros is not degenerate (the 5 % margin and mode rules do not apply)."""
        self.assertEqual(self.problems(), [])
        self.assertEqual(self.problems(baseline_scores=[0.0] * 19 + [1.0]), [])

    def test_degenerate_only_at_zero_or_one(self):
        self.assertProblem(self.problems(baseline_scores=[0.0] * 20), "degenerate_mean")
        self.assertProblem(self.problems(baseline_scores=[1.0] * 20), "degenerate_mean")

    def test_other_failures(self):
        self.assertProblem(self.problems(peak_gpu_mem_gb=23.0), "gpu_memory")
        self.assertProblem(self.problems(error_count=1), "evaluator_errors")
        self.assertProblem(self.problems(adapter="other_adapter"), "adapter")
        self.assertProblem(self.problems(base_model_hash="sha256:" + "0" * 64), "base_model_hash")
        self.assertProblem(self.problems(item_set_hash="sha256:" + "0" * 64), "item_set_hash")

    def test_refused_at_collect(self):
        sc = self.fresh()
        write_canonical(sc.root / "smoke.json", V.smoke_record(base_model=sc.base_model,
                                                                item_set=sc.item_set_hash["valA"],
                                                                baseline_scores=[0.0] * 20))
        self.refused(sc, "EXT-SMOKE", "degenerate_mean")


if __name__ == "__main__":
    unittest.main()
