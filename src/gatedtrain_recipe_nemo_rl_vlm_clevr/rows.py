"""Registry rows of adapter `nemo_rl_vlm`: NeMo RL VLM GRPO on CLEVR (area 7).

The training row, the evaluator rows and the adapter class, loaded by `gatedtrain.ext.registry` through the
`gatedtrain.adapters` entry point (multimodal spec section 6)."""

from __future__ import annotations

from typing import Any

from gatedtrain.families.nemo_rl import NEMO_RL_COMMIT, NEMO_RL_PYTHON, NEMO_RL_RUN_OPTIONS, VLLM_WORKER_PYTHON
from gatedtrain.ext.rowkit import CONTAINER_PATHS, runtime_dir

NAME = "nemo_rl_vlm"
# The runtime directory: this repository's runtime/ (scripts, prompt, models, data, images, items; mounted read-only
# at its own host path, which is also its container path), or the override; earlier runs lie under the host's output
# root as `{output_root}/<OUT_NAME>`. Every script runs an interpreter of the pinned NeMo RL image, so there is no
# host interpreter tree.
RUNTIME_ENV = "GATEDTRAIN_RECIPE_NEMO_RL_VLM_CLEVR_RUNTIME"
OUT_NAME = "nemo-rl-vlm-clevr"
RUNTIME = runtime_dir(RUNTIME_ENV, __file__)

TRAINER_ROWS: dict[str, dict[str, Any]] = {}
EVALUATOR_ROWS: dict[str, dict[str, Any]] = {}

# Area "VLM" (multimodal spec section 6 rows 7 and 7s with the row 7 deviations; the area 7 decisions in
# runtime/README.md, 2026-11-14; runtime/run_vlm_nemo.py, ptc_vlm_hooks.py, vlm_env_logged.py, gen_vqa.py, score_vqa.py,
# read 2026-11-14, and its probe outputs; gatedtrain tests/README.md section 11.24, EXT-200 to EXT-213). One stage: NeMo RL's synchronous VLM GRPO (`rl.nemo_rl_vlm@0.1`) of Qwen3.5-0.8B (vision tower
# frozen) on CLEVR-CoGenT TrainA, full weights, in the derived NeMo RL image; no sidecar, no executor. Evaluation is two
# steps in the same image: greedy vLLM generation (gen_vqa.py, GPU) and scoring with the area's own standard-library
# answer parser (score_vqa.py, CPU, `python -I`).
TRAINER_ROWS["nemo_rl_vlm"] = {
    "uses": ("rl.nemo_rl_vlm@0.1",),
    # Pinned by NeMo RL's commit (env-freeze.txt `# nemo-rl commit <40 hex>`, ptc_vlm_hooks.write_run_args), as area 6.
    "pins": (),
    "nemo_rl_commit": NEMO_RL_COMMIT,
    "outputs": ("cuts/checkpoint-<step>/ full-weight cuts with cut.json written last", "run_args.json",
                "env-freeze.txt", "env_calls.jsonl", "nemo/logs/exp_<n>/train_data_step<N>.jsonl"),
    # EXT-201: ptc_vlm_hooks.py write_cut: the consolidated weights (`*.safetensors`, one or more, with their index when
    # NeMo RL writes one), config, tokenizer and processor files, the deployment sampler; cut.json `deployable` is every
    # file but base_ref.json, training_info.json and cut.json.
    "components": ("*.safetensors", "model.safetensors.index.json", "config.json", "tokenizer.json",
                   "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json",
                   "video_preprocessor_config.json", "vocab.json", "merges.txt", "sampler.json"),
    # EXT-204: observed per-row log (env_calls.jsonl of the logging VLMEnvironment joined with NeMo RL's step log); a
    # log that does not match the run refuses; the `iteration` fallback is an explicit decision, never automatic.
    "consumption": {"target": "row", "fallback": None},
    "gpu_h_estimate": 2.0,                           # estimate (README "Profiles and cost": area total 1.3-2.0)
    "evaluators": ("vqa.clevr_items@0.1", "vqa.clevr_valb_items@0.1"),   # deciding first (MM-38)
    "framework": "nemo_rl",
    # EXT-209: the runtime directory (scripts, model, data, images, items) and the training and evaluation outputs (an
    # evaluation mounts a cut read-only at /inputs/cut, scoring the generation output at /inputs/gen). No host
    # interpreter: every script runs an interpreter of the pinned image.
    "input_roots": (RUNTIME, "{output_root}/" + OUT_NAME),
    "spec_binding": False,                           # the smoke launch precedes the registration
    # Every name, none a credential (HF_TOKEN is never allowed).
    "env_names": ("FLASHINFER_CUDA_ARCH_LIST", "HF_HUB_OFFLINE", "HOME", "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED",
                  "RAY_memory_monitor_refresh_ms", "TORCH_CUDA_ARCH_LIST"),
    "scan_skip": (),
    "interpreter_trees": (),
    # No host interpreter: argv[0] is always one of the image's (`image_interpreters`, pinned by the image ID).
    "interpreter": NEMO_RL_PYTHON,
    "image_interpreters": {"run_vlm_nemo.py": NEMO_RL_PYTHON, "gen_vqa.py": VLLM_WORKER_PYTHON,
                           "score_vqa.py": NEMO_RL_PYTHON},
    "container_paths": CONTAINER_PATHS,
    "env_paths": ("HOME",),
    "run_options": {"run_vlm_nemo.py": NEMO_RL_RUN_OPTIONS, "gen_vqa.py": NEMO_RL_RUN_OPTIONS},
    # EXT-210: scoring runs the standard-library scorer in isolated mode (`python -I score_vqa.py`).
    "interpreter_flags": {"score_vqa.py": "-I"},
}

# Area "VLM" evaluators (decisions "Area 7 decisions": deciding `exact_match` on items-valA, T3 with the
# `construct_alignment` and `shared_annotation` disclosures; items-valB supporting only). Scorer: score_vqa.py (its own
# answer parser, not NeMo RL's exact_answer_alphanumeric_reward); code_sha256 is the `scorer_code_sha256` it writes
# (SHA-256 of the compact sorted JSON {"score_vqa.py": <its SHA-256>}, score_vqa.py sha256 2bb71c3c...: the 2026-11-14
# gate's f527e4f1... with the published name map applied (ptc-hub 0.2.0: 345e66b0..., gatedtrain 0.3.0: this), schema
# id `gatedtrain.item_eval` and nothing else; results/2026-11-14/gate/registration-ptc-hub.yaml and
# registration-gatedtrain.yaml `rename_proof`, RESULT.md); the version the string it writes in
# the image (`/opt/nemo_rl_venv/bin/python`, Python 3.13.14 per probe-a6 run_args.json `python`). An item is one
# registered (image, question) pair; its `text` "<image sha256>\n<question>" is the exact-overlap key against the
# training rows' `input` (EXT-206: a question repeated on another image is no overlap).
SCORE_VQA_SHA256 = "sha256:2bb71c3c85df376a87182703f53e91819b54a05e0486634da3030d1b109e1886"
SCORE_VQA_CODE_SHA256 = "sha256:7e07e8aec57723e83575ed9f8af29d863c2c8fd58c0f242b1ab92a45a3edf63f"
_VQA = {
    "direction": "higher_is_better",
    "score_range": (0, 1),
    "binary": True,
    "tier": "T3",
    # MM-4: evaluator faults (no generation for a registered (item, seed); a reference without a canonical form).
    "error_classes": ("missing_output", "scorer_error"),
    # MM-4: the policy's faults, scored 0 (a canonical answer that is not the reference; no canonical answer).
    "score_outcomes": ("wrong_answer", "no_answer"),
    "payload_text_field": "text",
    "item_payload": ("text", "question", "answer", "image_sha256"),
    "normalizer": False,
    "env_version": False,
    # The evaluator's sampler (score_vqa.py: gen_vqa.py SAMPLER plus the generation backend and the prompt file hash).
    "sampler_keys": ("backend", "chat_template", "max_model_len", "max_new_tokens", "messages", "prompt_block_sha256",
                     "samples_per_seed", "strategy", "temperature", "top_p"),
    "framework": "vllm",
    "samples": "500, 1",
    # 05 section 6.1, R15-1: another answer parser (a tool) than the reward's, the same construct.
    "construct_aligned": {
        "reward": "nemo_rl_vlm_format_exact_alnum",
        "statement": ("exact_match measures answer correctness on held-out CLEVR-CoGenT questions, the construct the "
                      "training reward (NeMo RL's VLM environment: 0.2 x format + 0.8 x exact_alnum on the <answer> "
                      "block) also rewards, with another answer parser (score_vqa.py: the last <answer> block, else the "
                      "last integer, number word or yes/no token, number words as digits; not "
                      "exact_answer_alphanumeric_reward): a pass shows that the model answers more held-out questions "
                      "correctly under an independent parser, partly through the answer format the reward pays for, "
                      "not a general gain in visual reasoning (05 section 6.1, R15-1)")},
    # 05 section 6.1, R16-1: the references come from the training answers' annotation process.
    "shared_annotation": {
        "convention": ("CLEVR functional-program answers from the same scene generator and question engine as the "
                       "training answers (integers 0 to 10 or yes/no)"),
        "statement": ("The references of exact_match are held-out CLEVR-CoGenT answers made by the same CLEVR program "
                      "executor over the same scene generator as the training answers; no training stage consumed "
                      "them, so the programmatic scorer stays T3 (05 section 6.1, R16-1). Part of a gain may be "
                      "learning that answer convention (a single count or yes/no): a pass shows more correct CLEVR "
                      "answers on unseen scenes, not a general gain in visual question answering.")},
}
_VQA_SCORER = {"name": "clevr_exact_match_own_parser", "version": "clevr_exact_v1; python 3.13.14",
               "weights_sha256": None, "code_sha256": SCORE_VQA_CODE_SHA256}
EVALUATOR_ROWS["vqa.clevr_items@0.1"] = {**_VQA, "metric": "exact_match", "scorer": dict(_VQA_SCORER)}
# items-valB (condition B: other colour-shape combinations): the same scorer, reported only (supporting).
EVALUATOR_ROWS["vqa.clevr_valb_items@0.1"] = {**_VQA, "metric": "exact_match", "scorer": dict(_VQA_SCORER)}

PLUGIN = {"adapters": {"nemo_rl_vlm": "gatedtrain_recipe_nemo_rl_vlm_clevr.adapter:NemoRlVlmAdapter"}, "trainer_rows": TRAINER_ROWS,
          "evaluator_rows": EVALUATOR_ROWS}
