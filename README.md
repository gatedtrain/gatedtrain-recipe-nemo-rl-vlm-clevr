# gatedtrain recipe: NeMo RL VLM GRPO on CLEVR-CoGenT

A [gatedtrain](https://github.com/gatedtrain/gatedtrain) recipe: NeMo RL's synchronous VLM GRPO trains Qwen3.5-0.8B
(full weights, vision tower frozen) on CLEVR-CoGenT TrainA questions, and gatedtrain gates the last cut against the base model on
`exact_match` of 500 held-out ValA items (ValB, other colour-shape combinations, is reported as supporting only).
Training, greedy vLLM generation and scoring all run in the derived NeMo RL image (`gatedtrain-a6-nemo-rl:1`, uid 1000,
`--network none`). Prototype.

The gate files under `results/` cite the maintainers' decision record (`decisions/MVP-MULTIMODAL.md`, "Area 7
decisions") and code comments cite the framework's review rounds ("Review 9 M1"): those records are not published.
`runtime/README.md` records the area 7 decisions this recipe depends on.

## Layout

| Path | What |
|---|---|
| `src/gatedtrain_recipe_nemo_rl_vlm_clevr/` | the gatedtrain adapter `nemo_rl_vlm` (`adapter.py`) and its registry rows (`rows.py`), registered through the `gatedtrain.adapters` entry point |
| `runtime/` | what runs in the container: `run_vlm_nemo.py` (launcher), `ptc_vlm_hooks.py` (cuts, run record), `vlm_env_logged.py` (row-id processor and logging VLM environment), `gen_vqa.py` (vLLM generation), `score_vqa.py` (standard-library answer parser), the prompt file, the host-side `fetch_models.py` and `make_data.py`, the pins (`models/MODELS.json`, `raw/RAW.json`), the data metadata and the two registered item sets (`items-valA/`, `items-valB/`) |
| `tests/` | the adapter's tests on fixture runs (no GPU, no model needed) and gatedtrain's conformance check |
| `results/2026-11-14/` | the gate files of the published result (redacted copies, host paths replaced by `<data-root>`) |

## Setup

```bash
pip install "gatedtrain @ git+https://github.com/gatedtrain/gatedtrain@v0.3.0"   # the framework, at the release this recipe was tested with
pip install -e .                      # the adapter, into the environment that runs gatedtrain
cd runtime
python fetch_models.py                # the base model and the three pinned parquet shards (about 2.8 GB; network)
python -I make_data.py                # images, training shards, train.jsonl (needs pyarrow)
```

The adapter finds the runtime at `runtime/` of this checkout (or `GATEDTRAIN_RECIPE_NEMO_RL_VLM_CLEVR_RUNTIME`). The
runtime is mounted read-only at its own host path, so that path is also the one the container sees. Every script runs
with an interpreter of the pinned NeMo RL image (`gatedtrain.families.nemo_rl`); there is no host interpreter. The
runs' outputs go under the host's gatedtrain output root (`<output_root>/nemo-rl-vlm-clevr`). `runtime/README.md` has the research notes,
the trainer settings found by GPU probes, the launch facts and the decisions behind the metric.

The adapter locates `runtime/` from its own source tree (`src/gatedtrain_recipe_nemo_rl_vlm_clevr/../../runtime`),
which exists only in a source checkout or an editable install (`pip install -e .`). After a non-editable install (a
wheel), set `GATEDTRAIN_RECIPE_NEMO_RL_VLM_CLEVR_RUNTIME` to the `runtime/` directory of a checkout.

Some runtime names predate the project's rename to gatedtrain and are kept because the published run and the
committed metadata carry them: the hooks module `runtime/ptc_vlm_hooks.py` (the adapter checks the run's
`hooks_sha256` against the file of that name beside the launched script) and the runtime's own schema ids `ptc_mm.*`
(`cut.json`, the data, raw-shard, model and item-set metadata, whose hashes are registered).

Note: the item sets name each image by its path relative to `runtime/` (`data/images/<sha256>.png`) and by its
SHA-256, so `make_data.py` regenerates them byte for byte on any host (the item-set hashes do not depend on where the
runtime lives). The training rows carry absolute image paths, because NeMo RL opens them inside the container where
the runtime is mounted at its host path; they are regenerated, never committed.
So `data/train.jsonl` and its SHA-256 depend on where the runtime lies. The committed `data/data_meta.json` records
`train_file_sha256` of the runtime the published run trained from (the run's `run_args.json` records the same value).
When everything else in `data_meta.json` comes out byte for byte the same, `make_data.py` keeps the committed file
unchanged and prints the local hash instead (`train_file_sha256 here: sha256:...`); any other difference writes fresh
metadata. A run you train from this runtime records the local hash in its own `run_args.json`, which the adapter checks
against `data/train.jsonl`, not against `data_meta.json`.

## Tests

```bash
python -m unittest discover -s tests -t tests
```

Tests that read regenerated files (the base model, the training shards) skip when they are absent. One test checks
that another recipe's training stage type (`rl.nemo_rl_env@0.1`) is refused for this adapter; it skips unless
`gatedtrain-recipe-nemo-rl-cad` is installed beside this recipe.

## Data and licenses

This repository is licensed under Apache-2.0 (`LICENSE`); `THIRD_PARTY_NOTICES.md` lists the third-party material it
contains (CLEVR-CoGenT questions and answers in the item sets and fixtures, NeMo RL's prompt file, Qwen3.5 tokenizer and
configuration files in a fixture).

| What | Source and pin | License | Notes |
|---|---|---|---|
| Base model Qwen3.5-0.8B | `Qwen/Qwen3.5-0.8B` @ `2fc06364715b967f1860aea9cf38778875588b17` (`runtime/fetch_models.py`, files checked by SHA-256 in `runtime/models/MODELS.json`) | Apache-2.0 (model card; the `LICENSE` file in the repository) | Qwen team, Alibaba Cloud. The trained cuts are derivatives of it |
| CLEVR-CoGenT v1.0 (images, questions, answers) | through MMInstruction's Hugging Face repackaging: `Clevr_CoGenT_TrainA_70K_Complex` @ `d079bf10…`, `Clevr_CoGenT_ValA` @ `f786093c…`, `Clevr_CoGenT_ValB` @ `a844ddba…` (one parquet shard each, SHA-256 in `runtime/raw/RAW.json`) | CC BY 4.0 (Johnson et al., CVPR 2017, https://cs.stanford.edu/people/jcjohns/clevr/) | the repackaging declares no license of its own; it is used under the source's CC BY 4.0. Images are downloaded, not redistributed; the item sets carry questions, answers and image hashes |
| NeMo RL | github.com/NVIDIA-NeMo/RL @ `4f39fd86289a2ffd5aaf5d202702f7fd7185c924` | Apache-2.0 | used unmodified inside its image; `runtime/prompt_clevr_cot.txt` is a byte copy of its `examples/prompts/clevr_cogent_cot.txt` |
| Trainer, generation and scoring image | `nemo-rl:4f39fd8` built from NeMo RL's `docker/Dockerfile` at that commit (base `nvcr.io/nvidia/cuda-dl-base:26.05-cuda13.2-devel-ubuntu24.04`, image ID `sha256:8cd5b714ea91ec7ea9a976c75b0ae6f6932140cf7a8aea803100de6331a2113a`), then gatedtrain's `containers/nemo-rl-uid1000/` on top (`gatedtrain-a6-nemo-rl:1`, image ID `sha256:7a4078e70b28cc6de622fa3cf25c7596c52337fa3f1779488b402c5fa1c74e69`, file ownership changes only) | NeMo RL Apache-2.0; the CUDA base image under NVIDIA's container license terms | built locally, not redistributed |

Load only model cuts and NeMo RL resume state (`nemo/checkpoints/`) that you produced yourself: NeMo RL's resume
state is a torch distributed checkpoint, which unpickles on load.

## License

This repository is licensed under the Apache License, Version 2.0 (`LICENSE`); see `NOTICE`. Third-party material it
contains is listed with its license in `THIRD_PARTY_NOTICES.md`; models, data and images the recipe downloads or
builds are listed under "Data and licenses" above. Names of third-party projects and products are trademarks of
their owners; this project is not affiliated with or endorsed by them.

## Result

See `RESULT.md`.
