Prototype result of a single registered run; not a benchmark claim; not affiliated with or endorsed by the owners of the models, datasets, or frameworks used.

# Result (2026-11-14): gate PASS

NeMo RL 4f39fd8 (derived image `ptc-a6-nemo-rl:1`; ptc, ptc-skel and ptc-hub are earlier names of gatedtrain, see "Rename to
gatedtrain 0.3.0"; uid 1000, `--network none`), synchronous VLM GRPO of
Qwen/Qwen3.5-0.8B @ `2fc06364` (full weights, vision tower frozen) on CLEVR-CoGenT TrainA: 60 steps x 16 prompts x 8
generations (7,680 samples, rows 0-959 in file order), cuts 20/40/60, learning rate 2e-6 with a 10-step warm-up, KL 0;
0.40 GPU-hours for the full run (peak GPU memory 19.7 GB, host 28.8 GiB). The training reward rose from 0.04 to about
0.8. The gate compares the last cut (60) with the base model.

| | |
|---|---|
| Deciding metric | `exact_match` on 500 ValA items (own standard-library answer parser `score_vqa.py`, not the reward's; T3 with the `construct_alignment` and `shared_annotation` disclosures) |
| Baseline (base model) | 0.404 |
| Candidate (cut 60) | 0.730 |
| Delta, 95% cluster-bootstrap interval | +0.326 [0.270, 0.382] |
| p-value | <= 1.0e-04 |
| Items, samples per item | 500, 1 (greedy, eval seed 11) |
| Achieved power | 0.9995 (planning effect 0.15, sigma_plan 1.14 from the smoke) |
| Exact overlap with training | 0 (key: image SHA-256 and question) |
| Consumption precision | `row` (observed: the logging environment's per-sample log joined with NeMo RL's step logs, all 7,680 samples) |

The candidate puts 100 % of its answers in `<answer>` tags (base 0.6 %); truncation fell from 4.6 % to 0. Supporting
ValB (condition B, not gating): 0.392 to 0.674. The impact query returns cuts 20/40/60 for training shard 0, cuts
40/60 (first step 32) for shard 1, and no checkpoint for shard 2.

Re-collected on 2026-11-16 with this repository's adapter (installed as a plugin of ptc) on the original run's
outputs: every field of the gate report equals the pre-migration re-collect except the run-id-derived
`lockfile_hash`; the 7,680 ledger rows and the lineage are equal.

## Publication cleanup (2026-11-16)

Before publication, host paths and internal process notes were removed from the repository. The result above was
re-collected unchanged before this cleanup.

**Item sets.** The published item sets name each image by its path relative to `runtime/` (`data/images/<sha256>.png`)
instead of an absolute host path; `runtime/make_data.py` now writes them that way and `runtime/gen_vqa.py` resolves
them against its own directory. The rows, their order and every other field are unchanged (`image_sha256` already
identified each image), and `item_set_meta.json` names `images_dir` relatively too. A rebuild with the published
`make_data.py` from the pinned parquet shards reproduces both item sets byte for byte. Because the item-set files
changed, their hashes changed:

| Item set | Manifest hash, 2026-11-14 (absolute paths) | Manifest hash, published (relative paths) |
|---|---|---|
| `items-valA` (deciding) | `sha256:874cd934503022e32660523bdf652ebd6b957c905b46b90da8f88ebc0e00a9f8` | `sha256:95e2c36312fc5586723374c195dc75d996a61e7641ae96b297ae541b8d4ba294` |
| `items-valB` (supporting) | `sha256:367226a8a69f7c18c99624940539b3de46715fb5b2e357201c300f825914d793` | `sha256:f5bc4454566c0ab8573b0c89272da6207cd09424073a7908f0f9c9c3454d9fda` |

The 2026-11-14 gate ran on the absolute-path version of the same items: the registration and spec under
`results/2026-11-14/gate/` keep the original `items-valA` hash. `items.jsonl` itself (the
absolute-path versions are not published): valA `sha256:5e442332eb25cd42bddfaacbcc9926463ea1c343cc24ca7295e6293f1271bc2e`
published; valB `sha256:ca9efac260f968d6a5de4767d0c036a5e740469947c100b7a06a13def3abb366` published. The test fixture
`tests/fixtures/vlm/real-arm/` names the published `items.jsonl` hash (and the re-sealed `gen_meta.json` hash).

The 2026-11-14 manifest hashes `874cd934...` (`items-valA`, the registered item set) and `367226a8...` (`items-valB`)
cover the gate-time item-set files, which held host paths. Those files are not published, and the published copies
have the paths replaced, so neither hash can be recomputed from this repository. What can still be checked: the
published item sets (the same rows, with the manifest hashes in the table above) and the published per-item records
and summary statistics of the real-arm fixture (`tests/fixtures/vlm/real-arm/`).

**Gate files.** `results/2026-11-14/gate/spec.yaml` is a redacted copy (host paths replaced by `<data-root>`); the
gate-time original is not published.

`registration.yaml` and `trainer.yaml` are unchanged, so the hashes the spec pins for them still hold. The registration's header
comment names `decisions/MVP-MULTIMODAL.md`: that is the maintainers' decision record, not published; the comment
stays because the registration is pinned by hash. The redacted spec says "the maintainers' decision record (not
published)" instead.

**Code.** The pinned scorer `runtime/score_vqa.py` is unchanged. `runtime/gen_vqa.py` and `runtime/make_data.py`
changed as described above; `runtime/run_vlm_nemo.py` and `runtime/fetch_models.py` changed only in docstrings. The
run records name the pre-cleanup versions of these scripts.

## Rename to ptc-hub 0.2.0 (2026-11-17)

ptc-hub 0.2.0 renamed the persisted identifiers (schema ids, ledger writer id, analysis code version). The files under
results/2026-11-14/ and the byte-copied fixtures under tests/fixtures/vlm/real-arm/ keep the names ptc-skel 0.1.0
wrote; they are read by ptc `78eb913` (this and the other framework commits cited here, `e20736e` among them, are
unpublished pre-release commits of the framework, earlier named ptc-skel and ptc-hub; the published framework is
`gatedtrain` `v0.3.0`), with which the historical verdict remains reproducible from the original
`registration.yaml`/`spec.yaml` (and the scorer pin before this change). The gate definition is carried forward by
results/2026-11-14/gate/registration-ptc-hub.yaml (amends
sha256:504eadce5145024c0f3e0f8edc2f646d0d10d79d431eb43309ef34ae35e3ea7c, the registration the gate ran with;
post_hoc: false) and spec-ptc-hub.yaml. Re-pinned for the rename: `runtime/score_vqa.py`
`f527e4f145b56beddbadca4f82bc77a6558f86f93c9e3343711db17c66481727` ->
`345e66b07d025e205f317378ea8d405af46ea7da0c0b74daeb57d96ec5a1ef62` (only the new pin is accepted; the arms of the
2026-11-14 run and the real-arm fixture record the old one).

- Scorer (design revision 17, R17-3): the new `score_vqa.py` is exactly ptc's `renamed.apply_name_map` of the gate-time
  file (it writes `"schema": "ptc_hub.item_eval"`, nothing else changed), kept byte for byte as
  `results/2026-11-14/gate/scorer-before/score_vqa.py` (from git). Its composite `code_sha256` moves from
  `sha256:f3eaba291f439827fa93115b6aebed5d18e8c5b0054013c87f3292de2719abaf` to
  `sha256:d5581fbfc1f5116c3515b99d6852bca240c1a32b032569f324087f55c7689744`, so the evaluator `vqa.clevr_items@0.1`
  has a new identity: `manifest_hash` `sha256:199b3978987bc6a36579033317840c73e1a7cae6cc341967f4e575b71558cef9` ->
  `sha256:61f7222e79925161921efb01ecb60c4164fc40fc93953422a2c765f3c094cf3b`, proved by the amendment's `rename_proof`
  (`composite: true`; ptc checks it at validation). The supporting `vqa.clevr_valb_items@0.1` shares the scorer and its
  identity moves the same way; it is not registered. No other registered value changed: the sampler and
  `eval_distribution` (the absolute-path ValA item set `sha256:874cd934...` the gate ran on) are as registered.
- Files: `registration-ptc-hub.yaml` (SHA-256 `898b62172b38abe3734e75eb9858c68d708f5465b1dc468d680f765054e8c059`)
  differs from `registration.yaml` only in `analysis_code_version: ptc-hub 0.2.0`, `amends`, `created_at`
  (2026-11-17T18:29:42Z), the evaluator's `manifest_hash` and its comment, `post_hoc: false`, `rename_proof`, the
  amendment comment, and the header comment's source references (an internal decision-document name replaced by "the
  maintainers' decisions"; comments are not compared); `spec-ptc-hub.yaml` (SHA-256
  `2f6d5042da5989ed22ac3a28e4660f0be6612baeb2f9bd581d61749f0194ed93`) differs from `spec.yaml` only in the registration
  uri, the same header-comment rewording and the name map in a comment. `ptc-hub validate` accepts it (with
  `<data-root>` filled in); the original spec under ptc-hub 0.2.0 is refused with `error[REG-FIELD]
  registration.analysis_code_version` naming the amendment.
- Re-judged on a scratch replay with ptc `e20736e`, on the original items (the absolute-path ValA set the spec pins):
  copies of the run's launch record (`train-full-1.launch.json`), smoke record (`train-smoke-1.smoke.json`) and both
  scoring directories had only their schema ids replaced (`ptc_hub.launch_record`, `ptc_hub.smoke_record`,
  `ptc_hub.item_eval`) and the proved scorer hashes: in each arm's `eval_meta.json` `scorer.code_sha256`
  `f3eaba29...` -> `d5581fbf...`, and in the `env-runtime.json` beside it (which the adapter checks against its pin)
  the same pair in `scorer_code_sha256` and the member `score_vqa.py` `f527e4f1...` -> `345e66b0...`. The launch
  record's label is a checksum of the record and gpu-step.sh logs that label in the host's `gpu-guard.log`, so the
  replay record was moved to a scratch host-work directory holding a copy of the host's `gpu-step.sh` (its `WORK=` line
  changed) and the run's two guard-log lines under the re-hashed label; `host_work`, `gpu_step_sha256`,
  `host_config_sha256` (original input roots) and `label` were recomputed with ptc's own functions. The stage output
  is the original; the runtime override names the original runtime directory. `ext-collect` on `spec-ptc-hub.yaml`
  reported `registration_amendment: version_only`, `registration_amends: sha256:504eadce...`, `post_hoc: false`, and
  against the canonical 2026-11-14 report: status PASS, no status reasons, delta_hat 0.326, the interval, p value
  1.0e-04, achieved power 0.99949, n_items 500, baseline 0.404, candidate 0.730, item-set and sampler hashes equal (65
  report fields equal). The differences are exactly `analysis_code_version`, `created_at`, `run_id`, `schema`,
  `lockfile_hash`, `registration_hash` (`898b6217...`), `registration_amends`, `registration_amendment` (new), and
  `evaluator_hash` / `evaluator_hash_registered` (`199b3978...` -> `61f7222e...`, the pair the rename proof proves).
  The ledger (7,680 rows) is equal except `run_id`, the run-id-derived `sample_id` and `writer_id`
  (`ptc-hub.ledger@0.2.0`); the lineage (3 checkpoints, 4 shards, no orphans) is equal except the run id, and its
  edges except the run id, the registration hash and the schema id.

## Rename to gatedtrain 0.3.0 (2026-11-18)

The framework was renamed again, from ptc-hub 0.2.0 to gatedtrain 0.3.0 (distribution and import `gatedtrain`, schema
ids `gatedtrain.*`, ledger writer id `gatedtrain.ledger@0.3.0`); this repository is now
`gatedtrain-recipe-nemo-rl-vlm-clevr` (import `gatedtrain_recipe_nemo_rl_vlm_clevr`, entry point group
`gatedtrain.adapters`, runtime override `GATEDTRAIN_RECIPE_NEMO_RL_VLM_CLEVR_RUNTIME`). The files under
results/2026-11-14/ keep their bytes (`registration.yaml`/`spec.yaml` as the gate ran them, the `-ptc-hub` files as
the ptc-hub 0.2.0 amendment, `scorer-before/` as the gate-time scorer), and so do the byte-copied fixtures under
tests/fixtures/vlm/. The sections above describe what was done at the time and keep the names of their time (ptc,
ptc-hub, `ptc-a6-nemo-rl:1`).

Kept names: `runtime/ptc_vlm_hooks.py` (the adapter checks a run's `hooks_sha256` against the file of that name beside
the launched script, so a renamed file would refuse the published run) and the runtime's own schema ids `ptc_mm.*`
(`ptc_mm.vlm_cut` is checked in the published cuts; `ptc_mm.vlm_data`, `vlm_items`, `vlm_models`, `vlm_raw` are in
committed metadata whose hashes are registered).

Re-pinned (only the new pin is accepted): `runtime/score_vqa.py`
`sha256:345e66b07d025e205f317378ea8d405af46ea7da0c0b74daeb57d96ec5a1ef62` ->
`sha256:2bb71c3c85df376a87182703f53e91819b54a05e0486634da3030d1b109e1886` (name map only: it writes
`gatedtrain.item_eval`), so its composite `code_sha256` moves from
`sha256:d5581fbfc1f5116c3515b99d6852bca240c1a32b032569f324087f55c7689744` to
`sha256:7e07e8aec57723e83575ed9f8af29d863c2c8fd58c0f242b1ab92a45a3edf63f` and the evaluator `vqa.clevr_items@0.1` gets
the identity `sha256:9cb0766d58f981712a4521b707d974d3f9bf469d450a11a7c4586883c5d38909` (was
`sha256:61f7222e79925161921efb01ecb60c4164fc40fc93953422a2c765f3c094cf3b`; the supporting
`vqa.clevr_valb_items@0.1` moves the same way and is not registered). `runtime/run_vlm_nemo.py` changed in one
comment (each run records its own `script_sha256`; the adapter does not pin it).

- Amendment: results/2026-11-14/gate/registration-gatedtrain.yaml (SHA-256
  `7758f515362bfaa476c34481d1164f78339a897a74bfd32283a42ac5b2b525d0`, amends
  sha256:898b62172b38abe3734e75eb9858c68d708f5465b1dc468d680f765054e8c059, the ptc-hub amendment; post_hoc: false)
  differs from `registration-ptc-hub.yaml` only in `analysis_code_version: gatedtrain 0.3.0`, `amends`, `created_at`
  (2026-11-18T09:49:52Z), the evaluator's `manifest_hash` and its comment, `rename_proof` (now from the ptc-hub-era
  scorer, `code_sha256_before` `d5581fbf...`, file `scorer-ptc-hub/score_vqa.py`, a byte copy from git that hashes to
  `345e66b0...`; the framework's `apply_name_map` of it is today's `score_vqa.py`), and comments (the amendment header
  and the name map); `spec-gatedtrain.yaml` (SHA-256
  `105bdbd382179c7d2173ac8b8a88e3b6e646930aec35893a385ac40eac554173`) differs from `spec-ptc-hub.yaml` only in the
  registration uri, comments and the placeholder `<mm-root>` for the runtime's data root. `gatedtrain validate`
  accepts it (with `<mm-root>` filled in); `spec-ptc-hub.yaml` under gatedtrain 0.3.0 is refused with
  `error[REG-FIELD] registration.analysis_code_version` naming the amendment.
- Re-judged on a scratch replay with gatedtrain 0.3.0 (framework HEAD of 2026-11-18), as for the ptc-hub rename:
  copies of the run's launch record, smoke record and both scoring directories, made from the original records
  (ptc-skel ids), had only their schema ids replaced (`gatedtrain.launch_record`, `gatedtrain.smoke_record`,
  `gatedtrain.item_eval`) and the proved scorer hashes (each arm's `scorer.code_sha256` and the `env-runtime.json`
  `scorer_code_sha256` `f3eaba29...` -> `7e07e8ae...`, its member `score_vqa.py` `f527e4f1...` -> `2bb71c3c...`); the
  launch record moved to a scratch host-work directory (a copy of the host's `gpu-step.sh` with its `WORK=` line
  changed, the run's two guard-log lines under the re-hashed label), with `host_work`, `gpu_step_sha256`,
  `host_config_sha256` and `label` recomputed by the framework's own functions. The stage output is the original; the
  runtime override names the original runtime directory. `ext-collect` on `spec-gatedtrain.yaml` reported
  `registration_amendment: version_only`, `registration_amends: sha256:898b6217...`, `post_hoc: false`, and against
  the canonical 2026-11-14 report: status PASS, no status reasons, delta_hat 0.326, interval [0.270, 0.382], p value
  1.0e-04, achieved power 0.99949, n_items 500, item-set and sampler hashes equal. The differences are exactly
  `analysis_code_version`, `created_at`, `run_id`, `schema`, `lockfile_hash`, `registration_hash` (`7758f515...`),
  `registration_amends`, `registration_amendment` (new), and `evaluator_hash` / `evaluator_hash_registered`
  (`199b3978...` -> `9cb0766d...`, the identity the two rename proofs carry forward). The ledger (7,680 rows) is equal
  except `run_id`, the run-id-derived `sample_id` and `writer_id` (`gatedtrain.ledger@0.3.0`); the lineage (3
  checkpoints, 4 shards, no orphans) is equal except the run id.
