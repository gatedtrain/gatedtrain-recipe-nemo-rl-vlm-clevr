# Area "VLM": CLEVR-CoGenT visual question answering with NeMo RL VLM GRPO (prototype, unpromotable)

Multimodal prototype, area 7 ("VLM | NeMo RL VLM GRPO | Qwen3.5-0.8B (vision tower frozen), CLEVR"; seventh in
the run order), multimodal spec §6 row 7 (training side and evaluation side), MM-38 (smoke), R15-1 / R16-1 (05 §6.1:255-257). Runtime: the Tier B NeMo RL image
(or area 6's derived `gatedtrain-a6-nemo-rl:1`) for training and generation; scoring is CPU-only Python with no model.

Status (2026-11-14): **done, gate PASS.** Full run `out/area7-vlm/full-1` (60 steps), collected as
`runs/area7-vlm/run-20261114T190322Z-3ac6a159`: exact_match on 500 ValA items 0.404 -> 0.730, delta 0.326
[0.270, 0.382]; ValB (supporting) 0.392 -> 0.674. The sections below up to "Open questions" are the research draft
written before anything ran; where the implementation differs, "Implementation (as run)" at the end is authoritative
(it records the area 7 decisions and result).

Pinned sources read: a NeMo RL clone (commit `4f39fd86289a2ffd5aaf5d202702f7fd7185c924`, version 0.8.0,
`nemo_rl/package_info.py`) and a NeMo Evaluator clone. NeMo RL paths below are relative to the NeMo RL clone.

## Pins (proposal; nothing downloaded yet)

| Item | Value |
|---|---|
| Trainer image | Tier B `nemo-rl:4f39fd8`, `sha256:8cd5b714ea91ec7ea9a976c75b0ae6f6932140cf7a8aea803100de6331a2113a`, or area 6's `gatedtrain-a6-nemo-rl:1` `sha256:7a4078e7…` (vLLM patch targets writable by uid 1000; needed when the trainer runs as 1000, area 6 lesson). vLLM 0.29.0 (`uv.lock:8973-8974`). Policy backend: Automodel (`examples/configs/grpo_math_1B.yaml:176-177`, `automodel_cfg.enabled: true`). |
| Base model | `Qwen/Qwen3.5-0.8B` @ `2fc06364715b967f1860aea9cf38778875588b17` (HF API `https://huggingface.co/api/models/Qwen/Qwen3.5-0.8B`): Apache-2.0, not gated, `pipeline_tag image-text-to-text`, architecture `Qwen3_5ForConditionalGeneration`, `model_type qwen3_5`, 873,438,784 parameters (BF16 873,436,192 + F32 2,592), one weight file `model.safetensors-00001-of-00001.safetensors` 1,746,942,600 bytes plus index, tokenizer, `preprocessor_config.json`, `video_preprocessor_config.json`, `chat_template.jinja`. Not available locally at preparation time (no Qwen VL checkpoint usable by NeMo RL). |
| Data | `MMInstruction/Clevr_CoGenT_TrainA_70K_Complex` @ `d079bf102e8fde42032b229043f809586cbfb765` (70,000 rows, 27 parquet shards, 8.76 GB), `MMInstruction/Clevr_CoGenT_ValA` @ `f786093c7bb7b45684d8d5791e3fc8726a427233` (5,000 rows, 2 shards, 829 MB), `MMInstruction/Clevr_CoGenT_ValB` @ `a844ddba31aae3655be1979935ccb0c43123ec62` (5,000 rows, 2 shards, 822 MB). The HF repos declare **no license** (card metadata only); the source, CLEVR-CoGenT v1.0, is "released under the Creative Commons CC BY 4.0 license" (https://cs.stanford.edu/people/jcjohns/clevr/). The MMInstruction re-packaging (questions selected, `<answer>` tags added) has no stated license: open question 2. |
| Python (scoring, host pilots) | proposal: the shared CPython 3.12.16 (`<python-root>/...`) with the standard library only for the scorer (no model, no third-party parser). |

## Model (question 1)

**Qwen3.5-0.8B exists and its architecture is wired into NeMo RL at this commit, but no recipe or test uses it.**

- Automodel factory maps `qwen3_5` to `NeMoAutoModelForImageTextToText` (`nemo_rl/models/automodel/utils.py:35-48`,
  :39); the same table has `qwen2_5_vl`, `qwen2_vl`, `smolvlm`, `gemma3`, but **not `qwen3_vl`** (so Qwen3-VL-2B would
  fall back to `NeMoAutoModelForCausalLM`, :57, and is not a candidate).
- vLLM side: `Qwen3_5ForConditionalGeneration` is in `TOKENIZER_REQUIRED_ARCHITECTURES`
  (`nemo_rl/models/generation/vllm/worker_utils.py:17-27`), and `run_vlm_grpo.py:118-123` asserts
  `skip_tokenizer_init == False` for VLMs. That vLLM 0.29.0 in the image loads the dense 0.8B model is **unverified**
  (inferred from the Qwen3.5 VLM recipes; vLLM 0.31.0 in the area-1 venv has `model_executor/models/qwen3_5.py`).
- Docs: VLM support lists "Qwen3.5-35B-A3B/397B-A17B, Qwen2.5VL-3B" (`docs/about/model-support.md:27-29`); the Qwen3.5
  page says shipped recipes are long-run validated and "other variants and configurations are runnable but have not
  all been validated" (`docs/guides/models/qwen/qwen3-5.md:26-30`). The only small-VLM GRPO recipe is Qwen2.5-VL-3B on
  CLEVR (`examples/configs/vlm_grpo_3B.yaml:20-21`, `recipes/vlm/vlm_grpo-qwen2.5-vl-3b-instruct-clevr-1n8g-automodel.v1.yaml`).
  Automodel's own VLM SFT example says "change model id to Qwen/Qwen3.5-2B and Qwen/Qwen3.5-0.8B to finetune smaller
  models" (`3rdparty/Automodel-workspace/Automodel/examples/vlm_finetune/qwen3_5/qwen3_5_4b.yaml:16`) - SFT in
  Automodel, not NeMo RL GRPO.
- Architecture (HF `config.json` at the pinned revision): hybrid text model, 24 layers, 18 Gated DeltaNet
  (`linear_attention`) + 6 full-attention, hidden 1024, vocab 248,320, tied embeddings; vision tower 12 layers,
  hidden 768, patch 16, spatial merge 2 (about 90 M parameters, **estimate** from the dimensions). A 480 x 320 CLEVR
  image is about (480/32) x (320/32) = 150 visual tokens (**estimate**). Qwen3.5 needs flash-linear-attention for full
  speed on Automodel (about 2x slower without, `qwen3-5.md:260-271`); the `automodel` extra pulls
  `nemo-automodel[moe]` (`pyproject.toml:145`), which includes it since v0.7.0 (`qwen3-5.md:280-285`). Present in the
  image: **unverified**.
- Thinking mode: the model card says "Qwen3.5-0.8B operates in non-thinking mode by default" (README line 585 at the
  pinned revision); its chat template appends `<think>\n\n</think>\n\n` to the generation prompt unless
  `enable_thinking` is true. NeMo RL's VLM processor calls `processor.apply_chat_template(..., add_generation_prompt=True)`
  without template kwargs (`nemo_rl/data/processors.py:758-762`, :776-782), and `policy.tokenizer.chat_template_kwargs`
  is bound only to the tokenizer's `apply_chat_template` (`nemo_rl/algorithms/utils.py:492-495`), so **VLM GRPO runs
  non-thinking** (inferred, verify at the smoke from the logged prompt).
- Base ability: the card reports CountBench 77.0 (thinking) / 68.6 (non-thinking) for Qwen3.5-0.8B (README table
  near line 436). CLEVR accuracy is **not reported** (**unverified**; estimate 30-60 %).

**Vision tower freezing.** `policy.automodel_cfg.automodel_kwargs.freeze_config.freeze_vision_tower: true`
(`docs/model-quirks.md:51-79`; typed at `nemo_rl/models/policy/__init__.py:153-170`); Automodel applies it at
build time (`3rdparty/.../nemo_automodel/_transformers/infrastructure.py:624-626` ->
`components/utils/model_utils.py:364-384`, which freezes modules named `vision`, `visual`, `image_encoder`). Without
`freeze_config` nothing is frozen (`model-quirks.md:68-73`), and a misspelt key is ignored silently (:74-76), so the
launcher must check after start-up that the vision parameters have `requires_grad False` (proposal: log the count of
trainable parameters at the smoke; Qwen3.5 calls its tower `model.visual` - **unverified** at this transformers
version).

Alternatives if 0.8B fails at the smoke: `Qwen/Qwen2.5-VL-3B-Instruct` (the recipe model; full fine-tune does not fit
24 GB: about 3 B x 16 bytes of fp32 weights, gradients and AdamW state, **estimate**), SmolVLM-256M/500M (`smolvlm`
is in the factory, but no NeMo RL VLM GRPO recipe or vLLM refit test for it - **unverified**), Qwen3.5-2B (same
architecture, about 2.5x the memory; would need LoRA, `grpo_math_1B.yaml:198-212`, whose VLM support is
**unverified**).

## Data (question 2)

**What NeMo RL's built-in path does.** `CLEVRCoGenTDataset` maps `train` / `valA` / `valB` to the three MMInstruction
repos (`nemo_rl/data/datasets/response_datasets/clevr.py:69-80`) and calls `load_dataset(<repo>)["train"]` with **no
revision** (:84), so it would fetch the whole 8.76 GB TrainA set and is not pinned. The formatter puts the image as a
base64 string and the question as text, and strips `<answer>` tags from the solution (:23-59). Not used here.

**Proposal: export pinned shards to a local dataset that `ResponseDataset` reads.**
- `ResponseDataset(data_path=...)` keeps rows in file order and, when a `messages` column is present, only adds
  `task_name` (`response_dataset.py:42-76`); `vlm_hf_data_processor` accepts generic `messages` rows
  (`processors.py:663-664`), formats the text through the prompt file (:683-692), and resolves an image given as a
  local path, `file://` URL, data URI or URL (`nemo_rl/data/multimodal_utils.py:1521-1548`). So a JSONL row
  `{"row_id", "messages": [{"role": "user", "content": [{"type": "image", "image": "<abs path>.png"}, {"type": "text",
  "text": <question>}]}, {"role": "assistant", "content": <answer>}]}` with PNGs in the read-only area mount trains
  without NeMo RL edits. That the HF processor's `apply_chat_template` loads a local path inside the image is
  **unverified** (fallback: data URIs, about 230 KB per row).
- Download (needs approval; above the 50 MB limit of this research step): **one** TrainA shard
  (`data/train-00000-of-00027.parquet`, 333 MB, about 2,590 rows) is enough for 60 steps x 16 prompts (960 rows);
  **one** ValA shard (410 MB, about 2,500 rows) for 500 items; optionally one ValB shard (412 MB) for a transfer
  split. Total 0.75-1.16 GB. Each shard downloaded at the pinned revision through `hf_hub_download`
  (`env -u HF_TOKEN`), hashed, recorded in `data/data_meta.json`.
- Content (first 100 rows of each split via the datasets-server rows API, 2026-11-14): images 480 x 320; questions
  are CLEVR counting, comparison and existence questions; answers are integers 0-10 or yes/no. Answer shares:
  TrainA `1` 18 %, `0` 17 %, `2` 15 %, `no` 13 %; ValA `0` 23 %, `1` 20 %, `no` 12 %, `2` 11 %, `yes` 6 %; ValB
  `1` 23 %, `0` 18 %, `3` 11 %. A constant answer "0" would score about 0.2.
- **Condition A vs B** (CLEVR page): A restricts cube colours to gray / blue / brown / yellow and cylinder colours to
  red / green / purple / cyan; B swaps them. ValA is in-distribution for TrainA, ValB tests compositional transfer.
- Overlap: CLEVR splits use different rendered scenes; the export checks image SHA-256 and exact (image, question)
  pairs between training rows and items (0 required; MM-11 exact-text overlap uses the question text, which repeats
  across images, so the check needs the image hash in the payload: "What gatedtrain needs" item 6).
- Alternative if the licence is not acceptable: re-derive question/answer pairs from the original CLEVR-CoGenT v1.0
  questions JSON (106 MB without images, CC BY 4.0) and download images only for the selected scenes from the
  original 24 GB zip (not partially downloadable: **unverified**), or render scenes with the CLEVR generator
  (Blender; out of scope). Recommendation: use the MMInstruction shards with the CC BY 4.0 attribution to the CLEVR
  authors and record the missing repackaging licence as a disclosed gap.

## Trainer (question 3)

**Entry point.** `examples/run_vlm_grpo.py` (default config `vlm_grpo_3B.yaml`, :63-66): loads the processor with
`get_tokenizer(..., get_processor=True)` (:108-110), builds data and environments with
`setup_response_data(..., is_vlm=True)` (:125-128), then the sync trainer (`select_sync_trainer`, :196-217).
Hydra-style overrides as in area 6 (keys absent from the base need `++`).

**Recipe** `examples/configs/vlm_grpo_3B.yaml` (defaults `grpo_math_1B.yaml`, :3): 8 prompts per step (:13),
`train_micro_batch_size 1`, `logprob_batch_size 4`, `max_total_sequence_length 2048`, dynamic batching on, sequence
packing off (:22-29), lr 5e-7 (:30-32), `max_new_tokens 1024`, vLLM `skip_tokenizer_init false`,
`mm_processor_cache_gb 0` (:33-39), data `clevr-cogent` train / valA with prompt `examples/prompts/clevr_cogent_cot.txt`,
processor `vlm_hf_data_processor`, env `clevr-cogent` (:41-51), reward `format` 0.2 + `exact_alnum` 0.8 (:54-60),
`cluster.gpus_per_node 2` (:82-83, must become 1). The prompt asks for `<think> ... </think> <answer> ... </answer>`
and "a single numerical output ... or yes/no" (`examples/prompts/clevr_cogent_cot.txt:1-5`).

**Environment and reward.** With `is_vlm=True` every env name is created as the registered `vlm` environment
(`nemo_rl/data/utils.py:197-204`), i.e. `VLMEnvironment` (`nemo_rl/environments/utils.py:55-57`), which joins the
assistant turns, calls the configured reward functions per sample in Ray workers and returns reward and
`terminated=1` (`nemo_rl/environments/vlm_environment.py:146-219`); rewards of not properly ended samples are zeroed
(:229-231). `format_reward`: 0.25 for `<think>...</think>`, 0.75 for `<answer>...</answer>`
(`nemo_rl/environments/rewards.py:52-67`); `exact_answer_alphanumeric_reward`: the text inside the first
`<answer>...</answer>` (greedy match), alphanumeric characters only, lower case, equal to the ground truth's
(:70-85). Combined by weight (`vlm_environment.py:66-95`). `register_env` refuses to replace `vlm`
(`environments/utils.py:140-144`).

**Row order.** `StatefulDataLoader(..., shuffle=data_config["shuffle"], drop_last=True)` (`nemo_rl/algorithms/grpo.py:882-893`);
the base sets `data.shuffle: true` (`grpo_math_1B.yaml:617`). With `data.shuffle=false` and a local JSONL the order
is the file order, as in Tier B and area 6.

**Consumption (proposal).** Target `row`, observed, as area 6:
1. Register a processor (`register_processor`, `processors.py:1145`) that calls `vlm_hf_data_processor` and adds
   `extra_env_info["row_id"]` from the row (the stock one passes only `ground_truth` and optional `choices`,
   `processors.py:670-672`).
2. Point the registry entry `ENV_REGISTRY["vlm"]["actor_class_fqn"]` at a thin subclass of `VLMEnvironment` defined
   in this area (`vlm_env_logged.py`, proposal) before calling `main()` (a launcher-side wrapper, MM-27; `register_env`
   cannot be used because `vlm` exists). It calls the stock `step`, then appends one fsynced line per sample
   `{call, position, row_id, reward, response_sha256}` to `env_calls.jsonl`. The reward stays NeMo RL's own code.
3. Join with NeMo RL's `train_data_step<N>.jsonl` (`nemo_rl/algorithms/grpo_sync.py:1428-1460`, rewards in order),
   exactly the area-6 adapter rule (16 rows x 8 per step, rows in file order).
Fallback if the join does not hold at the smoke: MM-22 `iteration` (file order, `num_prompts_per_step` rows per step)
as Tier B - that is a maintainer decision after the smoke (area 6 practice: no automatic fallback). So the spec row's
"`iteration` if the VLM dataset path reads exported shards (unverified)" is now verified as possible (local JSONL
through `ResponseDataset`), and `row` is reachable with the wrapper.

**Memory on 24 GB (all estimates; the smoke decides).** Automodel loads fp32 master parameters unless the optimizer
keeps its own fp32 master (`nemo_rl/models/automodel/setup.py:776-780`); mixed precision computes in bf16 with fp32
reductions (:531-534). About 780 M trainable language-model parameters x 16 bytes (fp32 weight, gradient, AdamW
moments) = about 12.5 GB, plus the frozen tower, plus vLLM's share and the 248,320-wide logits (2,048 packed tokens x
248,320 x 4 bytes = 2.0 GB per microbatch in fp32). Area 6 (0.5 B, vocab 151,936) already peaked at 24.2 GB with vLLM
0.25 and 2,560-token packs. **Expect the 0.8B full fine-tune not to fit with the area-6 settings.** Settings that
exist at this commit:
- `policy.generation.vllm_cfg.gpu_memory_utilization` (`grpo_math_1B.yaml:587`; area 6 used 0.25; 0.15-0.2 proposed,
  the model is 1.7 GB in bf16), `max_model_len` (:588), `enforce_eager` (:592).
- `policy.automodel_cfg.activation_checkpointing` (:186), `cpu_offload` (:184; FSDP `CPUOffloadPolicy`, setup.py:49;
  costs host RAM, already the binding constraint), `policy.logprob_chunk_size` (:168),
  `policy.dynamic_batching.train_mb_tokens` / `logprob_mb_tokens` (:434-439) or sequence packing (:441-447).
- A memory-lean optimizer as in the Qwen3.5 VLM recipe: TE `FusedAdam` with `master_weights: true`,
  `store_param_remainders: true`, bf16 moments
  (`recipes/vlm/vlm_grpo-qwen3.5-35ba3b-geo3k-2n8g-automodel-ep16.yaml:25-33`), which lets Automodel load bf16
  parameters (setup.py:777-778): about 10 bytes per parameter instead of 16 (**estimate**; Transformer Engine in the
  Automodel venv **unverified**).
- `loss_fn.reference_policy_kl_penalty: 0` removes the reference-model logprob pass (area 1 used `beta=0` for
  memory; the Qwen3.5 recipe sets 0.0 too, :4-5).
- LoRA (`automodel_cfg.lora_cfg`, `grpo_math_1B.yaml:198-212`): smallest memory, VLM support **unverified**.
Proposed smoke ladder (first that peaks <= 22.5 GB wins, MM-38): (a) TE FusedAdam bf16 moments + KL 0 + vLLM 0.2 +
activation checkpointing + 2,048-token microbatches; (b) as (a) plus `logprob_chunk_size`; (c) `cpu_offload`
(host-RAM risk); (d) LoRA.

**Sequence budget (proposal).** Non-thinking output plus the prompt's think/answer request: `max_new_tokens 512`,
`max_total_sequence_length 1024` (prompt about 150 visual + 120 text tokens, **estimate**). Truncated samples get
reward 0 (`vlm_environment.py:229-231`).

**Hooks reused from area 6** (copy and adapt, do not import across areas): `run_args.json`, `env-freeze.txt` with the
NeMo RL commit line, `attempts.jsonl`, the atomic cut writer around `publish` with `cut.json` last, tokenizer files
copied (not hard-linked: NeMo RL keeps the download's mtimes), seeds. New for VLM: the cut must carry the processor
files (`preprocessor_config.json`, `video_preprocessor_config.json`, `chat_template.jinja`) from the base, since
which processor files NeMo RL saves under `policy/tokenizer/` for a VLM is **unverified**; and the consolidated weight
file name for a VLM (single `model.safetensors` or sharded with an index) is **unverified**.

## Evaluation (question 4)

**What the spec's "NeMo Evaluator VLM path, `solver.image_detail` registered" means.** In NeMo Evaluator the simple
solver has `image_detail: str = "auto"` (`src/nemo_evaluator/config/solvers.py:27-36`). NeMo Evaluator builds a
`VLMSolver` only when `image_detail != "auto"` (`src/nemo_evaluator/orchestration/orchestrator.py:272-275`);
otherwise it builds `ChatSolver`, which never sends `task.images` (`src/nemo_evaluator/solvers/chat.py:32-37`). So
"auto" silently evaluates without images; that is why the spec says "registered (low or high)". The value is put
into each OpenAI `image_url` block as `detail` (`src/nemo_evaluator/engine/model_client.py:208-228`); local paths are
inlined as data URIs (:55-74). Whether vLLM's OpenAI server honours `detail` for Qwen3.5 is **unverified** (likely
ignored, so "low" and "high" would score the same). There is no CLEVR benchmark in NeMo Evaluator (only a VLMEvalKit
MMBench example, `examples/configs/13_vlmevalkit_mmbench.yaml`); it would need a custom benchmark plus a vLLM server
inside the `--network none` container.

**Recommendation: own per-item evaluator, as areas 4-6** (proposal, `vqa.clevr_items@0.1`):
1. `gen_vqa.py` (GPU, NeMo RL image, the vLLM worker venv as area 6's `gen_cad.py`): vLLM offline `LLM.generate`
   with `multi_modal_data={"image": PIL}` and the same chat template and prompt text as training, bf16, at most 512
   new tokens; one `SamplingParams(seed=…)` per (item, seed). Writes `gen/<item>__s<seed>.json` and `gen_meta.json`.
2. `score_vqa.py` (CPU, standard library only): its **own answer parser**, not NeMo RL's: take the text after the
   last `</think>` if present, prefer the last `<answer>...</answer>` block, else the last number or yes/no token;
   map number words (zero..twenty) to digits; compare the canonical answer with the reference (integer equality or
   yes/no). Writes MM-1 arms with sub-metrics `format_ok`, `answer_found`, `truncated`, `n_new_tokens`,
   `is_count_question`.
Reasons: same pattern as areas 4-6 (one implementation of the arm schema, `--limit` for smoke arms, no server or
network); NeMo Evaluator adds a server and an untested custom benchmark for no gate benefit; and a separate parser is
what keeps the metric T3 (next section).

## Gate design (question 5)

**Deciding metric** `exact_match` (proposal name), higher, [0, 1], `binary: true` with K = 1.

**Tier (R15-1, R16-1).** R15-1 counts an "answer parser" as a tool (05 §6.1:255). Scoring with NeMo RL's
`exact_answer_alphanumeric_reward` itself would be the same tool as the reward (T0, cannot block). With the own parser
above the metric is **T3, construct-aligned** (both reward answer correctness; disclosure `construct_alignment`).
References: CLEVR answers come from the same functional-program executor over the same scene generator as the
training answers; under R16-1 held-out references from the same annotation process leave a programmatic scorer at
T3 with the convention disclosed (`shared_annotation`; the convention here is "count words as digits, yes/no").
Proposal: both disclosures.

**Items and split.** Proposal: 500 items from ValA (condition A, in-distribution; the spec's pick), stratified by
answer type in proportion; optional supporting arm on 500 ValB items (condition B transfer), reported only.

**N, K, power.** With greedy decoding repeats are identical, so the spec's K 4 only makes sense with sampling. Two
options (open question 4): (a) K 1 greedy, eval seed 11 recorded (areas 4-6 practice); (b) K 4 at the GRPO sampling
temperature (1.0) with common seeds. Paired binary difference sd = sqrt(d - Δ²) with discordance rate d; at d 0.25,
sd about 0.49 and N 500 gives MDE about 0.061 (one-sided α 0.025, power 0.8: n = ((1.960 + 0.842) σ / Δ)²,
**estimate**); K 4 lowers the per-item variance (**estimate**: MDE about 0.04).

**MM-38 risk.** For a binary metric the baseline is degenerate only at mean 0 or 1 (MM-38). Base
accuracy **unverified**; CountBench 68.6 non-thinking suggests well inside (0, 1). The real risks are format and
ceiling: if the base rarely emits `<answer>` tags the reward is near 0 but the own parser still extracts the last
number (by design; check `answer_found` share in the smoke arm), and if base accuracy on ValA is above about 0.85
there is little room (choose ValB or harder question types then).

**Consumption precision.** `row` (observed, env log joined with `train_data_step<N>.jsonl`) as target; `iteration`
(MM-22) as the fallback, chosen explicitly if the join fails.

## Profiles and cost (question 6; all estimates)

Basis: area 6 GRPO 60 steps x 16 x 8 on a 0.5 B model, 0.44 GPU-h incl. startup; Tier B 0.6 B about 29 s per step of
256 samples. Here 0.87 B, about 150 extra visual tokens per prompt, up to 512 new tokens, TE optimizer or offload.

| Run | Steps x batch | GPU-h (estimate) |
|---|---|---|
| GRPO smoke (one per memory setting tried) | 5 x (16 x 8) | 0.12-0.15 each; plan 2 attempts = 0.3 |
| GRPO full | 60 x (16 x 8) = 960 prompts, 7,680 samples | 0.7-1.2 (35-60 s per step + about 5 min start) |
| Eval generation, K 1 | 500 items | 0.05 per arm (vLLM start about 2 min) |
| Eval generation, K 4 | 2,000 generations | 0.08-0.1 per arm |
| Smoke arms | 20 items | 0.03 per arm |
| Scoring | | 0 (CPU, seconds) |

Area total about 1.3-2.0 GPU-h (spec estimate 3.0). **Host RAM risk: high.** Area 6 GRPO peaked at 29.1 GiB of host RAM
with heavy swap use; here the model is 1.75x larger and image payloads travel through Ray (Tier B and area 6 use
`RAY_memory_monitor_refresh_ms=0`). `cpu_offload` would add the fp32 optimizer state (about 9-12 GB) to host RAM and
is the last resort. Watch host RAM in the smoke record (MM-38 field).

## Launch request facts (proposal; mirrors area 6)

Trainer: image as above, `--runtime=nvidia --gpus all`, `--shm-size=16g --ulimit memlock=-1 --ulimit
nofile=65535:65535 -w /opt/nemo-rl`, user 1000:1000 (derived image), `--network none` (Ray on loopback works, area
6), env `HF_HUB_OFFLINE=1 RAY_memory_monitor_refresh_ms=0 TORCH_CUDA_ARCH_LIST=8.9 FLASHINFER_CUDA_ARCH_LIST=8.9
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Mounts: `<runtime>` (this directory) ro at its host path (scripts,
model, data, items), `<output-root>/<run>` rw at `/out`. argv
`/opt/nemo_rl_venv/bin/python <runtime>/run_vlm_nemo.py --profile smoke|full --out
/out --seed 7100 --base <runtime>/models/qwen3.5-0.8b` (proposal), which calls `run_vlm_grpo.main()` with
overrides: `cluster.gpus_per_node=1`, `data.shuffle=false`, `++data.train.dataset_name=ResponseDataset`,
`++data.train.data_path=.../data/train.jsonl`, `data.validation=null`, `++data.train.split_validation_size=0`,
`data.default.processor=<registered row-id processor>`, `data.default.env_name=clevr-cogent`,
`policy.model_name=<base>`, `++policy.automodel_cfg.automodel_kwargs.freeze_config.freeze_vision_tower=true`
(plus `freeze_audio_tower=true`, `freeze_language_model=false`), `grpo.num_prompts_per_step=16`,
`grpo.num_generations_per_prompt=8`, `policy.train_global_batch_size=128`, `grpo.val_period=0` or validation off,
checkpointing `save_consolidated=every`, `keep_top_k=null`, logger tensorboard and wandb off, the memory ladder
settings. Generation arm: the vLLM worker venv interpreter as area 6, mounts as the trainer plus the cut at
`/inputs/cut` ro. Scoring: CPU, `gpus: false`, `--network none`.

## What gatedtrain needs (adapter work; not done here, `gatedtrain/ext/` untouched)

The registry row `nemo_rl_ext` is area-6 specific (`registry.py:448-520`: `AREA6` roots, CAD outputs, executor
sidecar, `uses` SFT then env GRPO; `adapters/nemo_rl_ext.py` stages `sft`/`grpo`, CAD families). Area 7 needs, as an
extension or a sibling row `rl.nemo_rl_vlm@0.1`:
1. Stage `vlm_grpo`, single stage, base = the Qwen3.5-0.8B directory (MM-21 `base_ref.json` with repo and revision).
2. `model_hash` components: weights (file name(s) as NeMo RL consolidates a VLM, **unverified**), `config.json`,
   `generation_config.json`, tokenizer files, `chat_template.jinja`, `preprocessor_config.json`,
   `video_preprocessor_config.json`, `sampler.json`.
3. Input roots, env names, image interpreters for `run_vlm_nemo.py` and `gen_vqa.py`; no sidecar.
4. Consumption: `env_calls.jsonl` (row-id logged by the wrapper env) joined with `train_data_step<N>.jsonl`, as the
   area-6 GRPO plan; data snapshot = `data/shards/`.
5. Evaluator rows `vqa.clevr_items@0.1` (deciding, binary, T3, disclosures `construct_alignment` and
   `shared_annotation`), optional `vqa.clevr_valb_items@0.1` (supporting).
6. Exact-overlap check: the question text repeats across images, so the payload text field should be
   `<image sha256>\n<question>` (proposal) on both the items and the training rows, or the check must also compare
   image hashes.
7. `mount_binding_problems` (area 6 open item: `ext-launch --spec` raises AttributeError for `nemo_rl_ext`).

## Open questions (at preparation time; "Implementation (as run)" below records how they were settled)

Marked **[choice]** where a design or option choice is involved, **[routine]** otherwise.

1. **[choice] Model.** (a) Qwen3.5-0.8B full fine-tune with the memory ladder (recommended: it is the decision's model,
   Apache-2.0, ungated, wired in NeMo RL at this commit); (b) Qwen3.5-0.8B with LoRA from the start (lower memory,
   VLM LoRA unverified); (c) SmolVLM-500M (smaller, no NeMo RL VLM recipe); (d) Qwen2.5-VL-3B with LoRA (the only
   recipe model, slower, more memory).
2. **[choice] Data licence and download.** The MMInstruction repos declare no licence; the CLEVR source is CC BY 4.0.
   (a) use one TrainA shard + one ValA shard (+ one ValB) at pinned revisions, about 0.75-1.16 GB, with CLEVR
   attribution and the gap disclosed (recommended); (b) rebuild from the original CLEVR-CoGenT v1.0 (24 GB zip);
   (c) another dataset.
3. **[choice] Evaluator path.** (a) own `gen_vqa.py` + `score_vqa.py` with an independent parser (recommended: T3,
   same pattern as areas 4-6); (b) NeMo Evaluator VLM path with `image_detail: high` and a custom benchmark (needs a
   vLLM server in the container; its scorer must still differ from the reward's parser).
4. **[choice] K and decoding.** (a) K 1 greedy (recommended: simplest, areas 4-6); (b) K 4 sampled (spec's K; more
   power, sampling noise in the baseline).
5. **[choice] Held-out split.** (a) ValA 500 deciding, ValB 500 supporting (recommended); (b) ValA only; (c) ValB
   deciding (transfer; smaller expected effect).
6. **[choice] Tier disclosures.** (a) T3 with `construct_alignment` and `shared_annotation` (recommended); (b) T3 with
   `construct_alignment` only.
7. **[routine] Memory ladder at the smoke** (above); KL 0; vLLM 0.2.
8. **[routine] Consumption join** verified at the smoke; falling back to `iteration` is an explicit choice.
9. **[routine] Prompt.** Keep `clevr_cogent_cot.txt` (think + answer tags) unless the smoke shows the non-thinking
   generation prompt conflicts with it (format reward near 0); a change is before registration and recorded.

## Implementation (as run)

Files: `fetch_models.py` (model and three pinned parquet shards, `models/MODELS.json`, `raw/RAW.json`), `make_data.py`
(`data/shards/` 4 x 500 TrainA rows and `data/train.jsonl`, `data/images/<sha256>.png`, `items-valA/`, `items-valB/`
500 each, whose `image` is relative to this directory; overlap checks 0), `run_vlm_nemo.py` (launcher), `ptc_vlm_hooks.py` (cuts, run record), `vlm_env_logged.py`
(row-id processor `clevr_rowid`, logging `VLMEnvironment` subclass registered as `vlm`), `prompt_clevr_cot.txt` (NeMo
RL's `clevr_cogent_cot.txt`, byte copy), `gen_vqa.py`, `score_vqa.py`, `gate/` (spec, registration, trainer config).

Trainer settings that differ from the draft, each found by a GPU probe (`records/area7-vlm/probe-*`, `refit-diag-1`):
- vLLM `max_num_seqs` 128: Qwen3.5's linear-attention layers need one Mamba cache block per sequence (probe-a1).
- Automodel backend `linear=torch`, `attn=sdpa`, `rope_fusion=false`: with the Transformer Engine defaults the state
  dict holds `_extra_state` keys that the Qwen3.5 dense adapter's `convert_single_tensor_to_hf` passes on (it ignores
  NeMo RL's `exclude_key_regex`), and the refit into vLLM failed (probe-a2; vLLM itself reloads the checkpoint).
- `data_plane.enabled=false` (NeMo RL's classic `grpo_train`) with KL 0 and
  `grpo.skip_reference_policy_logprobs_calculation`: the data-plane trainer fetches reference logprobs even when the
  pass is skipped (probe-a3, a5). The classic trainer writes the same `train_data_step<N>.jsonl`.
- Training and logprob micro batch 1 (one sequence of at most 1,024 tokens): the 248,320-wide fp32 logits held the
  policy worker at 22-23 GB otherwise (probe-a6/a7/b2/b3); `expandable_segments` breaks the CUDA IPC refit under
  `--cap-drop ALL` (probe-b4), allocator GC settings made it worse (probe-b5). Peak with micro batch 1: 17.7-19.7 GB.
  Memory rung `a` (TE FusedAdam, bf16 moments, activation checkpointing, vLLM 0.2) is what ran; rungs b/c are unused.
- lr 2e-6 with a 10-step linear warm-up (the base 50-step warm-up would keep most of a 60-step run under 1e-6).

Observed: the base model answers without `<answer>` tags (0.6 % tagged on ValA), so NeMo RL's `exact_alnum` reward was
near 0 at the start (step-1 mean 0.038); the run learned the tags within about 10 steps and the training reward rose to
about 0.8. Consumption: `env_calls.jsonl` joins NeMo RL's step logs for all 7,680 samples (`row`, observed).
