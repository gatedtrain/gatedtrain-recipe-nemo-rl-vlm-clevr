#!/usr/bin/env python
"""Generation half of the area-"VLM" evaluator `vqa.clevr_items@0.1`: one response per (item, evaluation seed).

GPU run inside the NeMo RL image with its vLLM worker venv (vLLM 0.29.0, as area 6's gen_cad.py): vLLM offline
`LLM.generate` with the item's image as `multi_modal_data` and the prompt built exactly as in training (NeMo RL's
`vlm_hf_data_processor`, processors.py:663-762): one user message `[{type: image}, {type: text, text:
PROMPT.format(question)}]` through the checkpoint's own processor chat template with `add_generation_prompt=True`
and no template kwargs (so Qwen3.5 runs non-thinking: the template appends an empty think block). Writes text only;
`score_vqa.py` scores on CPU.

Deployment sampler (05 section 3.3, the "SAMPLER" block, also written into every cut by the trainer hooks): greedy,
one sample per (item, seed), at most 512 new tokens, bf16.

Outputs (`--out`): `gen/<item>__s<seed>.json` `{item, seed, repeat, prompt_sha256, text, n_new_tokens,
finish_reason}`, `gen_meta.json` (model dir, file hashes of the model's deployable files, sampler, prompt block hash,
backend versions, timing).

Usage (container, vLLM worker venv):
  gen_vqa.py --model /inputs/cut --items .../items-valA --out /out --eval-seeds 11 [--limit 20]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPT_FILE = HERE / "prompt_clevr_cot.txt"

# ---- SAMPLER (begin) -------------------------------------------------------------------------------------------
SAMPLER = {
    "strategy": "greedy",
    "temperature": 0.0,
    "top_p": 1.0,
    "max_new_tokens": 512,
    "samples_per_seed": 1,
    "chat_template": "checkpoint processor chat template, add_generation_prompt=True, no template kwargs "
                     "(Qwen3.5 non-thinking)",
    "messages": "[{role: user, content: [{type: image}, {type: text, text: prompt_clevr_cot.txt.format(question)}]}]",
    "max_model_len": 1024,
}
# ---- SAMPLER (end) ---------------------------------------------------------------------------------------------
DEPLOYABLE_NAMES = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json",
                    "merges.txt", "preprocessor_config.json", "video_preprocessor_config.json",
                    "model.safetensors.index.json")


def sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def prompt_block_sha256() -> str:
    return sha256_file(PROMPT_FILE)


def user_text(question: str) -> str:
    return PROMPT_FILE.read_text(encoding="utf-8").format(question)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--items", type=Path, required=True, help="items-valA (deciding) or items-valB (supporting)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--eval-seeds", default="11")
    p.add_argument("--limit", type=int)
    a = p.parse_args(argv)
    t0 = time.time()
    out = a.out.resolve()
    if (out / "gen").exists() and any((out / "gen").iterdir()):
        raise SystemExit(f"{out}/gen exists and is not empty")
    (out / "gen").mkdir(parents=True, exist_ok=True)
    items = [json.loads(x) for x in open(a.items / "items.jsonl", encoding="utf-8")]
    items = items[:a.limit] if a.limit else items
    seeds = [int(s) for s in a.eval_seeds.split(",")]
    cut = None
    if (a.model / "cut.json").exists():   # a cut: every listed file must match before it is used (MM-20)
        cut = json.loads((a.model / "cut.json").read_text(encoding="utf-8"))
        bad = [f for f, h in cut["files"].items() if sha256_file(a.model / f) != h]
        extra = sorted({f.name for f in a.model.iterdir()} - set(cut["files"]) - {"cut.json"})
        if bad or extra:
            raise SystemExit(f"cut {a.model}: hash mismatch {bad}, unlisted files {extra}")

    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    import vllm

    proc = AutoProcessor.from_pretrained(str(a.model), local_files_only=True)
    requests = []
    for it in items:
        img_path = HERE / it["image"]          # relative to this runtime directory (make_data.py)
        raw = img_path.read_bytes()
        if sha256_bytes(raw) != it["image_sha256"]:
            raise SystemExit(f"{it['item']}: {img_path} does not hash to the item's image_sha256")
        msgs = [{"role": "user", "content": [{"type": "image", "image": str(img_path)},
                                             {"type": "text", "text": user_text(it["question"])}]}]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        requests.append({"prompt": prompt, "multi_modal_data": {"image": Image.open(img_path).convert("RGB")}})
    llm = LLM(model=str(a.model), dtype="bfloat16", seed=0, max_model_len=SAMPLER["max_model_len"],
              gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 1}, mm_processor_cache_gb=0)
    for k, seed in enumerate(seeds):
        sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=SAMPLER["max_new_tokens"], seed=seed)
        outs = llm.generate(requests, sp)
        for it, req, o in zip(items, requests, outs):
            doc = {"item": it["item"], "seed": seed, "repeat": k, "prompt_sha256": sha256_bytes(req["prompt"].encode()),
                   "text": o.outputs[0].text, "n_new_tokens": len(o.outputs[0].token_ids),
                   "finish_reason": o.outputs[0].finish_reason}
            (out / "gen" / f"{it['item']}__s{seed}.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n",
                                                                   encoding="utf-8")
    deploy = sorted(f.name for f in a.model.iterdir()
                    if f.name in DEPLOYABLE_NAMES or f.name.endswith(".safetensors"))
    import transformers
    meta = {"backend": "vllm", "model_dir": str(a.model.resolve()),
            "cut": {"global_step": cut["global_step"], "cut_json_sha256": sha256_file(a.model / "cut.json")}
            if cut else None,
            "model_files": {f: sha256_file(a.model / f) for f in deploy},
            "sampler": SAMPLER, "eval_seeds": seeds, "items": len(items),
            "items_sha256": sha256_file(a.items / "items.jsonl"), "limit": a.limit,
            "prompt_block_sha256": prompt_block_sha256(),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "versions": {"vllm": vllm.__version__, "transformers": transformers.__version__, "dtype": "bfloat16"},
            "wall_seconds": round(time.time() - t0, 1),
            "seconds_per_response": round((time.time() - t0) / max(1, len(items) * len(seeds)), 3)}
    (out / "gen_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("items", "wall_seconds", "seconds_per_response", "versions")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
