#!/usr/bin/env python
"""Scoring half of the area-"VLM" evaluator `vqa.clevr_items@0.1` (CPU, standard library only, no model).

Deciding metric `exact_match` (higher is better, binary, [0, 1]) with this script's own answer parser - not NeMo
RL's `exact_answer_alphanumeric_reward` (rewards.py:70-85), so the metric is T3 under R15-1 (05 section 6.1: an
answer parser is a tool) with a construct-alignment disclosure (both reward answer correctness):

1. Drop everything up to the last `</think>` if one occurs.
2. If an `<answer>...</answer>` block occurs, the candidate is the content of the last one (non-greedy) and
   `answer_tagged` = 1; otherwise the candidate is the remaining text and `answer_tagged` = 0.
3. Canonical answer: the last token of the candidate that is an integer (digits), a number word zero..twenty, or
   yes/no (case-insensitive, word boundaries); number words become digits, leading zeros are dropped. No such
   token: no answer (`answer_found` = 0), score 0 (MM-4: the policy's fault).
4. Score 1 when the canonical answer equals the item's canonical reference answer (an integer or yes/no; an item
   whose reference has no canonical form is a `scorer_error`), else 0.

Sub-metrics (finite numbers only, MM-1): `answer_tagged`, `answer_found`, `truncated` (finish reason `length`),
`n_new_tokens`, `is_yes_no` (the reference is yes/no). Errors (evaluator faults, MM-4): `missing_output` (no gen
file for a registered (item, seed)), `scorer_error`.

Outputs (`--out`): `outputs/<item>__r<k>.json`, `scores.jsonl`, `summary.json` (statistics, MM-38 check, answer
distribution), `env-runtime.json`, a byte copy of `gen_meta.json`, and `arm-exact_match/` with MM-1 `items.jsonl`,
MM-2 `eval_meta.json` (`framework` = `{name, version}` of the generation backend) and hard-linked `outputs/`.

Usage (CPU, any Python >= 3.10):
  score_vqa.py --gen DIR --items DIR --out DIR [--eval-seeds 11] [--limit N] [--evaluator-id ID]
               [--model-hash H] [--base-model-hash H] [--item-set-hash H]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVALUATOR_ID = "vqa.clevr_items@0.1"               # items-valA (deciding)
ADAPTER = "nemo_rl_vlm"
SCORER_VERSION = "clevr_exact_v1"
NUMBER_WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                "sixteen seventeen eighteen nineteen twenty").split()
TOKEN_RE = re.compile(r"\b(\d+|" + "|".join(NUMBER_WORDS) + r"|yes|no)\b", re.I)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


def sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(Path(p).read_bytes())


def canonical(doc) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def write_json(path: Path, doc) -> None:
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def canon_token(tok: str) -> str:
    t = tok.lower()
    if t in NUMBER_WORDS:
        return str(NUMBER_WORDS.index(t))
    if t.isdigit():
        return str(int(t))
    return t


def parse_answer(text: str) -> dict:
    """Response text -> {answer (canonical or None), answer_tagged}."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    blocks = ANSWER_RE.findall(text)
    tagged = bool(blocks)
    cand = blocks[-1] if tagged else text
    toks = TOKEN_RE.findall(cand)
    return {"answer": canon_token(toks[-1]) if toks else None, "answer_tagged": int(tagged)}


def canon_reference(ref: str) -> str | None:
    toks = TOKEN_RE.findall(ref)
    if len(toks) != 1 or TOKEN_RE.sub("", ref).strip():
        return None
    return canon_token(toks[0])


def stats(vals) -> dict:
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if not n:
        return {"n": 0}
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else None
    return {"n": n, "mean": mean, "sd": sd, "share_one": sum(v >= 1 for v in vals) / n}


def self_test() -> int:
    cases = [("<answer> 3 </answer>", "3", 1), ("so <answer>Three</answer>", "3", 1), ("<answer>yes</answer>", "yes", 1),
             ("I count 2 cubes, so the answer is 4.", "4", 0), ("<think>2 or 5</think> <answer>05</answer>", "5", 1),
             ("<answer>1</answer> ... <answer>no</answer>", "no", 1), ("nothing here", None, 0),
             ("there are none of them", None, 0), ("<answer></answer> 7", None, 1)]
    bad = [(t, parse_answer(t), want) for t, want, tag in cases
           if parse_answer(t) != {"answer": want, "answer_tagged": tag}]
    refs = [("0", "0"), ("10", "10"), ("yes", "yes"), ("No", "no"), ("blue", None), ("2 3", None)]
    bad += [(r, canon_reference(r), want) for r, want in refs if canon_reference(r) != want]
    print("self-test", "ok" if not bad else f"FAILED {bad}")
    return 1 if bad else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gen", type=Path, help="directory with gen/ and gen_meta.json (gen_vqa.py)")
    p.add_argument("--items", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--eval-seeds", default="11")
    p.add_argument("--limit", type=int)
    p.add_argument("--evaluator-id", default=EVALUATOR_ID)
    p.add_argument("--model-hash")
    p.add_argument("--base-model-hash")
    p.add_argument("--item-set-hash")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.gen and a.items and a.out):
        p.error("--gen, --items and --out are required")
    t0 = time.time()
    out = a.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    (out / "outputs").mkdir(parents=True, exist_ok=True)
    gen_meta = json.loads((a.gen / "gen_meta.json").read_text(encoding="utf-8"))
    items = [json.loads(x) for x in open(a.items / "items.jsonl", encoding="utf-8")]
    seeds = [int(s) for s in a.eval_seeds.split(",")]
    if gen_meta["eval_seeds"] != seeds:
        raise SystemExit(f"gen_meta eval_seeds {gen_meta['eval_seeds']} != --eval-seeds {seeds}")
    if gen_meta["items_sha256"] != sha256_file(a.items / "items.jsonl"):
        raise SystemExit("gen_meta.json names another items.jsonl")
    if a.limit:
        items = items[:a.limit]

    rows = []
    for it in items:
        ref = canon_reference(it["answer"])
        for k, s in enumerate(seeds):
            gp = a.gen / "gen" / f"{it['item']}__s{s}.json"
            doc = json.loads(gp.read_text(encoding="utf-8")) if gp.exists() else None
            row = {"item": it["item"], "repeat": k, "seed": s, "error": None, "score": None, "sub_metrics": None}
            parsed = None
            if doc is None:
                row["error"] = {"class": "missing_output", "message": "no generation for this (item, seed)"}
            elif ref is None:
                row["error"] = {"class": "scorer_error", "message": f"reference {it['answer']!r} has no canonical form"}
            else:
                parsed = parse_answer(doc["text"])
                row["score"] = float(parsed["answer"] == ref)
                row["sub_metrics"] = {"answer_tagged": parsed["answer_tagged"],
                                      "answer_found": int(parsed["answer"] is not None),
                                      "truncated": int(doc["finish_reason"] == "length"),
                                      "n_new_tokens": doc["n_new_tokens"], "is_yes_no": int(ref in ("yes", "no"))}
            rel = f"outputs/{it['item']}__r{k}.json"
            write_json(out / rel, {"item": it["item"], "repeat": k, "seed": s, "question": it["question"],
                                   "reference": it["answer"], "response": doc["text"] if doc else None,
                                   "parsed_answer": parsed["answer"] if parsed else None, "score": row["score"],
                                   "sub_metrics": row["sub_metrics"]})
            row["output_ref"] = {"path": rel, "sha256": sha256_file(out / rel)}
            rows.append(row)
    write_jsonl(out / "scores.jsonl", rows)

    code_sha = {"score_vqa.py": sha256_file(Path(__file__).resolve())}
    scorer_code = sha256_bytes(canonical(code_sha))
    sampler = {**gen_meta["sampler"], "backend": gen_meta["backend"],
               "prompt_block_sha256": gen_meta["prompt_block_sha256"]}
    sampler_hash = sha256_bytes(canonical(sampler))
    v = gen_meta.get("versions") or {}
    framework = {"name": "vllm", "version": v.get("vllm")}
    arm = out / "arm-exact_match"
    arm.mkdir()
    lines = []
    for r in rows:
        link_or_copy(out / r["output_ref"]["path"], arm / r["output_ref"]["path"])
        lines.append({"item": r["item"], "repeat": r["repeat"], "seed": r["seed"], "score": r["score"],
                      "error": r["error"], "sub_metrics": r["sub_metrics"], "output_ref": r["output_ref"]})
    write_jsonl(arm / "items.jsonl", lines)
    write_json(arm / "eval_meta.json", {
        "schema": "ptc_skel.item_eval", "schema_version": "1.0", "evaluator_id": a.evaluator_id,
        "adapter": ADAPTER, "metric": "exact_match", "direction": "higher_is_better", "score_range": [0, 1],
        "checkpoint": {"model_hash": a.model_hash, "base_model_hash": a.base_model_hash},
        "normalizer_sha256": None, "env_version_sha256": None, "item_set_hash": a.item_set_hash,
        "eval_seeds": seeds, "sampler": sampler, "sampler_hash": sampler_hash,
        "scorer": {"name": "clevr_exact_match_own_parser", "version": f"{SCORER_VERSION}; python {sys.version.split()[0]}",
                   "weights_sha256": None, "code_sha256": scorer_code},
        "framework": framework, "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})

    ok = [r for r in rows if not r["error"]]
    st = stats([r["score"] for r in ok])
    mean = st.get("mean")
    answers: dict[str, int] = {}
    for r in ok:
        a_ = json.loads((out / r["output_ref"]["path"]).read_text())["parsed_answer"]
        answers[str(a_)] = answers.get(str(a_), 0) + 1
    summary = {"items": len(items), "repeats": len(seeds), "responses": len(rows), "errors": len(rows) - len(ok),
               "error_classes": sorted({r["error"]["class"] for r in rows if r["error"]}),
               "exact_match": st,
               "answer_tagged_share": sum(r["sub_metrics"]["answer_tagged"] for r in ok) / len(ok) if ok else None,
               "answer_found_share": sum(r["sub_metrics"]["answer_found"] for r in ok) / len(ok) if ok else None,
               "truncated_share": sum(r["sub_metrics"]["truncated"] for r in ok) / len(ok) if ok else None,
               "mean_new_tokens": sum(r["sub_metrics"]["n_new_tokens"] for r in ok) / len(ok) if ok else None,
               "parsed_answers": dict(sorted(answers.items(), key=lambda kv: -kv[1])),
               "degeneracy_mm38": {"exact_match": {"pass": mean is not None and 0 < mean < 1, "binary": True}},
               "wall_seconds": round(time.time() - t0, 1), "sampler_hash": sampler_hash,
               "gen_backend": gen_meta["backend"], "model_dir": gen_meta["model_dir"],
               "missing_hashes": [k for k, v in (("model_hash", a.model_hash), ("item_set_hash", a.item_set_hash))
                                  if v is None]}
    write_json(out / "env-runtime.json", {
        "python": sys.version.split()[0], "executable": sys.executable, "code_sha256": code_sha,
        "scorer_code_sha256": scorer_code, "items_sha256": sha256_file(a.items / "items.jsonl"),
        "gen_meta_sha256": sha256_file(a.gen / "gen_meta.json"), "uid": os.getuid(), "gid": os.getgid()})
    write_json(out / "summary.json", summary)
    shutil.copyfile(a.gen / "gen_meta.json", out / "gen_meta.json")
    print(json.dumps({k: summary[k] for k in ("items", "errors", "exact_match", "answer_tagged_share",
                                              "answer_found_share", "truncated_share", "degeneracy_mm38")},
                     indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
