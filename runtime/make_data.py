#!/usr/bin/env python
"""Export the pinned CLEVR-CoGenT shards of area "VLM" to local training rows and gate items (host step, CPU).

Inputs: `raw/<split>/data/*.parquet` (fetch_models.py; RAW.json pins repo, revision and SHA-256). Outputs:
- `data/images/<sha256>.png` : every image used, named by the SHA-256 of its PNG bytes (written unchanged).
- `data/shards/shard-0000N.jsonl` (500 rows each) and their concatenation `data/train.jsonl`: training rows from
  TrainA, in a fixed permutation (seed "<seed>:order:train"), because the trainer reads the file in order. Row:
  `{row_id, input, messages, image_sha256, source}`; `messages` is NeMo RL's generic message form
  (`[{role: user, content: [{type: image, image: <abs path>}, {type: text, text: <question>}]}, {role: assistant,
  content: [{type: text, text: <answer>}]}]`, processors.py:663-672; both contents are lists so that the JSON
  loader infers one Arrow type for `content`; the row-id processor of run_vlm_nemo.py turns the assistant content
  back into the answer string, the ground truth handed to the environment). `input` = "<image sha256>\n<question>" is the overlap payload (the question text repeats across
  images, so exact-text overlap is checked on image and question together).
- `items-valA/`, `items-valB/` : 500 gate items each, a seeded sample (seed "<seed>:items:<split>") of that split's
  shard. Row: `{item, text, question, answer, image, image_sha256, source}`; `text` is the same overlap payload;
  `image` is the image's path relative to this runtime directory (`data/images/<sha256>.png`), so the published item
  sets name no host path (gen_vqa.py resolves it against its own directory; `image_sha256` identifies the image).
  `item_set_meta.json` records the source and counts (`images_dir` relative too).
- `data/data_meta.json` : counts, sources, answer distribution, the overlap checks.

Host-specific hash: the training rows name their images by absolute path (NeMo RL loads them from there), so
`data/train.jsonl` and `train_file_sha256` depend on where this runtime lies. The committed `data_meta.json` records
the hash of the runtime the published run trained from (its run_args.json records the same). When a run reproduces
the committed `data_meta.json` byte for byte except `train_file_sha256`, the committed file is kept unchanged and the
local hash is printed instead (`train_file_sha256 here: ...`); any other difference writes fresh metadata.

Disjointness (checked, 0 required): no image SHA-256 and no (image, question) pair of the training rows occurs in
either item set; the two item sets share no image. The answer is the text inside `<answer> ... </answer>` of the
source `solution`, stripped.

Usage (host): <venv with pyarrow>/bin/python -I make_data.py [--seed 20261114]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
DATA = HERE / "data"
ROWS_PER_SHARD = 500
N_TRAIN = 2000
N_ITEMS = 500
ANSWER_RE = re.compile(r"^\s*<answer>\s*(.*?)\s*</answer>\s*$", re.S)


def sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())


def read_split(split: str) -> list[dict]:
    import pyarrow.parquet as pq

    files = sorted((RAW / split / "data").glob("*.parquet"))
    rows = []
    for f in files:
        for i, r in enumerate(pq.read_table(f).to_pylist()):
            m = ANSWER_RE.match(r["solution"])
            if not m:
                raise SystemExit(f"{f.name} row {i}: solution {r['solution']!r} has no <answer> block")
            png = r["image"]["bytes"]
            if not png.startswith(b"\x89PNG\r\n\x1a\n"):
                raise SystemExit(f"{f.name} row {i}: image is not PNG")
            rows.append({"question": r["problem"].strip(), "answer": m.group(1).strip(), "png": png,
                         "image_sha256": sha256_bytes(png),
                         "source": {"file": f"{split}/data/{f.name}", "index": i, "image_path": r["image"]["path"]}})
    return rows


def save_image(row: dict, images: Path) -> Path:
    p = images / (row["image_sha256"].split(":")[1] + ".png")
    if not p.exists():
        p.write_bytes(row["png"])
    return p


def payload(row: dict) -> str:
    return f"{row['image_sha256']}\n{row['question']}"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")


def render(meta: dict) -> bytes:
    return (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode("utf-8")


def recorded_train_sha256(meta: dict, committed: bytes | None) -> str | None:
    """The `train_file_sha256` of the `committed` data_meta.json if `meta` with it renders to exactly its bytes, else
    None."""
    try:
        old = json.loads(committed) if committed is not None else None
    except ValueError:
        return None
    h = old.get("train_file_sha256") if isinstance(old, dict) else None
    if not isinstance(h, str):
        return None
    return h if render({**meta, "train_file_sha256": h}) == committed else None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seed", type=int, default=20261114)
    a = p.parse_args(argv)
    raw_meta = json.loads((RAW / "RAW.json").read_text())
    for split, s in raw_meta["splits"].items():
        for rel, f in s["files"].items():
            if sha256_file(RAW / split / rel) != f["sha256"]:
                raise SystemExit(f"raw/{split}/{rel}: SHA-256 differs from RAW.json")

    try:
        committed = (DATA / "data_meta.json").read_bytes()
    except OSError:
        committed = None
    for d in (DATA, HERE / "items-valA", HERE / "items-valB"):
        if d.exists():
            shutil.rmtree(d)
    images = DATA / "images"
    images.mkdir(parents=True)

    # Items first, so they do not depend on the training-set size.
    items: dict[str, list[dict]] = {}
    for split in ("valA", "valB"):
        rows = read_split(split)
        pick = random.Random(f"{a.seed}:items:{split}").sample(range(len(rows)), N_ITEMS)
        out = []
        for k, i in enumerate(sorted(pick)):
            r = rows[i]
            img = save_image(r, images)
            out.append({"item": f"clevr-{split}-{k:04d}", "text": payload(r), "question": r["question"],
                        "answer": r["answer"], "image": img.relative_to(HERE).as_posix(),
                        "image_sha256": r["image_sha256"],
                        "source": r["source"]})
        items[split] = out

    train_all = read_split("trainA")
    item_imgs = {it["image_sha256"] for s in items.values() for it in s}
    item_pairs = {it["text"] for s in items.values() for it in s}
    kept = [r for r in train_all if r["image_sha256"] not in item_imgs and payload(r) not in item_pairs]
    if len(kept) < N_TRAIN:
        raise SystemExit(f"only {len(kept)} TrainA rows left after the overlap filter, need {N_TRAIN}")
    kept = kept[:N_TRAIN]
    random.Random(f"{a.seed}:order:train").shuffle(kept)
    train = []
    for k, r in enumerate(kept):
        img = save_image(r, images)
        train.append({"row_id": f"clevr-trainA-{k:05d}", "input": payload(r), "image_sha256": r["image_sha256"],
                      "messages": [{"role": "user", "content": [{"type": "image", "image": str(img)},
                                                                {"type": "text", "text": r["question"]}]},
                                   {"role": "assistant", "content": [{"type": "text", "text": r["answer"]}]}],
                      "source": r["source"]})

    shards = DATA / "shards"
    shards.mkdir()
    for s in range(0, len(train), ROWS_PER_SHARD):
        write_jsonl(shards / f"shard-{s // ROWS_PER_SHARD:05d}.jsonl", train[s:s + ROWS_PER_SHARD])
    with open(DATA / "train.jsonl", "wb") as f:
        for sh in sorted(shards.glob("shard-*.jsonl")):
            f.write(sh.read_bytes())

    checks = {
        "train_vs_items_image_overlap": len({r["image_sha256"] for r in train} & item_imgs),
        "train_vs_items_pair_overlap": len({r["input"] for r in train} & item_pairs),
        "valA_vs_valB_image_overlap": len({i["image_sha256"] for i in items["valA"]}
                                          & {i["image_sha256"] for i in items["valB"]}),
        "train_rows_dropped_by_overlap_filter": len(train_all) - len([r for r in train_all if r["image_sha256"]
                                                                       not in item_imgs and payload(r) not in item_pairs]),
    }
    if any(v for k, v in checks.items() if k.endswith("_overlap")):
        raise SystemExit(f"overlap checks failed: {checks}")

    for split, rows in items.items():
        d = HERE / f"items-{split}"
        d.mkdir()
        write_jsonl(d / "items.jsonl", rows)
        meta = {"schema": "ptc_mm.vlm_items", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
                "split": split, "n_items": len(rows), "seed": a.seed, "selection": f"random.Random('{a.seed}:items:{split}').sample",
                "source": raw_meta["splits"][split], "license": raw_meta["license"],
                "answers": dict(Counter(r["answer"] for r in rows).most_common()),
                "images_dir": images.relative_to(HERE).as_posix()}
        (d / "item_set_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    meta = {"schema": "ptc_mm.vlm_data", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
            "seed": a.seed, "counts": {"train": len(train), "items-valA": len(items["valA"]),
                                       "items-valB": len(items["valB"]),
                                       "images": len(list(images.glob("*.png")))},
            "train_source": raw_meta["splits"]["trainA"], "license": raw_meta["license"],
            "train_answers": dict(Counter(r["messages"][1]["content"][0]["text"] for r in train).most_common()),
            "train_file_sha256": sha256_file(DATA / "train.jsonl"), "checks": checks,
            "order": "items sampled first; training rows in source order after the overlap filter, first N_TRAIN, "
                     "permuted once"}
    here = meta["train_file_sha256"]
    recorded = recorded_train_sha256(meta, committed)
    if recorded is not None and recorded != here:
        meta["train_file_sha256"] = recorded
        print(f"data_meta.json reproduced except the host-specific train_file_sha256; kept the committed file "
              f"(recorded {recorded}); train_file_sha256 here: {here}", flush=True)
    (DATA / "data_meta.json").write_bytes(render(meta))
    print(json.dumps(meta["counts"]), json.dumps(checks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
