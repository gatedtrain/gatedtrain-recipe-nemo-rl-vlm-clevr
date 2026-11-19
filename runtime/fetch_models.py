#!/usr/bin/env python
"""Download the base model and the three pinned CLEVR-CoGenT shards of area "VLM" (host step, network on, no GPU).

The model goes to `models/qwen3.5-0.8b/` as plain files (no symbolic links, MM-20); one parquet shard of each of
the three MMInstruction dataset repositories goes to `raw/<split>/` (input of `make_data.py`, never mounted into a
run). huggingface_hub's own download metadata (`<local_dir>/.cache/huggingface/`) is moved out to `hf-meta/<name>/`.
No token is used or read: `token=False` on every call, `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, and `HF_HOME` points to
the fresh `hf-home/` of this directory; a token in the environment is removed from this process's environment
before huggingface_hub is imported (its value is never read). Xet transfer is disabled (plain HTTPS).

`models/MODELS.json` and `raw/RAW.json` record per repository: repo, revision (40-hex commit), license, the files
and their SHA-256 and bytes; `--verify` re-hashes and compares. Run with any venv that has huggingface_hub (the
area-4 venv, read-only).

Licences: the model is Apache-2.0. The MMInstruction repositories declare no licence; their source, CLEVR-CoGenT
v1.0 (Johnson et al., https://cs.stanford.edu/people/jcjohns/clevr/), is released under CC BY 4.0. Both are recorded
in RAW.json and disclosed in the README.

Usage (host):
  python fetch_models.py            # download what is missing, write MODELS.json and RAW.json
  python fetch_models.py --verify   # re-hash every file
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
RAW = HERE / "raw"

# Revisions resolved 2026-11-14 from the Hub API; the model reported `gated: false`, license apache-2.0.
QWEN35_FILES = ["LICENSE", "chat_template.jinja", "config.json", "merges.txt",
                "model.safetensors-00001-of-00001.safetensors", "model.safetensors.index.json",
                "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
                "video_preprocessor_config.json", "vocab.json"]
MODEL_SPECS = {
    "qwen3.5-0.8b": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17", QWEN35_FILES, "apache-2.0"),
}
CLEVR_LICENSE = ("source CLEVR-CoGenT v1.0, CC BY 4.0 (https://cs.stanford.edu/people/jcjohns/clevr/); "
                 "the MMInstruction repackaging declares no licence")
RAW_SPECS = {
    "trainA": ("MMInstruction/Clevr_CoGenT_TrainA_70K_Complex", "d079bf102e8fde42032b229043f809586cbfb765",
               ["data/train-00000-of-00027.parquet"]),
    "valA": ("MMInstruction/Clevr_CoGenT_ValA", "f786093c7bb7b45684d8d5791e3fc8726a427233",
             ["data/train-00000-of-00002.parquet"]),
    "valB": ("MMInstruction/Clevr_CoGenT_ValB", "a844ddba31aae3655be1979935ccb0c43123ec62",
             ["data/train-00000-of-00002.parquet"]),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def plain_files(root: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise SystemExit(f"{p} is a symbolic link; the directory must be plain (MM-20)")
        if p.is_file():
            out[p.relative_to(root).as_posix()] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return out


def _hub_env() -> None:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH"):
        os.environ.pop(name, None)          # removed, never read
    os.environ["HF_HOME"] = str(HERE / "hf-home")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    (HERE / "hf-home").mkdir(parents=True, exist_ok=True)


def _move_meta(dest: Path, name: str) -> None:
    meta = dest / ".cache"
    if meta.exists():
        target = HERE / "hf-meta" / name
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(meta), str(target))


def download() -> int:
    _hub_env()
    from huggingface_hub import hf_hub_download

    record = {"schema": "ptc_mm.vlm_models", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
              "models": {}}
    for name, (repo, revision, files, lic) in MODEL_SPECS.items():
        dest = MODELS / name
        for f in files:
            hf_hub_download(repo, f, revision=revision, local_dir=dest, token=False)
        _move_meta(dest, name)
        got = plain_files(dest)
        if sorted(got) != sorted(files):
            raise SystemExit(f"{dest} holds {sorted(got)}, not {sorted(files)}")
        record["models"][name] = {"repo": repo, "revision": revision, "license": lic, "files": got,
                                  "bytes": sum(f["bytes"] for f in got.values())}
        print(f"{name}: {len(got)} files, {record['models'][name]['bytes'] / 1e6:.1f} MB", flush=True)
    (MODELS / "MODELS.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    raw = {"schema": "ptc_mm.vlm_raw", "schema_version": "1.0", "tags": ["prototype", "unpromotable"],
           "license": CLEVR_LICENSE, "splits": {}}
    for split, (repo, revision, files) in RAW_SPECS.items():
        dest = RAW / split
        for f in files:
            hf_hub_download(repo, f, repo_type="dataset", revision=revision, local_dir=dest, token=False)
        _move_meta(dest, f"raw-{split}")
        got = plain_files(dest)
        if sorted(got) != sorted(files):
            raise SystemExit(f"{dest} holds {sorted(got)}, not {sorted(files)}")
        raw["splits"][split] = {"repo": repo, "revision": revision, "files": got,
                                "bytes": sum(f["bytes"] for f in got.values())}
        print(f"{split}: {len(got)} files, {raw['splits'][split]['bytes'] / 1e6:.1f} MB", flush=True)
    (RAW / "RAW.json").write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def verify() -> int:
    bad = 0
    record = json.loads((MODELS / "MODELS.json").read_text(encoding="utf-8"))
    for name, m in record["models"].items():
        got = plain_files(MODELS / name)
        ok = got == m["files"]
        bad += not ok
        print(f"{name}: {'ok' if ok else 'MISMATCH'} ({len(got)} files)", flush=True)
    raw = json.loads((RAW / "RAW.json").read_text(encoding="utf-8"))
    for split, s in raw["splits"].items():
        got = plain_files(RAW / split)
        ok = got == s["files"]
        bad += not ok
        print(f"{split}: {'ok' if ok else 'MISMATCH'} ({len(got)} files)", flush=True)
    return 1 if bad else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--verify", action="store_true")
    args = p.parse_args(argv)
    return verify() if args.verify else download()


if __name__ == "__main__":
    sys.exit(main())
