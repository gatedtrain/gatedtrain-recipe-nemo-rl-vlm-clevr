"""NeMo RL pieces of area "VLM" that log observed consumption (MM-27: wrappers around framework objects; no NeMo RL
source is edited). Imported by run_vlm_nemo.py in the driver and by Ray in the environment actor (NeMo RL image,
commit 4f39fd86289a2ffd5aaf5d202702f7fd7185c924).

1. `clevr_rowid_processor` (registered as `clevr_rowid` by run_vlm_nemo.py, processors.py:1145): turns the
   training row's assistant content list back into the answer string (make_data.py writes both contents as lists so
   the JSON loader infers one Arrow type), drops the null keys the loader adds to content parts, calls NeMo RL's own
   `vlm_hf_data_processor` (processors.py:617) and adds `row_id` to `extra_env_info`, which NeMo RL hands to the
   environment as that sample's metadata.
2. `LoggedVLMEnvironment`: NeMo RL's `VLMEnvironment` (vlm_environment.py:131; its reward code unchanged) whose
   `step` also appends one fsynced line per sample to `env.vlm.log_path` (`env_calls.jsonl`):
   `{call, position, row_id, reward, response_sha256, utc}`. run_vlm_nemo.py points `ENV_REGISTRY["vlm"]` at it
   (every VLM env name is created as the registered `vlm` env, data/utils.py:201, and `register_env` refuses to
   replace an existing name, environments/utils.py:140-144). `reward` is the value `step` returns, before
   `global_post_process_and_metrics` zeroes the reward of samples that did not end (vlm_environment.py:229-231).
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone


def _sha256_text(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()


def _clean_parts(content):
    if isinstance(content, list):
        return [{k: v for k, v in part.items() if v is not None} for part in content]
    return content


def clevr_rowid_processor(datum_dict, task_data_spec, processor, max_seq_length, idx):
    from nemo_rl.data.processors import vlm_hf_data_processor

    user, assistant = datum_dict["messages"]
    answer = assistant["content"]
    if isinstance(answer, list):
        if len(answer) != 1 or answer[0].get("type") != "text":
            raise ValueError(f"row {datum_dict.get('row_id')}: assistant content {answer!r} is not one text part")
        answer = answer[0]["text"]
    datum = dict(datum_dict)
    datum["messages"] = [{"role": "user", "content": _clean_parts(user["content"])},
                         {"role": "assistant", "content": answer}]
    out = vlm_hf_data_processor(datum, task_data_spec, processor, max_seq_length, idx)
    out["extra_env_info"]["row_id"] = datum_dict["row_id"]
    return out


try:  # the Ray actor exists only inside the NeMo RL image
    import ray
    from nemo_rl.environments.vlm_environment import VLMEnvironment as _RemoteVLMEnvironment

    _Base = _RemoteVLMEnvironment.__ray_actor_class__

    @ray.remote(max_restarts=0, max_task_retries=0)  # pragma: no cover
    class LoggedVLMEnvironment(_Base):
        def __init__(self, cfg):
            cfg = dict(cfg)
            self.log_path = cfg.pop("log_path")
            self.calls = 0
            super().__init__(cfg)

        def step(self, message_log_batch, metadata, return_extracted_answer=False):
            ret = super().step(message_log_batch, metadata, return_extracted_answer)
            self.calls += 1
            utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            rewards = ret.rewards.tolist()
            with open(self.log_path, "a", encoding="utf-8") as f:
                for i, (conv, md, rw) in enumerate(zip(message_log_batch, metadata, rewards)):
                    resp = "".join(str(m["content"]) for m in conv if m["role"] == "assistant")
                    f.write(json.dumps({"call": self.calls, "position": i, "row_id": md["row_id"], "reward": rw,
                                        "response_sha256": _sha256_text(resp), "utc": utc}, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
            return ret
except ImportError:  # host: no ray / nemo_rl
    LoggedVLMEnvironment = None
