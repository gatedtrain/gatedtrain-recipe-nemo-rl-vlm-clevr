# Third-party notices

This repository is licensed under Apache-2.0 (`LICENSE`). It contains the following third-party material.

## CLEVR-CoGenT v1.0 (questions and answers)

- Where: `runtime/items-valA/items.jsonl` and `runtime/items-valB/items.jsonl` (500 questions and answers each, with
  the SHA-256 of each image; the images themselves are not in this repository), and test fixtures derived from them
  and from TrainA rows (`tests/fixtures/vlm/items-valA-first12.jsonl`, `tests/fixtures/vlm/probe-a6/data/`,
  `tests/fixtures/vlm/real-arm/`).
- Source: CLEVR-CoGenT v1.0, Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence
  Zitnick, Ross Girshick, "CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning",
  CVPR 2017, https://cs.stanford.edu/people/jcjohns/clevr/
- License: Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/
- Obtained through the Hugging Face repackaging by MMInstruction (questions selected, `<answer>` tags added):
  `MMInstruction/Clevr_CoGenT_TrainA_70K_Complex` @ `d079bf102e8fde42032b229043f809586cbfb765`,
  `MMInstruction/Clevr_CoGenT_ValA` @ `f786093c7bb7b45684d8d5791e3fc8726a427233`,
  `MMInstruction/Clevr_CoGenT_ValB` @ `a844ddba31aae3655be1979935ccb0c43123ec62` (files and SHA-256 in
  `runtime/raw/RAW.json`). These repositories declare no license of their own; the material is used under the
  source's CC BY 4.0 license.
- Changes: rows sampled, the answer extracted from the `<answer>` tags, the image referenced by its SHA-256 and a
  path relative to `runtime/` (`runtime/make_data.py`).
- **No endorsement:** the CLEVR authors do not endorse this repository or its results.

## NeMo RL prompt file

- Where: `runtime/prompt_clevr_cot.txt` and `tests/fixtures/vlm/prompt_clevr_cot.txt` (byte copies, SHA-256
  `d88becf2d8bd00d8ec1678786317304b268d729ebad9e519ca8c5f281cb31d8b`).
- Source: NeMo RL, https://github.com/NVIDIA-NeMo/RL, `examples/prompts/clevr_cogent_cot.txt` at commit
  `4f39fd86289a2ffd5aaf5d202702f7fd7185c924`.
- License: Apache-2.0 (the text is the same as this repository's `LICENSE`).

## Qwen3.5-0.8B configuration and tokenizer files

- Where: `tests/fixtures/vlm/probe-a6/cuts/checkpoint-5/` (`chat_template.jinja`, `config.json`, `tokenizer_config.json`,
  `preprocessor_config.json`, `video_preprocessor_config.json`, as written into a trained cut; the weights and
  `tokenizer.json` are stand-ins).
- Source: `Qwen/Qwen3.5-0.8B` @ `2fc06364715b967f1860aea9cf38778875588b17`, Qwen team, Alibaba Cloud.
- License: Apache-2.0 (the model repository's `LICENSE`).
