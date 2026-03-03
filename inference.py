import argparse
import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


# -------------------------
# Repro / IO
# -------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str, max_examples: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_examples is not None and len(rows) >= max_examples:
                break
    return rows


# -------------------------
# Prompting
# -------------------------


def build_prompt_with_tags(question: str, options: Optional[List[str]]) -> str:
    """
    Forces the model to output:
      <think> ... </think>
      <answer> ... </answer>

    For MCQ, we *require* that <answer> contains ONLY the option letter.
    """
    q = (question or "").strip()

    if options:
        letters = [chr(ord("A") + i) for i in range(len(options))]
        opt_text = "\n".join(
            [f"{letters[i]}. {options[i]}" for i in range(len(options))]
        )
        return (
            f"{q}\n\nOptions:\n{opt_text}\n"
            "First output the thinking process in <think> </think> tags,"
            "Then output the final answer in <answer> </answer> tags.\n"
            "IMPORTANT: In <answer>, output ONLY the single option letter, for example <answer>A</answer>"
        )

    return (
        f"{q}\n\n"
        "First output the thinking process in <think> </think> tags.\n"
        "Then output the final answer in <answer> </answer> tags.\n"
        "IMPORTANT: In <answer>, output ONLY the short final answer, no extra words."
    )


# -------------------------
# Parsing
# -------------------------


def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"^[\s\W_]+|[\s\W_]+$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def extract_answer_tag(text: str) -> Optional[str]:
    """
    Try strict <answer> ... </answer>, then loose <answer> ... (no closing tag).
    """
    m = re.search(
        r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL
    )
    if m:
        return m.group(1).strip()

    m = re.search(r"<answer>\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        chunk = m.group(1).strip()
        # Keep first non-empty line only (avoid swallowing trailing junk)
        for ln in chunk.splitlines():
            ln = ln.strip()
            if ln:
                return ln
    return None


def parse_mcq_letter(text: str, num_options: int) -> Optional[str]:
    letters = [chr(ord("A") + i) for i in range(num_options)]

    # Prefer content inside <answer>
    tagged = extract_answer_tag(text)
    if tagged:
        m = re.search(r"\b([A-Z])\b", tagged.strip().upper())
        if m:
            cand = m.group(1)
            if cand in letters:
                return cand

    # Next: look for "Answer: X" or "Final: X"
    m = re.search(r"(answer|final)\s*[:\-]?\s*([A-Z])\b", text, flags=re.IGNORECASE)
    if m:
        cand = m.group(2).upper()
        if cand in letters:
            return cand

    # Next: first standalone capital letter token
    m = re.search(r"\b([A-Z])\b", text.strip())
    if m:
        cand = m.group(1).upper()
        if cand in letters:
            return cand

    return None


def parse_freeform(text: str) -> Tuple[str, bool]:
    """
    Returns (pred, had_answer_tag)
    """
    tagged = extract_answer_tag(text)
    if tagged:
        # strip common prefixes inside tag
        tagged = re.sub(
            r"^(answer|final)\s*[:\-]\s*", "", tagged, flags=re.IGNORECASE
        ).strip()
        return normalize_text(tagged), True

    # fallback to last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "", False

    last = lines[-1]
    last = re.sub(r"^(answer|final)\s*[:\-]\s*", "", last, flags=re.IGNORECASE).strip()
    return normalize_text(last), False


# -------------------------
# Model wrapper
# -------------------------


class QwenVLPredictor:
    def __init__(
        self,
        model_name_or_path: str,
        dtype: torch.dtype,
        attn_impl: str,
        device_map: str = "auto",
    ) -> None:
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            attn_implementation=attn_impl,
            device_map=device_map,
        )
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

    @torch.inference_mode()
    def generate_once(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: Optional[int],
        min_pixels: int,
        max_pixels: int,
    ) -> str:
        if seed is not None:
            set_seed(seed)

        for p in image_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Image not found: {p}")

        content = []
        for p in image_paths:
            content.append(
                {
                    "type": "image",
                    "image": p,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                }
            )
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        images, videos = process_vision_info(messages)

        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs.update(
                temperature=temperature,
                top_p=top_p,
            )
            if top_k > 0:
                gen_kwargs["top_k"] = top_k

        gen_ids = self.model.generate(**inputs, **gen_kwargs)

        trimmed = gen_ids[0][len(inputs.input_ids[0]) :]
        out_text = self.processor.decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return out_text


# -------------------------
# pass@k evaluation
# -------------------------


@dataclass
class SampleResult:
    pred: str
    correct: bool
    used_answer_tag: bool


def is_mcq(row: Dict[str, Any]) -> bool:
    return (row.get("options") is not None) and (row.get("answer_label") is not None)


def eval_one_row(
    predictor: QwenVLPredictor,
    row: Dict[str, Any],
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed0: int,
    min_pixels: int,
    max_pixels: int,
) -> Tuple[bool, List[SampleResult], bool]:
    """
    Returns:
      pass_success: bool
      samples: list[SampleResult]
      format_failure: bool (True if <answer> tag could not be found for ALL samples)
    """
    image_paths = row.get("image_paths") or []
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    if not image_paths:
        raise ValueError("Row has no image_paths")

    question = row.get("question", "")
    options = row.get("options")
    prompt = build_prompt_with_tags(question, options)

    samples: List[SampleResult] = []
    pass_success = False
    any_tag = False

    if is_mcq(row):
        gold = str(row["answer_label"]).strip().upper()
        num_options = len(options)

        for i in range(k):
            out = predictor.generate_once(
                image_paths=image_paths,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed0 + i,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            pred = parse_mcq_letter(out, num_options=num_options) or ""
            used_tag = extract_answer_tag(out) is not None
            any_tag = any_tag or used_tag
            correct = pred == gold
            pass_success = pass_success or correct
            samples.append(
                SampleResult(pred=pred, correct=correct, used_answer_tag=used_tag)
            )
    else:
        gold = normalize_text(str(row.get("answer", ""))) if row.get("answer") else ""
        for i in range(k):
            out = predictor.generate_once(
                image_paths=image_paths,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=seed0 + i,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            pred, used_tag = parse_freeform(out)
            any_tag = any_tag or used_tag
            correct = (pred == gold) if gold else False
            pass_success = pass_success or correct
            samples.append(
                SampleResult(pred=pred, correct=correct, used_answer_tag=used_tag)
            )

    # format failure if none of the k samples had a parseable <answer> tag
    format_failure = not any_tag
    return pass_success, samples, format_failure


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", type=str, required=True, help="HF model name or local path"
    )
    ap.add_argument("--data_jsonl", type=str, required=True)
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--attn_impl", type=str, default="flash_attention_2")
    ap.add_argument("--dtype", type=str, choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--min_pixels", type=int, default=1280 * 28 * 28)
    ap.add_argument("--max_pixels", type=int, default=16384 * 28 * 28)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    rows = load_jsonl(args.data_jsonl, max_examples=args.max_examples)
    if not rows:
        raise ValueError("No rows loaded from JSONL.")

    predictor = QwenVLPredictor(
        model_name_or_path=args.model,
        dtype=dtype,
        attn_impl=args.attn_impl,
        device_map="auto",
    )

    # Stats
    overall = {"n": 0, "passk": 0, "uniq": 0, "format_fail": 0}
    by_ds = defaultdict(lambda: {"n": 0, "passk": 0, "uniq": 0, "format_fail": 0})

    for idx, row in enumerate(rows):
        dsname = row.get("dataset_name", "ALL")
        seed0 = args.seed + idx * 1000
        success, samples, fmt_fail = eval_one_row(
            predictor=predictor,
            row=row,
            k=args.k,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed0=seed0,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )

        preds = [s.pred for s in samples if s.pred != ""]
        uniq = len(set(preds))

        overall["n"] += 1
        overall["passk"] += int(success)
        overall["uniq"] += uniq
        overall["format_fail"] += int(fmt_fail)

        by_ds[dsname]["n"] += 1
        by_ds[dsname]["passk"] += int(success)
        by_ds[dsname]["uniq"] += uniq
        by_ds[dsname]["format_fail"] += int(fmt_fail)

        if (idx + 1) % 10 == 0:
            print(
                f"[{idx+1}/{len(rows)}] pass@{args.k}={overall['passk']/overall['n']:.3f} "
                f"format_fail={overall['format_fail']/overall['n']:.3f}"
            )

    print("\n=== OVERALL ===")
    n = overall["n"]
    print(f"N = {n}")
    print(f"pass@{args.k} = {overall['passk']/n:.4f}")
    print(f"avg_unique_answers@{args.k} = {overall['uniq']/n:.2f}")
    print(
        f"format_failure_rate = {overall['format_fail']/n:.4f}  (no usable <answer> tag in any of K samples)"
    )

    print("\n=== BY DATASET ===")
    for dsname in sorted(by_ds.keys()):
        st = by_ds[dsname]
        n = st["n"]
        print(
            f"{dsname:20s}  N={n:5d}  pass@{args.k}={st['passk']/n:.4f}  "
            f"avg_unique@{args.k}={st['uniq']/n:.2f}  fmt_fail={st['format_fail']/n:.4f}"
        )


if __name__ == "__main__":
    main()
