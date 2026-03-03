import os
import json
import argparse
import ast
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from PIL import Image


def parse_options_field(
    opt: Any,
) -> Tuple[Optional[Dict[str, str]], Optional[List[str]]]:
    """
    MedVLThinker-Eval stores `options` as a string feature (often a serialized dict),
    sometimes already as a dict. We parse into:
      - options_dict: dict[str,str] or None
      - options_list: list[str] ordered by A,B,C,D... when possible
    """
    if opt is None or opt == "":
        return None, None

    options_dict = None
    if isinstance(opt, dict):
        options_dict = opt
    elif isinstance(opt, str):
        # Try JSON first
        try:
            options_dict = json.loads(opt)
        except Exception:
            # Try python literal (single quotes, etc.)
            try:
                options_dict = ast.literal_eval(opt)
            except Exception:
                return None, None
    else:
        return None, None

    if not isinstance(options_dict, dict):
        return None, None

    keys = list(options_dict.keys())

    def key_rank(k: Any):
        ks = str(k).strip().upper()
        if len(ks) == 1 and "A" <= ks <= "Z":
            return (0, ord(ks))
        return (1, ks)

    ordered_keys = sorted(keys, key=key_rank)
    options_list = [str(options_dict[k]) for k in ordered_keys]
    # Ensure dict values are strings
    options_dict = {str(k): str(v) for k, v in options_dict.items()}
    return options_dict, options_list


def safe_save_image(im: Image.Image, path_png: str) -> str:
    """
    Save PIL image safely:
      - Convert CMYK/P/LA/etc to RGB
      - Try PNG first
      - Fallback to JPEG if PNG fails
    Returns the actual saved path.
    """
    # Robust mode handling
    if im.mode == "P":
        # palette → rgba → rgb
        im = im.convert("RGBA").convert("RGB")
    elif im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")

    # Try PNG
    try:
        im.save(path_png, format="PNG")
        return path_png
    except Exception:
        # Fallback to JPEG (drops alpha)
        path_jpg = os.path.splitext(path_png)[0] + ".jpg"
        if im.mode == "RGBA":
            im = im.convert("RGB")
        im.save(path_jpg, format="JPEG", quality=95)
        return path_jpg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to export (e.g., test/train/validation)",
    )
    ap.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Limit number of exported examples",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    out_jsonl = os.path.join(args.out_dir, "data.jsonl")

    # Download dataset
    ds = load_dataset("UCSC-VLAA/MedVLThinker-Eval", split=args.split)

    n = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for ex in ds:
            if args.max_examples is not None and n >= args.max_examples:
                break

            # images: list[PIL.Image]
            images = ex.get("images", [])
            image_paths: List[str] = []

            dsname = str(ex.get("dataset_name", "UNK")).replace("/", "_")
            idx = ex.get("dataset_index", n)

            for j, im in enumerate(images):
                fn = f"{dsname}_{idx}_{j}.png"
                path_png = os.path.join(img_dir, fn)
                saved_path = safe_save_image(im, path_png)
                image_paths.append(saved_path)

            options_dict, options_list = parse_options_field(ex.get("options"))

            row = {
                "dataset_name": ex.get("dataset_name"),
                "dataset_index": ex.get("dataset_index"),
                "image_paths": image_paths,  # list[str]
                "question": ex.get("question"),
                "options_dict": options_dict,  # dict[str,str] or None
                "options": options_list,  # list[str] or None (ordered A,B,C,...) when possible
                "answer_label": ex.get(
                    "answer_label"
                ),  # e.g., "A" (for MCQ), may be None
                "answer": ex.get("answer"),  # free-form answer text, may be None
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

            if n % 100 == 0:
                print(f"Exported {n} examples...")

    print(f"\nDone.")
    print(f"Wrote {n} examples to: {out_jsonl}")
    print(f"Images saved under: {img_dir}")


if __name__ == "__main__":
    main()
