"""
prepare_wildbe_dataset.py

Downloads the WildBe-v2 dataset (FBK-TeV/WildBe-v2) from Hugging Face and
converts it from its native Hugging Face `datasets` parquet format into the
plain images/ + labels/ folder structure that Ultralytics YOLO expects.

Run this on your own machine (this dataset is not reachable from every
sandboxed environment) — not inside a restricted cloud shell.

Setup:
    pip install torch ultralytics datasets pillow pyyaml

Step 1 — ALWAYS run this first, before deciding scope:
    python prepare_wildbe_dataset.py --count-only

Step 2 — convert the v1 (single-class) baseline dataset:
    python prepare_wildbe_dataset.py --scope lingonberry --out ./wildbe_yolo_v1

Step 3 (later, Phase 5) — convert the v2 (multi-class) dataset:
    python prepare_wildbe_dataset.py --scope multiclass --out ./wildbe_yolo_v2
"""

import argparse
import json
from pathlib import Path

import yaml
from datasets import load_dataset

# Class map as published on the WildBe-v2 dataset card.
FULL_CLASSES = {
    0: "bilberry",
    1: "bilberry-unripe",
    2: "cloudberry",
    3: "cloudberry-unripe",
    4: "crowberry",
    5: "crowberry-unripe",
    6: "lingonberry",
    7: "lingonberry-unripe",
    8: "bog-bilberry",
    9: "bog-bilberry-unripe",
    10: "mushroom",
}


def count_classes(ds):
    """Print how many annotations exist per class, across all splits.
    Doesn't touch the image data, so it's fast even though the dataset is large.
    """
    counts = {cid: 0 for cid in FULL_CLASSES}
    total_images = 0
    for split_name in ds.keys():
        for row in ds[split_name]:
            total_images += 1
            for box in json.loads(row["labels"]):
                counts[box["class"]] += 1

    total_boxes = sum(counts.values())
    print(f"\nTotal images across all splits: {total_images}")
    print(f"Total annotated boxes: {total_boxes}\n")
    print(f"{'id':>3}  {'class':22}  {'count':>7}  {'% of boxes':>10}")
    for cid, name in FULL_CLASSES.items():
        n = counts[cid]
        pct = (n / total_boxes * 100) if total_boxes else 0.0
        print(f"{cid:3d}  {name:22}  {n:7d}  {pct:9.1f}%")
    return counts


def build_class_map(scope):
    """Returns (mapping, names): mapping is {original_class_id: new_0_indexed_id}
    for classes kept in this scope; names is the ordered class-name list for data.yaml.
    """
    if scope == "lingonberry":
        # Merge ripe (6) and unripe (7) into a single class.
        return {6: 0, 7: 0}, ["lingonberry"]
    if scope == "lingonberry-ripeness":
        # Keep ripe and unripe as two distinct classes.
        return {6: 0, 7: 1}, ["lingonberry", "lingonberry-unripe"]
    if scope == "multiclass":
        # All five berry species, ripe/unripe separate, mushroom dropped.
        mapping = {i: i for i in range(10)}
        names = [FULL_CLASSES[i] for i in range(10)]
        return mapping, names
    if scope == "all":
        mapping = {i: i for i in range(11)}
        names = [FULL_CLASSES[i] for i in range(11)]
        return mapping, names
    raise ValueError(f"Unknown scope: {scope}")


def convert_rows(rows_with_indices, out_split_name, out_dir, class_map, keep_empty):
    """rows_with_indices: iterable of (source_row_index, row). Writes into
    out_dir/images/<out_split_name>/ and out_dir/labels/<out_split_name>/.
    """
    img_dir = out_dir / "images" / out_split_name
    lbl_dir = out_dir / "labels" / out_split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    total = 0
    for i, row in rows_with_indices:
        total += 1
        boxes = json.loads(row["labels"])
        kept_lines = [
            f"{class_map[b['class']]} {b['x']:.6f} {b['y']:.6f} {b['width']:.6f} {b['height']:.6f}"
            for b in boxes
            if b["class"] in class_map
        ]
        if not kept_lines and not keep_empty:
            continue  # skip images with nothing relevant to this scope

        fname = f"{out_split_name}_{i:06d}"
        row["image"].convert("RGB").save(img_dir / f"{fname}.jpg", quality=95)
        (lbl_dir / f"{fname}.txt").write_text("\n".join(kept_lines))
        written += 1

    print(f"  {out_split_name}: wrote {written} / {total} images")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scope",
        choices=["lingonberry", "lingonberry-ripeness", "multiclass", "all"],
        default="lingonberry",
        help="Which classes to keep. See the roadmap's Phase 2/5 for the v1/v2 plan.",
    )
    parser.add_argument(
        "--out", default="./wildbe_yolo", help="Output folder for the converted dataset."
    )
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Drop images with zero boxes in the chosen scope (default: keep them as negatives).",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Just print per-class annotation counts and exit. Run this before deciding scope.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help=(
            "Fraction of the HF 'train' split to carve out as OUR OWN validation set, "
            "used only for monitoring during training (default 0.1 = 10%%). "
            "The HF 'validation'/'val' split is reserved separately as a true held-out "
            "TEST set, untouched until final evaluation — see Phase 4 of the roadmap."
        ),
    )
    args = parser.parse_args()

    print("Loading FBK-TeV/WildBe-v2 from Hugging Face (~7.87 GB on first download)...")
    ds = load_dataset("FBK-TeV/WildBe-v2")
    print(f"Splits found in the raw dataset: {list(ds.keys())}")

    if args.count_only:
        count_classes(ds)
        return

    class_map, names = build_class_map(args.scope)
    out_dir = Path(args.out)
    print(f"\nConverting with scope='{args.scope}' -> classes {names}")

    # WildBe-v2 ships only two splits: 'train' and 'validation' — there is no
    # separate test split. Reusing HF's 'validation' as our final test set is
    # fine (it was never trained on), but if we ALSO used it to watch metrics
    # during training, that would compromise it as a genuinely held-out set.
    # So: carve our own small validation slice OUT of 'train' (for
    # epoch-by-epoch monitoring / early stopping), and reserve all of HF's
    # 'validation' split as a real, untouched-until-the-end TEST set.
    hf_test_split = "validation" if "validation" in ds else ("val" if "val" in ds else None)
    if hf_test_split is None:
        print(
            "WARNING: no 'validation' or 'val' split found — cannot set aside a real "
            "test set automatically. Check `Splits found` above and adjust this script."
        )

    train_rows = ds["train"]
    n = len(train_rows)
    val_every = max(1, round(1 / args.val_frac)) if args.val_frac > 0 else 0

    our_train = [(i, train_rows[i]) for i in range(n) if not (val_every and i % val_every == 0)]
    our_val = [(i, train_rows[i]) for i in range(n) if val_every and i % val_every == 0]

    convert_rows(our_train, "train", out_dir, class_map, keep_empty=not args.drop_empty)
    convert_rows(our_val, "val", out_dir, class_map, keep_empty=not args.drop_empty)
    if hf_test_split:
        test_rows = ds[hf_test_split]
        convert_rows(
            [(i, test_rows[i]) for i in range(len(test_rows))],
            "test",
            out_dir,
            class_map,
            keep_empty=not args.drop_empty,
        )

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(names)},
    }
    if hf_test_split:
        data_yaml["test"] = "images/test"
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"\nWrote {yaml_path}")
    print(
        "\nSplit roles:\n"
        "  images/train -> used for actual gradient updates\n"
        "  images/val   -> carved from HF 'train'; used only to monitor training (Phase 3)\n"
        "  images/test  -> HF's original 'validation' split; never touched until Phase 4 "
        "final evaluation\n"
    )
    print(f"Ready to train, e.g.:\n  yolo detect train data={yaml_path} model=yolov8n.pt epochs=20")
    print(f"Ready to test (Phase 4), e.g.:\n  yolo detect val data={yaml_path} model=<your-trained-model>.pt split=test")


if __name__ == "__main__":
    main()
