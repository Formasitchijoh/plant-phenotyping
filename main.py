import os

# Must be set BEFORE huggingface_hub / datasets are imported anywhere below.
# Works around a known bug in Hugging Face's newer "Xet" transfer backend
# (RuntimeError: "CAS Client Error... error decoding response body" / similar
# xet_get failures, reported on macOS among other platforms) by forcing the
# older, more reliable HTTP/LFS download path instead.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import load_dataset
import argparse
import json
from pathlib import Path


import yaml

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
    """
    Print how many annotations (What we mean here is how many class in each label for a row) exits per class, accross all splits
    It does not touch the image so this will be fast 
    Remeber each row in the dataset has a label and each label has the classes that are found in the image eg 
    And image might have 3 muchroom (meaning 1 class), an image might have a muchroom and a billberry (meaning 2 classes)
    So in essence label contains the annotations for the image and we are counting how many annotations exist per class across all splits
    """

    counts = { class_id: 0 for class_id in FULL_CLASSES }
    total_images = 0

    # Loop through the train and validations keys
    for split_name in ds.keys():
        # For each train and validation split, loop through the rows in the dataset
        for row in ds[split_name]:
            total_images += 1
            # For each row, get the labels and count how many distinct classes are in that label
            """
            Label structure
            [
                {
                    "class": "muchroom"
                    "x": 11
                    "y": 22
                }
            ]
            """
            for box in json.loads(row["labels"]):
                counts[box["class"]] += 1
    
    total_boxes = sum(counts.values())
    print(f"Total images accross all split: {total_images}")
    print(f"Toral annotated boxex: {total_boxes}")
    print(f"{'id':>3} {'class':22} {'count':>7} {'% of boxes':>10}")

    for class_id, name in FULL_CLASSES.items():
        n = counts[class_id]
        percentage_count = (n / total_boxes * 100) if total_boxes else 0.0
        print(f"{class_id:3d} {name:22} {n:7d} {percentage_count:9.1f}%")


def build_class_map(scope):
    """
    Returns (mapping, names): mapping is {original_class_id: new_0_indexed_id}
    for classes kept in this scope; names is the ordered class_name list for data.yaml
    """

    if scope == "lingonberry":
        # Merge ripe (6) and unripe (7) into a single class
        return {6: 0, 7: 0}, ["lingonberry"]
    
    if scope == "lingonberry-ripeness":
         # Keep ripe and unripe as two distinct classes
        return {6: 0, 7: 1},["lingonberry", "lingonberry-unripe"]
    
    if scope == "multiclass":
        #All five berry species, ripe/unripe separate, muchroon dropped.
        mapping = {i: i for i in range(10)}
        names = {FULL_CLASSES[i]: i for i in range(10)}
        return mapping, names
    if scope == "all":
        mapping = {i: i for i in range(11)}
        names = {FULL_CLASSES[i]: i for i in range(11)}
        return mapping, names
    
    raise ValueError(f"Unknown scope: {scope} ")

def convert_rows(rows_with_indices, out_split_name, out_dir, class_map, keep_empty):
    # Here what we are doing is converting the rows in the dataset into a format that is compatible with YOLOv8.
    # We are also filtering out the classes that we do not want to use in the dataset
    # The results is written into adirectory structure that is compatible with YOLOv8.
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
            for b in boxes if b['class'] in class_map
        ]

        if not kept_lines and not keep_empty:
            continue # skip images with nothing relevent to this scope

        fname = f"{out_split_name}_{i:06d}"
        row["image"].convert("RGB").save(img_dir / f"{fname}.jpg", quality=95)
        (lbl_dir / f"{fname}.txt").write_text("\n".join(kept_lines))
        written += 1

    print(f" {out_split_name}: wrote {written} / {total} images")

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

    # Determining the cmd argument that were passed to the script
    class_map, names = build_class_map(args.scope)
    out_dir = Path(args.out)
    print(f"\nConverting with scope='{args.scope}' -> classes {names}")

    hf_test_split = "validation" if "validation" in ds else ("val" if "val" in ds else None)

    if hf_test_split is None:
        raise ValueError("No 'validation or 'val' split found in the dataset. Please check the dataset structure.")

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
        "names": {i: n for i , n in enumerate(names)}
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
