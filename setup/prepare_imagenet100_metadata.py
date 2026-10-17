#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare metadata files for running DreamOOD on a custom ImageFolder dataset."
    )
    parser.add_argument(
        "--image_root",
        required=True,
        help="ImageFolder root with train/val/test subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Where to write metadata files. Defaults to <image_root>/metadata.",
    )
    parser.add_argument(
        "--reference_split",
        default="train",
        help="Split used to define class order. Default: train",
    )
    parser.add_argument(
        "--manifest_path",
        default="",
        help="Optional label_manifest.json produced during parquet export.",
    )
    parser.add_argument(
        "--classnames_path",
        default=str(Path(__file__).resolve().parents[1] / "assets" / "imagenet_classnames.txt"),
        help="Fallback synset->text mapping file.",
    )
    return parser.parse_args()


def read_split_classes(split_dir: Path):
    if not split_dir.is_dir():
        return []
    return sorted(
        entry.name
        for entry in split_dir.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )


def load_manifest_mapping(manifest_path: Path):
    if not manifest_path.is_file():
        return {}
    raw = json.loads(manifest_path.read_text())
    mapping = {}
    for value in raw.values():
        synset = value.get("synset")
        text = value.get("text")
        if synset and text:
            mapping[synset] = text
    return mapping


def load_classnames_mapping(classnames_path: Path):
    mapping = {}
    if not classnames_path.is_file():
        return mapping
    for line in classnames_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        synset, text = line.split(" ", 1)
        mapping[synset] = text.strip()
    return mapping


def build_prompt(text: str):
    return f"A high-quality image of the {text}"


def suggest_token_alias(text: str):
    text = text.strip().replace("-", " ")
    return text.split()[0].lower()


def main():
    args = parse_args()
    image_root = Path(args.image_root).resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"Missing image_root: {image_root}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else image_root / "metadata"
    manifest_path = Path(args.manifest_path).resolve() if args.manifest_path else image_root / "label_manifest.json"
    classnames_path = Path(args.classnames_path).resolve()

    split_classes = {}
    for split in ("train", "val", "test"):
        classes = read_split_classes(image_root / split)
        if classes:
            split_classes[split] = classes

    if args.reference_split not in split_classes:
        available = ", ".join(sorted(split_classes)) or "none"
        raise ValueError(
            f"Reference split '{args.reference_split}' not found under {image_root}. Available: {available}"
        )

    class_order = split_classes[args.reference_split]
    reference_set = set(class_order)
    for split, classes in split_classes.items():
        if set(classes) != reference_set:
            missing = sorted(reference_set - set(classes))
            extra = sorted(set(classes) - reference_set)
            raise ValueError(
                f"Split '{split}' has different classes. Missing={missing[:10]}, extra={extra[:10]}"
            )

    synset_to_text = load_classnames_mapping(classnames_path)
    synset_to_text.update(load_manifest_mapping(manifest_path))
    missing_text = [synset for synset in class_order if synset not in synset_to_text]
    if missing_text:
        raise ValueError(
            "Could not resolve class text for synsets: {}".format(missing_text[:10])
        )

    ordered_metadata = []
    prompts = {}
    for index, synset in enumerate(class_order):
        text = synset_to_text[synset]
        prompt = build_prompt(text)
        token_alias = suggest_token_alias(text)
        ordered_metadata.append(
            {
                "index": index,
                "synset": synset,
                "text": text,
                "prompt": prompt,
                "token_alias": token_alias,
                "token_alias_needs_review": " " in text,
            }
        )
        prompts[synset] = prompt

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "class_order.txt").write_text("\n".join(class_order) + "\n")
    (output_dir / "synset_to_text.json").write_text(
        json.dumps({synset: synset_to_text[synset] for synset in class_order}, indent=2) + "\n"
    )
    (output_dir / "prompts.json").write_text(json.dumps(prompts, indent=2) + "\n")
    (output_dir / "class_order_with_text.tsv").write_text(
        "".join(f"{row['index']}\t{row['synset']}\t{row['text']}\n" for row in ordered_metadata)
    )
    (output_dir / "synset_to_token_alias.json").write_text(
        json.dumps({row["synset"]: row["token_alias"] for row in ordered_metadata}, indent=2) + "\n"
    )
    (output_dir / "token_alias_review.tsv").write_text(
        "".join(
            f"{row['index']}\t{row['synset']}\t{row['text']}\t{row['token_alias']}\t"
            f"{'yes' if row['token_alias_needs_review'] else 'no'}\n"
            for row in ordered_metadata
        )
    )
    (output_dir / "metadata_summary.json").write_text(
        json.dumps(
            {
                "image_root": str(image_root),
                "reference_split": args.reference_split,
                "splits_checked": sorted(split_classes),
                "num_classes": len(class_order),
                "manifest_path": str(manifest_path) if manifest_path.exists() else None,
                "classnames_path": str(classnames_path) if classnames_path.exists() else None,
                "classes": ordered_metadata,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Wrote metadata for {len(class_order)} classes to {output_dir}")
    print("Files:")
    for name in (
        "class_order.txt",
        "class_order_with_text.tsv",
        "synset_to_text.json",
        "prompts.json",
        "synset_to_token_alias.json",
        "token_alias_review.tsv",
        "metadata_summary.json",
    ):
        print(f"  - {output_dir / name}")


if __name__ == "__main__":
    main()
