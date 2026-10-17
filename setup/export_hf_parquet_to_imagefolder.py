#!/usr/bin/env python3
import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict

import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a Hugging Face image parquet dataset to ImageFolder layout."
    )
    parser.add_argument("--dataset_root", required=True, help="Dataset root containing data/ and label2text.json")
    parser.add_argument("--output_root", required=True, help="Output root for ImageFolder data")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        help="Splits to export. Default: train validation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Parquet batch size for streaming export",
    )
    parser.add_argument(
        "--max_examples_per_split",
        type=int,
        default=0,
        help="If > 0, export at most this many examples per split for smoke testing",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=1000,
        help="Print progress every N exported examples per split",
    )
    return parser.parse_args()


def synset_from_path(image_path: str) -> str:
    return Path(image_path).name.split("_", 1)[0]


def looks_like_synset(name: str) -> bool:
    return re.match(r"^n\d{8}$", name) is not None


def build_label_to_synset_mapping(data_dir: Path, num_classes: int) -> Dict[int, str]:
    label_to_synset = {}
    for split in ["train", "validation", "test"]:
        shard_paths = sorted(data_dir.glob("{}-*.parquet".format(split)))
        for shard_path in shard_paths:
            parquet_file = pq.ParquetFile(shard_path)
            for batch in parquet_file.iter_batches(batch_size=1024, columns=["image", "label"]):
                for example in batch.to_pylist():
                    label = int(example["label"])
                    synset = synset_from_path(example["image"]["path"])
                    if not looks_like_synset(synset):
                        continue

                    known_synset = label_to_synset.get(label)
                    if known_synset is None:
                        label_to_synset[label] = synset
                    elif known_synset != synset:
                        raise ValueError(
                            "Label {} mapped to both {} and {} while building label_to_synset."
                            .format(label, known_synset, synset)
                        )

                    if len(label_to_synset) == num_classes:
                        return label_to_synset
    return label_to_synset


def save_example(example: Dict[str, Any], dest_path: Path) -> None:
    image_obj = example["image"]
    image_bytes = image_obj.get("bytes")
    image_path = image_obj.get("path")
    if image_bytes is None or not image_path:
        raise ValueError(f"Missing image payload for {dest_path}")

    dest_path.write_bytes(image_bytes)


def export_split(
    data_dir: Path,
    split: str,
    output_root: Path,
    batch_size: int,
    max_examples_per_split: int,
    progress_every: int,
    label_to_synset: Dict[int, str],
) -> None:
    out_split = "val" if split == "validation" else split
    shard_paths = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not shard_paths:
        raise FileNotFoundError(f"No parquet shards found for split={split} in {data_dir}")

    split_root = output_root / out_split
    split_root.mkdir(parents=True, exist_ok=True)
    class_dirs = {}
    exported = 0
    split_start = time.time()
    split_total = sum(pq.ParquetFile(shard_path).metadata.num_rows for shard_path in shard_paths)
    if max_examples_per_split > 0:
        split_total = min(split_total, max_examples_per_split)
    print(
        f"[{out_split}] start: {len(shard_paths)} shard(s), target {split_total} example(s)",
        flush=True,
    )

    for shard_index, shard_path in enumerate(shard_paths, start=1):
        parquet_file = pq.ParquetFile(shard_path)
        shard_rows = parquet_file.metadata.num_rows
        print(
            f"[{out_split}] shard {shard_index}/{len(shard_paths)}: {shard_path.name} ({shard_rows} rows)",
            flush=True,
        )
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["image", "label"]):
            for example in batch.to_pylist():
                label = int(example["label"])
                image_path = example["image"]["path"]
                synset = synset_from_path(image_path)
                if not looks_like_synset(synset):
                    synset = label_to_synset.get(label)

                known_synset = label_to_synset.get(label)
                if synset is None:
                    raise ValueError(
                        "Could not determine synset for label {} from path {}. "
                        "No label_to_synset mapping was available.".format(label, image_path)
                    )
                if known_synset is None:
                    label_to_synset[label] = synset
                elif known_synset != synset:
                    raise ValueError(
                        f"Label {label} mapped to both {known_synset} and {synset}. "
                        "This parquet dataset is not safe to export as ImageFolder."
                    )

                class_dir = class_dirs.get(label)
                if class_dir is None:
                    class_dir = split_root / synset
                    class_dir.mkdir(parents=True, exist_ok=True)
                    class_dirs[label] = class_dir

                dest_path = class_dir / Path(image_path).name
                save_example(example, dest_path)
                exported += 1
                if progress_every > 0 and (exported % progress_every == 0 or exported == split_total):
                    elapsed = max(time.time() - split_start, 1e-9)
                    rate = exported / elapsed
                    print(
                        f"[{out_split}] exported {exported}/{split_total} examples "
                        f"({rate:.1f} img/s, elapsed {elapsed:.1f}s)",
                        flush=True,
                    )
                if max_examples_per_split > 0 and exported >= max_examples_per_split:
                    elapsed = max(time.time() - split_start, 1e-9)
                    rate = exported / elapsed
                    print(
                        f"[{out_split}] done: exported {exported} example(s) "
                        f"in {elapsed:.1f}s ({rate:.1f} img/s)",
                        flush=True,
                    )
                    return

    elapsed = max(time.time() - split_start, 1e-9)
    rate = exported / elapsed
    print(
        f"[{out_split}] done: exported {exported} example(s) "
        f"in {elapsed:.1f}s ({rate:.1f} img/s)",
        flush=True,
    )


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()

    data_dir = dataset_root / "data"
    label_path = dataset_root / "label2text.json"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Missing data directory: {data_dir}")
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label2text.json: {label_path}")

    label2text = json.loads(label_path.read_text())
    class_names = {int(label): text for label, text in label2text.items()}
    label_to_synset = build_label_to_synset_mapping(data_dir, num_classes=len(class_names))
    print(
        "[mapping] resolved {}/{} label->synset entries".format(
            len(label_to_synset), len(class_names)
        ),
        flush=True,
    )
    if len(label_to_synset) != len(class_names):
        missing_labels = sorted(set(class_names) - set(label_to_synset))
        raise ValueError(
            "Could not resolve synset for labels: {}".format(missing_labels[:10])
        )

    output_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        export_split(
            data_dir=data_dir,
            split=split,
            output_root=output_root,
            batch_size=args.batch_size,
            max_examples_per_split=args.max_examples_per_split,
            progress_every=args.progress_every,
            label_to_synset=label_to_synset,
        )

    manifest = {
        str(label): {
            "synset": label_to_synset.get(label),
            "text": class_names[label],
        }
        for label in sorted(class_names)
    }
    manifest_path = output_root / "label_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
