#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate token embeddings for a custom DreamOOD class set."
    )
    parser.add_argument(
        "--metadata_dir",
        required=True,
        help="Directory containing class_order.txt and synset_to_token_alias.json.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to write the .npy embedding array.",
    )
    parser.add_argument(
        "--model_name",
        default="openai/clip-vit-large-patch14",
        help="CLIP text model used by DreamOOD.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metadata_dir = Path(args.metadata_dir).resolve()
    output_path = Path(args.output_path).resolve()
    class_order = [
        line.strip()
        for line in (metadata_dir / "class_order.txt").read_text().splitlines()
        if line.strip()
    ]
    synset_to_alias = json.loads((metadata_dir / "synset_to_token_alias.json").read_text())
    synset_to_text = json.loads((metadata_dir / "synset_to_text.json").read_text())

    tokenizer = CLIPTokenizer.from_pretrained(args.model_name)
    model = CLIPTextModel.from_pretrained(args.model_name)
    token_weight = model.text_model.embeddings.token_embedding.weight.detach().cpu()

    rows = []
    details = []
    for index, synset in enumerate(class_order):
        alias = synset_to_alias.get(synset, synset_to_text[synset])
        tokens = tokenizer(
            [alias],
            truncation=True,
            max_length=77,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"]
        token_id = int(tokens[0][1])
        token_text = tokenizer.convert_ids_to_tokens([token_id])[0]
        rows.append(token_weight[token_id].numpy())
        details.append(
            {
                "index": index,
                "synset": synset,
                "text": synset_to_text[synset],
                "alias": alias,
                "token_id": token_id,
                "token_text": token_text,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.stack(rows).astype(np.float32))
    details_path = output_path.with_suffix(".json")
    details_path.write_text(json.dumps(details, indent=2) + "\n")
    print(f"Saved embeddings: {output_path}")
    print(f"Saved token metadata: {details_path}")


if __name__ == "__main__":
    main()
