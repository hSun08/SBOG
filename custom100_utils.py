import json
from pathlib import Path


LEGACY_IN100_CLASS_NAMES = [
    "stingray", "hen", "magpie", "kite", "vulture",
    "agama", "tick", "quail", "hummingbird", "koala",
    "jellyfish", "snail", "crawfish", "flamingo", "orca",
    "chihuahua", "coyote", "tabby", "leopard", "lion",
    "tiger", "ladybug", "fly", "ant", "grasshopper",
    "monarch", "starfish", "hare", "hamster", "beaver",
    "zebra", "pig", "ox", "impala", "mink",
    "otter", "gorilla", "panda", "sturgeon", "accordion",
    "carrier", "ambulance", "apron", "backpack", "balloon",
    "banjo", "barn", "baseball", "basketball", "beacon",
    "binder", "broom", "candle", "castle", "chain",
    "chest", "church", "cinema", "cradle", "dam",
    "desk", "dome", "drum", "envelope", "forklift",
    "fountain", "gown", "hammer", "jean", "jeep",
    "knot", "laptop", "mower", "library", "lipstick",
    "mask", "maze", "microphone", "microwave", "missile",
    "nail", "perfume", "pillow", "printer", "purse",
    "rifle", "sandal", "screw", "stage", "stove",
    "swing", "television", "tractor", "tripod", "umbrella",
    "violin", "whistle", "wreck", "broccoli", "strawberry",
]

LEGACY_C100_CLASS_NAMES = [
    "apples", "aquarium fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottles", "bowls", "boy", "bridge", "bus", "butterfly", "camel",
    "cans", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", "cloud",
    "cockroach", "couch", "crab", "crocodile", "cups", "dinosaur", "dolphin", "elephant",
    "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "computer keyboard",
    "lamp", "lawn-mower", "leopard", "lion", "lizard", "lobster", "man", "maple",
    "motorcycle", "mountain", "mouse", "mushrooms", "oak", "oranges", "orchids", "otter",
    "palm", "pears", "pickup truck", "pine", "plain", "plates", "poppies", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "roses", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflowers", "sweet peppers", "table", "tank", "telephone", "television",
    "tiger", "tractor", "train", "trout", "tulips", "turtle", "wardrobe", "whale",
    "willow", "wolf", "woman", "worm",
]


def _read_json(path: Path):
    return json.loads(path.read_text())


def resolve_custom_metadata_dir(opt):
    metadata_dir = getattr(opt, "custom_metadata_dir", "")
    if metadata_dir:
        path = Path(metadata_dir).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Missing custom_metadata_dir: {path}")
        return path
    return None


def load_custom_class_order(metadata_dir):
    return [
        line.strip()
        for line in (Path(metadata_dir) / "class_order.txt").read_text().splitlines()
        if line.strip()
    ]


def load_custom_synset_to_text(metadata_dir):
    return _read_json(Path(metadata_dir) / "synset_to_text.json")


def load_custom_synset_to_alias(metadata_dir):
    alias_path = Path(metadata_dir) / "synset_to_token_alias.json"
    if alias_path.is_file():
        return _read_json(alias_path)
    return {}


def get_legacy_class_names(id_data):
    if id_data == "in100":
        return LEGACY_IN100_CLASS_NAMES
    return LEGACY_C100_CLASS_NAMES


def get_class_texts_from_opt(opt):
    metadata_dir = resolve_custom_metadata_dir(opt)
    if metadata_dir is not None:
        class_order = load_custom_class_order(metadata_dir)
        synset_to_text = load_custom_synset_to_text(metadata_dir)
        return [synset_to_text[synset] for synset in class_order]
    return get_legacy_class_names(opt.id_data)


def get_token_labels_from_opt(opt):
    metadata_dir = resolve_custom_metadata_dir(opt)
    if metadata_dir is not None:
        class_order = load_custom_class_order(metadata_dir)
        synset_to_alias = load_custom_synset_to_alias(metadata_dir)
        synset_to_text = load_custom_synset_to_text(metadata_dir)
        return [synset_to_alias.get(synset, synset_to_text[synset]) for synset in class_order]
    return get_legacy_class_names(opt.id_data)
