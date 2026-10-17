from collections import OrderedDict
from pathlib import Path
import os


REPO_ROOT = Path(__file__).resolve().parents[1]


def _clean_candidates(values):
    cleaned = []
    seen = set()
    for value in values:
        if not value:
            continue
        path = str(Path(value).expanduser())
        if path in seen:
            continue
        seen.add(path)
        cleaned.append(path)
    return cleaned


def first_existing_path(candidates):
    for candidate in _clean_candidates(candidates):
        if Path(candidate).exists():
            return candidate
    return None


def resolve_path(explicit_path, candidates):
    searched = _clean_candidates(([explicit_path] if explicit_path else []) + list(candidates))
    resolved = first_existing_path(searched)
    if resolved is not None:
        return resolved, searched
    fallback = str(Path(explicit_path).expanduser()) if explicit_path else (searched[0] if searched else "")
    return fallback, searched


def remove_data_parallel_prefix(old_state_dict):
    new_state_dict = OrderedDict()
    for key, value in old_state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        new_state_dict[new_key] = value
    return new_state_dict


def default_token_embed_path(dataset):
    if dataset == "cifar100":
        return str(REPO_ROOT / "assets" / "token_embed_c100.npy")
    return str(REPO_ROOT / "data" / "imagenet100" / "metadata" / "token_embed_custom100.npy")


def default_id_feat_path(dataset):
    if dataset == "cifar100":
        return str(REPO_ROOT / "outputs" / "cifar100" / "metadata" / "id_feat_cifar100_199epoch.npy")
    return str(REPO_ROOT / "outputs" / "imagenet100" / "metadata" / "id_feat_custom100_full.npy")


# All dataset roots default to <repo>/data/... and can be overridden with SBOG_* environment variables.
DATA_ROOT = Path(os.environ.get("SBOG_DATA_ROOT", str(REPO_ROOT / "data")))


def default_cifar100_data_candidates():
    return _clean_candidates([
        os.environ.get("SBOG_CIFAR100_ROOT", ""),
        str(DATA_ROOT / "cifarpy"),
    ])


def default_in100_ood_candidates():
    return {
        "inat_root": _clean_candidates([os.environ.get("SBOG_INAT_ROOT", ""), str(DATA_ROOT / "imagenet_ood" / "iNaturalist_imagefolder")]),
        "places_root": _clean_candidates([os.environ.get("SBOG_PLACES_ROOT", ""), str(DATA_ROOT / "imagenet_ood" / "Places_imagefolder")]),
        "sun_root": _clean_candidates([os.environ.get("SBOG_SUN_ROOT", ""), str(DATA_ROOT / "imagenet_ood" / "SUN_imagefolder")]),
        "textures_root": _clean_candidates([os.environ.get("SBOG_TEXTURES_ROOT", ""), str(DATA_ROOT / "dtd" / "images")]),
    }


def default_cifar100_ood_candidates():
    return {
        "textures_root": _clean_candidates([os.environ.get("SBOG_TEXTURES_ROOT", ""), str(DATA_ROOT / "dtd" / "images")]),
        "svhn_root": _clean_candidates([os.environ.get("SBOG_SVHN_ROOT", ""), str(DATA_ROOT / "svhn")]),
        "places_root": _clean_candidates([os.environ.get("SBOG_PLACES365_ROOT", ""), str(DATA_ROOT / "places365_test")]),
        "lsun_root": _clean_candidates([os.environ.get("SBOG_LSUN_C_ROOT", ""), str(DATA_ROOT / "LSUN_C")]),
        "isun_root": _clean_candidates([os.environ.get("SBOG_ISUN_ROOT", ""), str(DATA_ROOT / "iSUN")]),
    }


def resolve_generated_ood_root(explicit_ood_root="", generated_ood_root="", my_info=""):
    candidates = []
    if explicit_ood_root:
        candidates.append(explicit_ood_root)

    if generated_ood_root and my_info:
        base = Path(generated_ood_root).expanduser() / my_info
        candidates.extend([str(base / "samples"), str(base)])
    elif generated_ood_root:
        base = Path(generated_ood_root).expanduser()
        candidates.extend([str(base / "samples"), str(base)])

    if my_info:
        info_path = Path(my_info).expanduser()
        candidates.extend([str(info_path / "samples"), str(info_path)])

    resolved, searched = resolve_path("", candidates)
    return resolved, searched


def format_missing_path_message(label, searched_paths):
    lines = [f"Missing {label}."]
    if searched_paths:
        lines.append("Checked:")
        for path in searched_paths:
            lines.append(f"  - {path}")
    return "\n".join(lines)
