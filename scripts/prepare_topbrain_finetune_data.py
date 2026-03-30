#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


NIFTI_SUFFIXES = (".nii.gz", ".nii")


@dataclass(frozen=True)
class Pair:
    case_id: str
    image_path: Path
    mask_path: Path


def resolve_path(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def file_stem_without_nifti(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    raise ValueError(f"Unsupported file extension for {name}")


def normalized_case_id(name: str) -> str:
    stem = file_stem_without_nifti(name)
    if stem.endswith("_0000"):
        return stem[:-5]
    return stem


def is_nifti(path: Path) -> bool:
    return path.is_file() and any(path.name.endswith(s) for s in NIFTI_SUFFIXES)


def build_index(folder: Path) -> Dict[str, Path]:
    if not folder.exists():
        return {}
    index: Dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if not is_nifti(path):
            continue
        case_id = normalized_case_id(path.name)
        if case_id in index:
            raise RuntimeError(f"Duplicate case id '{case_id}' in {folder}")
        index[case_id] = path
    return index


def collect_pairs(images_dir: Path, labels_dir: Path) -> Tuple[List[Pair], List[str], List[str]]:
    image_index = build_index(images_dir)
    label_index = build_index(labels_dir)
    image_ids = set(image_index.keys())
    label_ids = set(label_index.keys())
    common = sorted(image_ids & label_ids)
    missing_label = sorted(image_ids - label_ids)
    missing_image = sorted(label_ids - image_ids)
    pairs = [Pair(case_id=cid, image_path=image_index[cid], mask_path=label_index[cid]) for cid in common]
    return pairs, missing_label, missing_image


def split_train_val_test(
    train_pairs: Sequence[Pair],
    test_pairs: Sequence[Pair],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Pair], List[Pair], List[Pair]]:
    rng = random.Random(seed)

    train_pool = list(train_pairs)
    rng.shuffle(train_pool)

    explicit_test = list(test_pairs)
    if explicit_test:
        test_split = explicit_test
    else:
        if len(train_pool) < 3:
            raise RuntimeError(
                "Not enough labeled samples to auto-split train/val/test. Need at least 3 pairs in imagesTr/labelsTr "
                "or provide labeled imagesTs/labelsTs for test."
            )
        raw_test = int(round(len(train_pool) * test_ratio))
        test_count = max(1, min(raw_test, len(train_pool) - 2))
        test_split = train_pool[:test_count]
        train_pool = train_pool[test_count:]

    if len(train_pool) < 2:
        raise RuntimeError("Not enough training samples left after test split; need at least 2 for train+val.")

    raw_val = int(round(len(train_pool) * val_ratio))
    val_count = max(1, min(raw_val, len(train_pool) - 1))
    val_split = train_pool[:val_count]
    train_split = train_pool[val_count:]

    if len(train_split) == 0:
        raise RuntimeError("Train split is empty after splitting; adjust val_ratio/test_ratio.")

    return train_split, val_split, test_split


def ensure_clean_output(output_dir: Path, force: bool) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return

    if not force:
        return

    for child in ["train", "val", "test", "manifest.csv"]:
        path = output_dir / child
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return
    if mode == "hardlink":
        os.link(src, dst)
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    raise ValueError(f"Unsupported mode: {mode}")


def ext_for(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return ".nii.gz"
    if path.name.endswith(".nii"):
        return ".nii"
    raise ValueError(f"Unsupported extension: {path}")


def write_split(
    split_name: str,
    pairs: Sequence[Pair],
    output_dir: Path,
    mode: str,
) -> List[Dict[str, str]]:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    for pair in pairs:
        case_dir = split_dir / pair.case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        img_target = case_dir / f"img{ext_for(pair.image_path)}"
        mask_target = case_dir / f"mask{ext_for(pair.mask_path)}"

        link_or_copy(pair.image_path, img_target, mode)
        link_or_copy(pair.mask_path, mask_target, mode)

        rows.append(
            {
                "split": split_name,
                "case_id": pair.case_id,
                "image_path": str(img_target.resolve()),
                "mask_path": str(mask_target.resolve()),
                "image_source": str(pair.image_path.resolve()),
                "mask_source": str(pair.mask_path.resolve()),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TopBrain dataset for VesselFM finetuning.")
    parser.add_argument("--dataset-dir", default=os.getenv("DATASET_DIR", "./data/datasets/topBrain-2025"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    dataset_dir = resolve_path(args.dataset_dir, repo_root)
    output_dir = resolve_path(args.output_dir, repo_root) if args.output_dir else (dataset_dir / "vesselfm_finetune")

    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    labels_ts = dataset_dir / "labelsTs"

    if not images_tr.exists() or not labels_tr.exists():
        raise RuntimeError(
            f"Missing training folders. Expected {images_tr} and {labels_tr} to exist with paired NIfTI files."
        )

    if args.val_ratio <= 0 or args.val_ratio >= 1:
        raise RuntimeError("val-ratio must be in (0, 1)")
    if args.test_ratio <= 0 or args.test_ratio >= 1:
        raise RuntimeError("test-ratio must be in (0, 1)")

    train_pairs_raw, missing_label_tr, missing_image_tr = collect_pairs(images_tr, labels_tr)
    if len(train_pairs_raw) == 0:
        raise RuntimeError("No paired files found in imagesTr/labelsTr.")

    test_pairs_raw: List[Pair] = []
    if images_ts.exists() and labels_ts.exists():
        test_pairs_raw, _, _ = collect_pairs(images_ts, labels_ts)

    ensure_clean_output(output_dir, force=args.force)

    train_split, val_split, test_split = split_train_val_test(
        train_pairs=train_pairs_raw,
        test_pairs=test_pairs_raw,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    rows: List[Dict[str, str]] = []
    rows.extend(write_split("train", train_split, output_dir, args.link_mode))
    rows.extend(write_split("val", val_split, output_dir, args.link_mode))
    rows.extend(write_split("test", test_split, output_dir, args.link_mode))

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "case_id", "image_path", "mask_path", "image_source", "mask_source"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Prepared finetune dataset at: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"train={len(train_split)} val={len(val_split)} test={len(test_split)}")
    if missing_label_tr:
        print(f"Warning: {len(missing_label_tr)} training images without masks were skipped.")
    if missing_image_tr:
        print(f"Warning: {len(missing_image_tr)} training masks without images were skipped.")


if __name__ == "__main__":
    main()
