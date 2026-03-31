"""Inference that preserves original image geometry in output masks."""

import logging
import warnings
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import SlidingWindowInfererAdapt
from skimage.exposure import equalize_hist
from skimage.morphology import remove_small_objects
from tqdm import tqdm

from vesselfm.seg.inference import get_paths, load_model, resolve_output_suffix, strip_nifti_suffix
from vesselfm.seg.utils.data import generate_transforms
from vesselfm.seg.utils.evaluation import Evaluator, calculate_mean_metrics
from vesselfm.seg.utils.io import determine_reader_writer


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def resolve_write_metadata(image_metadata: Optional[dict]) -> Optional[dict]:
    if not isinstance(image_metadata, dict):
        return None
    other = image_metadata.get("other")
    if isinstance(other, dict) and all(k in other for k in ("spacing", "origin", "direction")):
        return {
            "spacing": other["spacing"],
            "origin": other["origin"],
            "direction": other["direction"],
        }
    if all(k in image_metadata for k in ("spacing", "origin", "direction")):
        return {
            "spacing": image_metadata["spacing"],
            "origin": image_metadata["origin"],
            "direction": image_metadata["direction"],
        }
    return None


def resample(image, factor=None, target_shape=None):
    if factor == 1:
        return image

    if target_shape:
        _, _, new_d, new_h, new_w = target_shape
    else:
        _, _, d, h, w = image.shape
        new_d, new_h, new_w = int(round(d / factor)), int(round(h / factor)), int(round(w / factor))
    return F.interpolate(image, size=(new_d, new_h, new_w), mode="trilinear", align_corners=False)


@hydra.main(config_path="configs", config_name="inference", version_base="1.3.2")
def main(cfg):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    logger.info(f"Using device {cfg.device}.")
    device = cfg.device

    model = load_model(cfg, device)
    model.to(device)
    model.eval()

    transforms = generate_transforms(cfg.transforms_config)

    output_folder = Path(cfg.output_folder)
    output_folder.mkdir(exist_ok=True)

    image_paths, mask_paths = get_paths(cfg)
    logger.info(f"Found {len(image_paths)} images in {cfg.image_path}.")

    file_ending = cfg.image_file_ending if cfg.image_file_ending else image_paths[0].suffix
    output_suffix = resolve_output_suffix(cfg)
    image_reader_writer = determine_reader_writer(file_ending)()
    save_writer = determine_reader_writer(file_ending)()

    inferer = SlidingWindowInfererAdapt(
        roi_size=cfg.patch_size,
        sw_batch_size=cfg.batch_size,
        overlap=cfg.overlap,
        mode=cfg.mode,
        sigma_scale=cfg.sigma_scale,
        padding_mode=cfg.padding_mode,
    )

    metrics_dict = {}
    with torch.no_grad():
        for idx, image_path in tqdm(enumerate(image_paths), total=len(image_paths), desc="Processing images."):
            image_np, image_metadata = image_reader_writer.read_images(image_path)
            write_metadata = resolve_write_metadata(image_metadata)

            preds = []
            for scale in cfg.tta.scales:
                image = transforms(image_np.astype(np.float32))[None].to(device)
                mask = torch.tensor(image_reader_writer.read_images(mask_paths[idx])[0]).bool() if mask_paths else None

                if cfg.tta.invert:
                    image = 1 - image if image.mean() > cfg.tta.invert_mean_thresh else image

                if cfg.tta.equalize_hist:
                    image_np_tta = image.cpu().squeeze().numpy()
                    image_equal_hist_np = equalize_hist(image_np_tta, nbins=cfg.tta.hist_bins)
                    image = torch.from_numpy(image_equal_hist_np).to(image.device)[None][None]

                original_shape = image.shape
                image = resample(image, factor=scale)
                logits = inferer(image, model)
                logits = resample(logits, target_shape=original_shape)
                preds.append(logits.cpu().squeeze())

            if cfg.merging.max:
                pred = torch.stack(preds).max(dim=0)[0].sigmoid()
            else:
                pred = torch.stack(preds).mean(dim=0).sigmoid()
            pred_thresh = (pred > cfg.merging.threshold).numpy()

            if cfg.post.apply:
                pred_thresh = remove_small_objects(
                    pred_thresh,
                    min_size=cfg.post.small_objects_min_size,
                    connectivity=cfg.post.small_objects_connectivity,
                )

            image_stem = strip_nifti_suffix(image_path)
            save_writer.write_seg(
                pred_thresh.astype(np.uint8),
                output_folder / f"{image_stem}{output_suffix}.{file_ending}",
                metadata=write_metadata,
            )

            if mask_paths is not None:
                metrics = Evaluator().estimate_metrics(pred, mask, threshold=cfg.merging.threshold)
                logger.info(f"Dice of {image_stem}: {metrics['dice'].item()}")
                logger.info(f"clDice of {image_stem}: {metrics['cldice'].item()}")
                metrics_dict[image_stem] = metrics

    if mask_paths is not None:
        mean_metrics = calculate_mean_metrics(list(metrics_dict.values()), round_to=cfg.round_to)
        logger.info(f"Mean metrics: dice {mean_metrics['dice'].item()}, cldice {mean_metrics['cldice'].item()}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
