""" Script to perform inference with vesselFM."""

import logging
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
import hydra
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from monai.inferers import SlidingWindowInfererAdapt
from skimage.morphology import remove_small_objects
from skimage.exposure import equalize_hist

from vesselfm.seg.utils.data import generate_transforms
from vesselfm.seg.utils.io import determine_reader_writer
from vesselfm.seg.utils.evaluation import Evaluator, calculate_mean_metrics


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def extract_state_dict(ckpt):
    """
    Supports:
    1) Raw model state_dict
    2) Lightning checkpoint with "state_dict" and optional "model." prefix
    """
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")


def load_model(cfg, device):
    try:
        logger.info(f"Loading model from {cfg.ckpt_path}.")
        ckpt = torch.load(Path(cfg.ckpt_path), map_location=device)
    except:
        logger.info(f"Loading model from Hugging Face.")
        hf_hub_download(repo_id='bwittmann/vesselFM', filename='meta.yaml') # required to track downloads
        ckpt = torch.load(
            hf_hub_download(repo_id='bwittmann/vesselFM', filename='vesselFM_base.pt'),
            map_location=device
        )

    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(extract_state_dict(ckpt), strict=True)
    return model


def strip_nifti_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def resolve_output_suffix(cfg) -> str:
    if hasattr(cfg, "output_suffix") and cfg.output_suffix is not None:
        suffix = str(cfg.output_suffix)
    else:
        # backward compatibility with old naming style
        suffix = f"_{cfg.file_app}pred" if getattr(cfg, "file_app", "") else "_pred"
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    return suffix


def list_images(image_path: Path, file_ending: str):
    if image_path.is_file():
        return [image_path]
    if image_path.is_dir():
        paths = [p for p in sorted(image_path.iterdir()) if p.is_file()]
        if file_ending:
            paths = [p for p in paths if p.name.lower().endswith(file_ending.lower())]
        return paths
    raise FileNotFoundError(f"image_path does not exist: {image_path}")


def get_paths(cfg):
    image_paths = list_images(Path(cfg.image_path), cfg.image_file_ending)
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {cfg.image_path} with ending '{cfg.image_file_ending}'.")
    if cfg.mask_path:
        mask_input_path = Path(cfg.mask_path)
        if mask_input_path.is_file():
            if len(image_paths) != 1:
                raise RuntimeError("mask_path is a file but image_path contains multiple images.")
            mask_paths = [mask_input_path]
        else:
            mask_paths = [mask_input_path / p.name for p in image_paths]
            assert all(
                mask_path.exists() for mask_path in mask_paths
            ), "All mask paths must exist and mask names must match image names."
    else:
        mask_paths = None
    return image_paths, mask_paths

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
    # seed libraries
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # set device
    logger.info(f"Using device {cfg.device}.")
    device = cfg.device

    # load model and ckpt
    model = load_model(cfg, device)
    model.to(device)
    model.eval()

    # init pre-processing transforms
    transforms = generate_transforms(cfg.transforms_config)

    # i/o
    output_folder = Path(cfg.output_folder)
    output_folder.mkdir(exist_ok=True)

    image_paths, mask_paths = get_paths(cfg)
    logger.info(f"Found {len(image_paths)} images in {cfg.image_path}.")

    file_ending = (cfg.image_file_ending if cfg.image_file_ending else image_paths[0].suffix)
    output_suffix = resolve_output_suffix(cfg)
    image_reader_writer = determine_reader_writer(file_ending)()
    save_writer = determine_reader_writer(file_ending)()

    # init sliding window inferer
    logger.debug(f"Sliding window patch size: {cfg.patch_size}")
    logger.debug(f"Sliding window batch size: {cfg.batch_size}.")
    logger.debug(f"Sliding window overlap: {cfg.overlap}.")
    inferer = SlidingWindowInfererAdapt(
        roi_size=cfg.patch_size, sw_batch_size=cfg.batch_size, overlap=cfg.overlap, 
        mode=cfg.mode, sigma_scale=cfg.sigma_scale, padding_mode=cfg.padding_mode
    )

    # loop over images
    metrics_dict = {}
    with torch.no_grad():
        for idx, image_path in tqdm(enumerate(image_paths), total=len(image_paths), desc="Processing images."):
            preds = [] # average over test time augmentations
            for scale in cfg.tta.scales:
                # apply pre-processing transforms
                image = transforms(image_reader_writer.read_images(image_path)[0].astype(np.float32))[None].to(device)
                mask = torch.tensor(image_reader_writer.read_images(mask_paths[idx])[0]).bool() if mask_paths else None
  
                # apply test time augmentation
                if cfg.tta.invert:
                    image = 1 - image if image.mean() > cfg.tta.invert_mean_thresh else image
                    
                if cfg.tta.equalize_hist:
                    image_np = image.cpu().squeeze().numpy()
                    image_equal_hist_np = equalize_hist(image_np, nbins=cfg.tta.hist_bins)
                    image = torch.from_numpy(image_equal_hist_np).to(image.device)[None][None]

                original_shape = image.shape
                image = resample(image, factor=scale)
                logits = inferer(image, model)
                logits = resample(logits, target_shape=original_shape)
                preds.append(logits.cpu().squeeze())

            # merging
            if cfg.merging.max:
                pred = torch.stack(preds).max(dim=0)[0].sigmoid()
            else:
                pred = torch.stack(preds).mean(dim=0).sigmoid()
            pred_thresh = (pred > cfg.merging.threshold).numpy()

            # post-processing
            if cfg.post.apply:
                pred_thresh = remove_small_objects(
                    pred_thresh, min_size=cfg.post.small_objects_min_size, connectivity=cfg.post.small_objects_connectivity
                )

            # save final pred
            image_stem = strip_nifti_suffix(image_path)
            save_writer.write_seg(
                pred_thresh.astype(np.uint8), output_folder / f"{image_stem}{output_suffix}.{file_ending}"
            )

            if mask_paths is not None:
                metrics = Evaluator().estimate_metrics(pred, mask, threshold=cfg.merging.threshold) # no post-processing
                logger.info(f"Dice of {image_stem}: {metrics['dice'].item()}")
                logger.info(f"clDice of {image_stem}: {metrics['cldice'].item()}")
                metrics_dict[image_stem] = metrics

    if mask_paths is not None:
        mean_metrics = calculate_mean_metrics(list(metrics_dict.values()), round_to=cfg.round_to)
        logger.info(f"Mean metrics: dice {mean_metrics['dice'].item()}, cldice {mean_metrics['cldice'].item()}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
