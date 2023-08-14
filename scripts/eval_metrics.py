"""
Script to evaluate the segmentation metrics.
The script takes in the path to the segmentation and ground truth images and
computes the following metrics:
1. Surface Dice
2. Hausdorff Distance
3. IoU
4. Dice

NOTE: The script assumes that the segmentation and ground truth images
have the same name. The script also assumes that the images are binary
images with pixel values 0 and 255. The script thresholds the images to 0 and 1
and computes the metrics. The script also assumes that the images are of size
[H, W] and not [H, W, C], and are of type uint8.

The script saves the metrics in a csv file.

Usage:
    python eval_metrics.py \
        --seg_path <path to segmentation images> \
        --gt_path <path to ground truth images> \
        --csv_path <path to save csv file>
"""


import concurrent.futures
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from monai.metrics import compute_dice, compute_hausdorff_distance, compute_iou
from tqdm import tqdm


def compute_metrics(gt_img_path: str, pred_img_path: str):
    gt_img = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)
    pred_img = cv2.imread(pred_img_path, cv2.IMREAD_GRAYSCALE)

    # make sure the images are of same size
    assert (
        gt_img.shape == pred_img.shape
    ), f"Images {gt_img_path} and {pred_img_path} are of different sizes"

    # threshold the images

    gt_img = gt_img > 128
    pred_img = pred_img > 128

    # change images to batch-first tensor [B,C,H,W]
    gt_img = torch.from_numpy(gt_img)[None, None, ...]
    pred_img = torch.from_numpy(pred_img)[None, None, ...]

    # compute the metrics
    hausdorff_distance = compute_hausdorff_distance(pred_img, gt_img) * 100

    iou = compute_iou(pred_img, gt_img, ignore_empty=False) * 100
    dice = compute_dice(pred_img, gt_img, ignore_empty=False) * 100

    all_ones_pred = torch.ones_like(pred_img)
    all_ones_dice = compute_dice(all_ones_pred, gt_img, ignore_empty=False) * 100
    ones_dice_diff = dice - all_ones_dice

    all_zeros_dice = torch.zeros_like(pred_img)
    all_zeros_dice = compute_dice(all_zeros_dice, gt_img, ignore_empty=False) * 100

    zeros_dice_diff = dice - all_zeros_dice

    return {
        "iou": iou.item(),
        "dice": dice.item(),
        "hausdorff_distance": hausdorff_distance.item(),
        "ones_dice_diff": ones_dice_diff.item(),
        "zeros_dice_diff": zeros_dice_diff.item(),
    }


def main(
    seg_path: Path,
    seg_glob_pattern: str,
    gt_path: Path,
    gt_glob_pattern: str,
    csv_path: Path,
    max_workers: Optional[int],
):
    assert csv_path.suffix == ".csv", f"csv_path {csv_path} should be a csv file"

    # create the parent folder of CSV file if it doesn't exist
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    np.set_printoptions(precision=5)

    seg_paths = list(seg_path.glob(seg_glob_pattern))
    assert len(seg_paths) > 0, f"No segmentation images found in {seg_path}"

    gt_paths = list(gt_path.glob(gt_glob_pattern))
    assert len(gt_paths) > 0, f"No ground truth images found in {gt_path}"

    assert len(seg_paths) == len(
        gt_paths
    ), "Number of segmentation and ground truth images are not equal"

    # Sorting the paths to make sure the images are in the same order
    seg_paths.sort()
    gt_paths.sort()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for image_paths in zip(seg_paths, gt_paths):
            pred_img_path, gt_img_path = image_paths

            assert pred_img_path.name == gt_img_path.name, (
                f"Segmentation image {pred_img_path} and ground truth image "
                f"{gt_img_path} have different names"
            )

            futures[
                executor.submit(compute_metrics, str(gt_img_path), str(pred_img_path))
            ] = image_paths

        aggregator = defaultdict(list)

        with tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Evaluating metrics",
        ) as pbar:
            for future in pbar:
                pred_img_path, gt_img_path = futures[future]
                try:
                    results = future.result()
                except Exception as exc:
                    print(
                        f"Exception while processing {pred_img_path} and {gt_img_path}: {exc}"
                    )
                else:
                    aggregator["filename"].append(str(pred_img_path))
                    for key, value in results.items():
                        aggregator[key].append(value)
                finally:
                    pbar.set_postfix(
                        {
                            "Mean Dice": np.nanmean(aggregator["dice"]),
                            "Mean IoU": np.nanmean(aggregator["iou"]),
                        }
                    )

    df = pd.DataFrame(aggregator)

    # print mean and std for each metric
    for key in df.columns:
        if key != "filename":
            print_mean_std(df, key)

    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved metrics to {csv_path}")


def print_mean_std(df: pd.DataFrame, column_name: str):
    column = df[column_name]
    print(
        column_name.replace("_", " ").title(),
        f"${column.mean().round(2)} \smallStd{{{column.std().round(2)}}}$",
    )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--seg-path",
        type=Path,
        required=True,
        help="path to segmentation files",
    )

    parser.add_argument(
        "--seg-glob-pattern",
        type=str,
        required=True,
        help="glob pattern to match segmentation files",
    )

    parser.add_argument(
        "--gt-path",
        type=Path,
        required=True,
        help="path to ground truth files",
    )
    parser.add_argument(
        "--gt-glob-pattern",
        type=str,
        required=True,
        help="glob pattern to match ground truth files",
    )

    parser.add_argument(
        "--csv-path", type=Path, required=True, help="path to save csv file"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="maximum number of workers to use for multiprocessing",
    )

    args = parser.parse_args()

    main(**vars(args))
