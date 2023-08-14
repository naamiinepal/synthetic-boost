import concurrent.futures
from enum import IntEnum
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from PIL import Image
from tqdm import tqdm


class OrganIndex(IntEnum):
    """
    Enum for the organ index in the multi-label mask
    """

    VENTRICULAR_CAVITY = 1
    MYOCARDIUM = 2
    ATRIUM_CAVITY = 3


def binarize_and_save(
    output_dir: Path,
    filename: Union[str, Path],
    np_img: np.ndarray,
    label: int,
):
    """
    Binarize a multi-label mask and save the binary mask

    Args:
        output_dir (Path): Output directory to save the binary mask
        filename (Union[str, Path]): Filename of the binary mask
        np_img (np.ndarray): Numpy array of the multi-label mask
        label (int): Label to binarize
    """
    output_filename = output_dir / filename
    output_filename.parent.mkdir(parents=True, exist_ok=True)

    bin_np_img: np.ndarray = np_img == label

    img = Image.fromarray(bin_np_img.astype(np.uint8) * 255)

    img.save(output_filename)


def load_and_save(
    root_dir: Path,
    ventricular_output_dir: Path,
    myocardium_output_dir: Path,
    atrium_output_dir: Path,
    file_path: Path,
):
    """
    Load a multi-label mask and save each label as a separate image

    Process-intsive function due to the use of numpy and comparision

    Args:
        root_dir (Path): Root directory containing the multi-label masks
        ventricular_output_dir (Path): Output directory to save the ventricular cavity masks
        myocardium_output_dir (Path): Output directory to save the myocardium masks
        atrium_output_dir (Path): Output directory to save the atrium cavity masks
        file_path (Path): Path to the multi-label mask
    """

    with Image.open(file_path) as img:
        np_img = np.asarray(img, dtype=np.uint8)

    filename = file_path.relative_to(root_dir)

    binarize_and_save(
        ventricular_output_dir, filename, np_img, OrganIndex.VENTRICULAR_CAVITY
    )
    binarize_and_save(myocardium_output_dir, filename, np_img, OrganIndex.MYOCARDIUM)
    binarize_and_save(atrium_output_dir, filename, np_img, OrganIndex.ATRIUM_CAVITY)


def main(
    root_dir: Path, output_root_dir: Path, glob_pattern: str, max_workers: Optional[int]
):
    print("Input directory: ", root_dir)
    print("Output root directory: ", output_root_dir)

    ventricular_output_dir = output_root_dir / f"mask_{OrganIndex.VENTRICULAR_CAVITY}"
    print("Ventricular output directory: ", ventricular_output_dir)

    myocardium_output_dir = output_root_dir / f"mask_{OrganIndex.MYOCARDIUM}"
    print("Myocardium output directory: ", myocardium_output_dir)

    atrium_output_dir = output_root_dir / f"mask_{OrganIndex.ATRIUM_CAVITY}"
    print("Atrium output directory: ", atrium_output_dir)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures_to_file_path: Dict[concurrent.futures.Future[None], Path] = {}
        for file_path in root_dir.glob(glob_pattern):
            f = executor.submit(
                load_and_save,
                root_dir,
                ventricular_output_dir,
                myocardium_output_dir,
                atrium_output_dir,
                file_path,
            )
            futures_to_file_path[f] = file_path

        print(f"\nProcessing multi-label masks with {executor._max_workers} workers")

        with tqdm(
            concurrent.futures.as_completed(futures_to_file_path),
            total=len(futures_to_file_path),
        ) as pbar:
            for future in pbar:
                file_path = futures_to_file_path[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"{file_path} generated an exception: {exc}")
                else:
                    pbar.set_postfix({"Input": file_path.stem})


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(
            "/mnt/Enterprise/PUBLIC_DATASETS/SDM_generated_data/all_frames_combined/sector_annotations"
        ),
        help="Root directory containing the multi-label masks",
    )

    parser.add_argument(
        "--output-root-dir",
        type=Path,
        default=Path(
            "/mnt/Enterprise/PUBLIC_DATASETS/SDM_generated_data/all_frames_combined/binary_anno"
        ),
        help="Root directory to save the binary masks",
    )

    parser.add_argument(
        "--glob-pattern",
        type=str,
        default="**/*.png",
        help="Glob pattern to search for multi-label masks",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of workers to use for parallel processing",
    )

    args = parser.parse_args()

    main(**vars(args))
