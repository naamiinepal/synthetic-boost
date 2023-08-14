import concurrent.futures
import json
import sys
from collections.abc import MutableSequence
from pathlib import Path
from string import Template
from typing import Any, Iterable, Mapping, Optional, Tuple, Union

from num2words import num2words
from PIL import Image
from tqdm import tqdm

sys.path.append("/mnt/Enterprise/rabin/VLM-SEG-2023/OFA")

from single_inference import get_answer, return_model

StrPath = Union[str, Path]


def convert_str_to_template(temp_str: Union[Iterable[str], str]):
    """Convert a string or an iterable of strings to a template

    Args:
        temp_str (Union[Iterable[str], str]): The string or iterable of strings to convert

    Returns:
        Union[Template, Tuple[Template]]: The template or tuple of templates
    """

    if isinstance(temp_str, str):
        return Template(temp_str)

    return tuple(map(Template, temp_str))


mask_index_to_parts = ["Left ventricular cavity", "Myocardium", "Left atrium cavity"]

p0 = ""

p1_templates = convert_str_to_template(
    (
        "$label_name of the heart.",
        "$label_name in the cardiac ultrasound.",
    )
)


p2_templates = convert_str_to_template(
    (
        "$label_name in $num_chambers-chamber view of the heart.",
        "$label_name in $num_chambers-chamber view in the cardiac ultrasound.",
    )
)

p3_templates = convert_str_to_template(
    (
        "$label_name in $num_chambers-chamber view of the heart at end of the $cycle cycle.",
        "$label_name in $num_chambers-chamber view in the cardiac ultrasound at end of the $cycle cycle.",
    )
)

p4_templates = convert_str_to_template(
    (
        "$label_name in $num_chambers-chamber view of the heart at end of the $cycle cycle of a $gender.",
        "$label_name in $num_chambers-chamber view in the cardiac ultrasound at end of the $cycle cycle of a $gender.",
    )
)

p5_templates = convert_str_to_template(
    (
        "$label_name in $num_chambers-chamber view of the heart at end of the $cycle cycle of a $age-year-old $gender.",
        "$label_name in $num_chambers-chamber view in the cardiac ultrasound at end of the $cycle cycle of a $age-year-old $gender.",
    )
)

p6_templates = convert_str_to_template(
    (
        "$label_name of $shape shape in $num_chambers-chamber view of the heart at end of the $cycle cycle of a $age-year-old $gender.",
        "$label_name of $shape shape in $num_chambers-chamber view in the cardiac ultrasound at end of the $cycle cycle of a $age-year-old $gender.",
    )
)


def get_no_vqa_json(
    raw_root: Path, image_mask_path: Tuple[Path, Path], mask_dir: StrPath
):
    """Get the json for a single image-mask pair without VQA

    Args:
        raw_root (Path): The path to the raw data
        image_mask_path (Tuple[Path, Path]): The tuple of the paths to the image and mask
        mask_dir (StrPath): The path to the mask directory

    Raises:
        ValueError: The image and mask names differ

    Returns:
        Dict[str, Any]: The json
    """

    image_path, mask_path = image_mask_path

    assert (
        image_path.name == mask_path.name
    ), f"Image and mask names differ at for Image path: {image_path}, mask path: {mask_path}."

    mask_path_parent_name = mask_path.parent.parent.name
    try:
        label_index = int(mask_path_parent_name[-1])
    except ValueError:
        raise ValueError(
            f"Mask directory name should end with a number for {mask_path}."
        )

    if label_index < 1 or label_index > 3:
        print(
            f"Clipping Label index: {label_index} because it is not in range 1-3 for {mask_path}"
        )

        # Get minimum of 1 and maximum of 3
        label_index = min(max(label_index, 1), 3)

    # Subtract to get the index in the list
    label_name = mask_index_to_parts[label_index - 1]

    temp_sub_kwargs = {"label_name": label_name}
    p1 = [temp.substitute(**temp_sub_kwargs) for temp in p1_templates]

    mask_path_stem = mask_path.stem

    patient_id, _chamber, _stage, *_ = mask_path_stem.split("_")

    num_chambers = "two" if _chamber == "2CH" else "four"

    temp_sub_kwargs["num_chambers"] = num_chambers
    p2 = [temp.substitute(**temp_sub_kwargs) for temp in p2_templates]

    cycle = "systole" if _stage == "ES" else "diastole"

    temp_sub_kwargs["cycle"] = cycle
    p3 = [temp.substitute(**temp_sub_kwargs) for temp in p3_templates]

    with open(raw_root / patient_id / f"Info_{_chamber}.cfg") as f:
        content = f.read()

        # Split into lines
    content = content.splitlines()

    # Separate by colon
    key_value_tuple = (line.split(":", 1) for line in content)

    # Remove leading and trailing spaces
    key_value_mapping = {key.strip(): value.strip() for key, value in key_value_tuple}

    _gender = key_value_mapping["Sex"]

    gender = "female" if _gender == "F" else "male"

    temp_sub_kwargs["gender"] = gender
    p4 = [temp.substitute(**temp_sub_kwargs) for temp in p4_templates]

    age = num2words(key_value_mapping["Age"])

    temp_sub_kwargs["age"] = age
    p5 = [temp.substitute(**temp_sub_kwargs) for temp in p5_templates]

    mask_name = str(mask_path.relative_to(mask_dir))

    img_name = image_path.name

    return {
        "task": {
            "img_name": img_name,
            "mask_name": mask_name,
            "prompts": {
                "p0": p0,
                "p1": p1,
                "p2": p2,
                "p3": p3,
                "p4": p4,
                "p5": p5,
            },
        },
        "meta": {
            "image_mask_path": image_mask_path,
            "label_name": label_name,
            "temp_sub_kwargs": temp_sub_kwargs,
        },
    }


def get_vqa_prompt(
    image_mask_path: Tuple[Path, Path],
    model,
    label_name: str,
    temp_sub_kwargs: Mapping[str, str],
):
    """Get the json for a single image-mask pair

    Args:
        image_mask_path (Tuple[Path, Path]): The tuple of the paths to the image and mask
        model (VQA): The VQA model
        label_name (str): The label name
        temp_sub_kwargs (Mapping[str, str]): The template substitution kwargs

    Raises:
        ValueError: The image and mask names differ

    Returns:
        dict: The json for the image-mask pair and prompts
    """

    image_path, mask_path = image_mask_path

    with Image.open(mask_path) as mask, Image.open(image_path) as image:
        # TODO: Incorporate later on
        shape = get_answer(
            model,
            image,
            mask,
            question=f"What is the shape of the {label_name.lower()} enclosed in the green box?",
        )

    vqa_prompt = [
        temp.substitute(**temp_sub_kwargs, shape=shape) for temp in p6_templates
    ]

    return vqa_prompt


def post_process_task(
    image_path: Path,
    training_json: MutableSequence,
    validation_json: MutableSequence,
    testing_json: MutableSequence,
    task,
    pbar: Optional[tqdm] = None,
    post_fix: Optional[Mapping[str, Any]] = None,
):
    """
    Post process the task

    Args:
        image_mask_path (Sequence[Path] | Mapping[int, Path]): The tuple of the paths to the image and mask
        training_json (MutableSequence): The list of training jsons
        validation_json (MutableSequence): The list of validation jsons
        testing_json (MutableSequence): The list of testing jsons
        task: The task
        pbar (tqdm): The progress bar
    """
    path_stem = image_path.stem

    if "training" in path_stem:
        training_json.append(task)
    elif "validation" in path_stem:
        validation_json.append(task)
    elif "testing" in path_stem:
        testing_json.append(task)
    else:
        print("Neither training nor validation nor testing")

    if post_fix is not None and pbar is not None:
        pbar.set_postfix(post_fix)


def get_json_data(
    img_dir: Path,
    img_pattern: str,
    mask_dir: Path,
    mask_pattern: str,
    output_dir: Path,
    max_workers: Optional[int],
):
    """Get the json data for the CAMUS dataset

    Args:
        img_dir (StrPath): The path to the image directory
        img_pattern (str): The glob pattern for the images
        mask_dir (StrPath): The path to the mask directory
        mask_pattern (str): The glob pattern for the masks
        output_dir (StrPath): The path to the output directory
        max_workers (int, optional): The number of max workers.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(img_dir.glob(img_pattern))
    assert len(image_paths) > 0, "No images found in the image directory."

    mask_paths = list(mask_dir.glob(mask_pattern))
    assert len(mask_paths) > 0, "No files found in the mask directory."

    assert (
        len(mask_paths) % len(image_paths) == 0
    ), "The number of masks should be the multiple of the number of images."

    assert len(image_paths) == len(
        set(path.name for path in mask_paths)
    ), "The number of images and the number of masks should be equal. "

    # Sort image and mask paths
    image_paths.sort(key=lambda x: x.stem)

    # Sort only by the last names
    mask_paths.sort(key=lambda x: x.stem)

    mask_multiple = len(mask_paths) // len(image_paths)

    image_paths = [path for path in image_paths for _ in range(mask_multiple)]

    assert len(image_paths) == len(
        mask_paths
    ), "The number of images and masks differ even after adjusting them."

    image_mask_paths = tuple(zip(image_paths, mask_paths))

    raw_root = Path("/mnt/Enterprise/PUBLIC_DATASETS/camus_database/training")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_image_mask_path = {}
        for image_mask_path in image_mask_paths:
            future = executor.submit(
                get_no_vqa_json, raw_root, image_mask_path, mask_dir
            )
            future_to_image_mask_path[future] = image_mask_path

        no_vqa_outputs = []

        for future in tqdm(
            concurrent.futures.as_completed(future_to_image_mask_path),
            total=len(future_to_image_mask_path),
            desc="No VQA Tasks",
        ):
            image_mask_path = future_to_image_mask_path[future]
            try:
                output = future.result()
            except Exception as exc:
                print(f"{image_mask_path} generated an exception: {exc}")
            else:
                no_vqa_outputs.append(output)

    training_json = []
    validation_json = []
    testing_json = []

    model = return_model()

    for output in tqdm(no_vqa_outputs, desc="Getting VQA"):
        meta = output["meta"]
        task = output["task"]

        vqa_prompt = get_vqa_prompt(**meta, model=model)

        task["prompts"]["p6"] = vqa_prompt

        image_path = meta["image_mask_path"][0]

        # post_fix = {
        #     "image": image_path.stem,
        #     "vqa": f"{shape}: {meta['label_name']}",
        # }

        post_process_task(
            image_path,
            training_json,
            validation_json,
            testing_json,
            task,
        )

    print(
        "\n\nNumber of tasks:",
        len(training_json),
        len(validation_json),
        len(testing_json),
    )

    total_len = len(training_json) + len(validation_json) + len(testing_json)

    print(
        "Proportion of tasks:",
        len(training_json) / total_len,
        len(validation_json) / total_len,
        len(testing_json) / total_len,
    )

    training_json.sort(key=lambda x: (x["img_name"], x["mask_name"]))

    with (output_dir / "train.json").open("w") as of:
        json.dump(training_json, of, indent=2)

    validation_json.sort(key=lambda x: (x["img_name"], x["mask_name"]))
    with (output_dir / "val.json").open("w") as of:
        json.dump(validation_json, of, indent=2)

    testing_json.sort(key=lambda x: (x["img_name"], x["mask_name"]))
    with (output_dir / "test.json").open("w") as of:
        json.dump(testing_json, of, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=Path, required=True)
    parser.add_argument("--img-pattern", type=str, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--mask-pattern", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=None)

    args = parser.parse_args()

    get_json_data(**vars(args))
