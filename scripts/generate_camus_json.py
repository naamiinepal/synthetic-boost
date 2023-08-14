import concurrent.futures
import json
import sys
from pathlib import Path
from string import Template
from typing import Iterable, List, Mapping, Optional, Tuple, Union

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
        "$label_name in $num_chambers-chamber view of the heart at end of the $cycle cycle of a $age-year-old $gender with $image_quality image quality.",
        "$label_name in $num_chambers-chamber view in the cardiac ultrasound at end of the $cycle cycle of a $age-year-old $gender with $image_quality image quality.",
    )
)

p7_templates = convert_str_to_template(
    (
        "$label_name of $shape shape in $num_chambers-chamber view of the heart at end of the $cycle cycle of a $age-year-old $gender with $image_quality image quality.",
        "$label_name of $shape shape in $num_chambers-chamber view in the cardiac ultrasound at end of the $cycle cycle of a $age-year-old $gender with $image_quality image quality.",
    )
)

required_keys = ("img_name", "mask_name", "prompts")


def load_and_sanitize(json_path: Path):
    with json_path.open() as f:
        tasks: List[Mapping[str]] = json.load(f)

    sanitized_tasks = []

    for task in tasks:
        san_task = {k: v for k, v in task.items() if k in required_keys}

        mask_name = san_task["mask_name"]

        stem, suffix = mask_name.rsplit(".", 1)

        root_stem, mask_index = stem.rsplit("_", 1)

        new_mask_name = f"mask_{mask_index}/{root_stem}.{suffix}"

        san_task["mask_name"] = new_mask_name

        sanitized_tasks.append(san_task)

    return sanitized_tasks


def get_no_vqa_json(
    raw_root: Path, image_mask_path: Tuple[Path, Path], mask_dir: StrPath
):
    """Get the json for a single image-mask pair

    Args:
        raw_root (Path): The path to the raw data
        image_mask_path (Tuple[Path, Path]): The tuple of the paths to the image and mask
        mask_dir (StrPath): The path to the mask directory

    Raises:
        ValueError: The image and mask names differ

    Returns:
        dict: The json for containing the bbox, prompts and sentences
    """

    image_path, mask_path = image_mask_path

    assert (
        image_path.stem + "_gt" == mask_path.stem
    ), f"Image and mask names differ. Image path: {image_path}, mask path: {mask_path}."

    mask_path_parent_name = mask_path.parent.name
    try:
        label_index = int(mask_path_parent_name[-1])
    except ValueError:
        raise ValueError(
            f"Mask directory name should end with a number for {mask_path}."
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

    with (raw_root / patient_id / f"Info_{_chamber}.cfg").open() as f:
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

    image_quality = key_value_mapping["ImageQuality"].lower()

    temp_sub_kwargs["image_quality"] = image_quality
    p6 = [temp.substitute(**temp_sub_kwargs) for temp in p6_templates]

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
                "p6": p6,
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
        temp.substitute(**temp_sub_kwargs, shape=shape) for temp in p7_templates
    ]

    return vqa_prompt


def get_json_data(
    camus_prev_anns_root: Path,
    img_dir: StrPath,
    img_pattern: str,
    mask_dir: StrPath,
    mask_pattern: str,
    output_dir: StrPath,
    max_workers: Optional[int],
):
    """Get the json data for the CAMUS dataset

    Args:
        camus_prev_anns_root (Path): The path to the previous annotations root directory
        img_dir (StrPath): The path to the image directory
        img_pattern (str): The glob pattern for the images
        mask_dir (StrPath): The path to the mask directory
        mask_pattern (str): The glob pattern for the masks
        output_dir (StrPath): The path to the output directory
        max_workers (Optional[int]): The max workers for multiprocessing
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and sanitizing previous annotations from", camus_prev_anns_root)

    train_tasks = load_and_sanitize(camus_prev_anns_root / "train.json")
    val_tasks = load_and_sanitize(camus_prev_anns_root / "val.json")
    test_tasks = load_and_sanitize(camus_prev_anns_root / "testA.json")

    combined_tasks = train_tasks + val_tasks + test_tasks

    combined_tasks.sort(key=lambda x: (x["img_name"], x["mask_name"]))

    val_patients = 50
    num_images_per_patient = 4
    num_classes = 3

    val_index = val_patients * num_images_per_patient * num_classes

    val_tasks = combined_tasks[:val_index]
    train_tasks = combined_tasks[val_index:]

    with (output_dir / "train.json").open("w") as f:
        json.dump(train_tasks, f, indent=2)

    with (output_dir / "val.json").open("w") as f:
        json.dump(val_tasks, f, indent=2)

    print("Train and validation tasks saved to", output_dir)

    img_dir = Path(img_dir)
    image_paths = list(img_dir.glob(img_pattern))
    assert len(image_paths) > 0, "No images found in the image directory."

    mask_dir = Path(mask_dir)
    mask_paths = list(mask_dir.glob(mask_pattern))
    assert len(mask_paths) > 0, "No files found in the mask directory."

    assert (
        len(mask_paths) % len(image_paths) == 0
    ), "The number of masks should be the multiple of the number of images."

    assert len(image_paths) == len(
        set(path.name for path in mask_paths)
    ), "The number of images and the number of masks should be equal. "

    # Sort image and mask paths
    image_paths.sort()

    # Sort only by the last names
    mask_paths.sort(key=lambda x: x.name)

    mask_multiple = len(mask_paths) // len(image_paths)

    image_paths = [path for path in image_paths for _ in range(mask_multiple)]

    assert len(image_paths) == len(
        mask_paths
    ), "The number of images and masks differ even after adjusting them."

    image_mask_paths = tuple(zip(image_paths, mask_paths))

    raw_root = Path("/mnt/Enterprise/PUBLIC_DATASETS/camus_database/testing")

    # model = return_model()

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

    testing_json = []

    model = return_model()

    for output in tqdm(no_vqa_outputs, desc="Getting VQA"):
        meta = output["meta"]
        task = output["task"]

        vqa_prompt = get_vqa_prompt(**meta, model=model)

        task["prompts"]["p7"] = vqa_prompt

        testing_json.append(task)

    # Make json deterministic
    testing_json.sort(key=lambda x: (x["img_name"], x["mask_name"]))

    with (output_dir / "test.json").open("w") as of:
        json.dump(testing_json, of, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--camus-prev-anns-root", type=Path, required=True)
    parser.add_argument("--img-dir", type=Path, required=True)
    parser.add_argument("--img-pattern", type=str, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--mask-pattern", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=None)

    args = parser.parse_args()

    get_json_data(**vars(args))
