import json
import random
from typing import Any, Dict, List, Literal, Optional, Union, get_args

import albumentations as A
import cv2
import open_clip
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from transformers import CLIPTokenizer

TOKENIZER_TYPE = Literal["biomedclip", "clip_seg"]
PROMPT_TYPE = Literal["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9"]


class ImageTextMaskDataset(Dataset):
    r"""
    Image-Text-Mask Dataset
    Args:
        tokenizer_type (TOKENIZER_TYPE): Type of tokenizer to use
        prompt_types (List[PROMPT_TYPE]): List of prompt types to use
        images_dir (str): Path to images directory
        masks_dir (str): Path to masks directory
        caps_file (Optional[str], optional): Path to captions file. Defaults to None.
        img_size (int, optional): Size of image. Defaults to 224.
        context_length (int, optional): Context length. Defaults to 77.
        transforms (Optional[A.Compose], optional): Transforms to apply to image. Defaults to None.
        override_prompt (Optional[str], optional): Text uesd for overriding prompt. Defaults to None.
        zero_prompt (bool, optional): Whether to send zero in the place of prompt. Defaults to False.
        data_num (Optional[int | float], optional): Number of data to use. For float Defaults to 1.0.

    Raises:
        TypeError: If tokenizer_type is not one of TOKENIZER_TYPE
        ValueError: If data_num is of type float and is not in range [0, 1]
    """

    def __init__(
        self,
        tokenizer_type: TOKENIZER_TYPE,
        prompt_type: PROMPT_TYPE,
        images_dir: str,
        masks_dir: str,
        caps_file: Optional[str] = None,
        img_size: int = 224,
        context_length: int = 77,
        transforms: Optional[A.Compose] = None,
        override_prompt: Optional[str] = None,
        zero_prompt: bool = False,
        data_num: Optional[int | float] = 1.0,
    ) -> None:
        super().__init__()

        self.prompt_type = prompt_type

        self.img_size = img_size
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transforms = transforms
        self.context_length = context_length
        self.data_num = data_num

        if tokenizer_type in get_args(TOKENIZER_TYPE):
            self.tokenizer_type = tokenizer_type

            if tokenizer_type == "biomedclip":
                self.tokenizer = open_clip.get_tokenizer(
                    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                ).tokenizer
            else:  # ie. tokenizer_type == "clipseg":
                self.tokenizer = CLIPTokenizer.from_pretrained(
                    "CIDAS/clipseg-rd64-refined"
                )
        else:
            raise TypeError(
                f"tokenizer_type must be one of {get_args(TOKENIZER_TYPE)} but got {tokenizer_type} instead"
            )

        self.zero_prompt = zero_prompt
        self.override_prompt = override_prompt

        with open(caps_file, "r") as fp:
            self.imgs_captions = json.load(fp)
            random.shuffle(self.imgs_captions)

        if type(self.data_num) == float:
            if self.data_num < 0 or self.data_num > 1:
                raise ValueError(
                    f"data_num must be in range [0, 1] but got {self.data_num} instead."
                )
            self.imgs_captions = self.imgs_captions[
                : int(len(self.imgs_captions) * self.data_num)
            ]
        else:
            self.imgs_captions = self.imgs_captions[: self.data_num]

        self.default_transforms = A.Compose(
            [A.Resize(height=img_size, width=img_size), A.Normalize(), ToTensorV2()]
        )

    def __len__(self):
        return len(self.imgs_captions)

    def __getitem__(self, index) -> Dict[str, Any]:
        cap = self.imgs_captions[index]

        image = cv2.imread(f"{self.images_dir}/{cap['img_name']}", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(f"{self.masks_dir}/{cap['mask_name']}", cv2.IMREAD_GRAYSCALE)

        # assert (
        #     image.shape[:2] == mask.shape[:2]
        # ), "Image and mask should have same dimensions"

        h, w = image.shape[:2]

        _, mask = cv2.threshold(mask, 127, maxval=1, type=cv2.THRESH_BINARY)

        # Use overrided prompt if provided
        if self.override_prompt:
            prompt = self.override_prompt
        else:
            if self.prompt_type == "random":
                prompt = random.choice(list(cap["prompts"].values()))
            else:
                prompt = cap["prompts"][self.prompt_type]

            if type(prompt) == list:
                prompt = random.choice(prompt)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            transformed = self.default_transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        text_enc = self.tokenizer(
            text=prompt,
            max_length=self.context_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = text_enc["input_ids"][0]
        attention_mask = text_enc["attention_mask"][0]
        return dict(
            pixel_values=image,
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask=mask,
            mask_name=cap["mask_name"],
            height=h,
            width=w,
        )
