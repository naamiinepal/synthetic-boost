from typing import Any, Dict, Optional
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
import os

class ImageMaskDataset(Dataset):
    r"""
    Image-Mask Dataset
    Args:
        tokenizer_type (TOKENIZER_TYPE): Type of tokenizer to use
        prompt_types (List[PROMPT_TYPE]): List of prompt types to use
        images_dir (str): Path to images directory
        masks_dir (str): Path to masks directory
        img_size (int, optional): Size of image. Defaults to 224.
        transforms (Optional[A.Compose], optional): Transforms to apply to image. Defaults to None.
        data_num (Optional[int | float], optional): Number of data to use. For float Defaults to 1.0.

    Raises:
        ValueError: If data_num is of type float and is not in range [0, 1]
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        img_size: int = 224,
        transforms: Optional[A.Compose] = None,
        grayscale: bool = False,
    ) -> None:
        super().__init__()

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.transforms = transforms
        self.grayscale = grayscale

        self.images = os.listdir(images_dir)
        if self.grayscale:
            self.default_transforms = A.Compose(
                [
                    A.Resize(height=img_size, width=img_size),
                    A.Normalize(),
                    A.ToGray(always_apply=True),
                    ToTensorV2(),
                ]
            )
        else:
            self.default_transforms = A.Compose(
                [A.Resize(height=img_size, width=img_size), A.Normalize(), ToTensorV2()]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index) -> Dict[str, Any]:
        
        file_name = self.images[index]
        
        image = cv2.imread(f"{self.images_dir}/{file_name}", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Since, mask and image have same file name
        mask = cv2.imread(f"{self.masks_dir}/{file_name}", cv2.IMREAD_GRAYSCALE)

        # assert (
        #     image.shape[:2] == mask.shape[:2]
        # ), "Image and mask should have same dimensions"

        h, w = image.shape[:2]

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            transformed = self.default_transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        return dict(
            pixel_values=image,
            mask=mask,
            mask_name=file_name,
            height=h,
            width=w,
        )


# name main
if __name__ == "__main__":
    ds = ImageMaskDataset(
        images_dir="/mnt/Enterprise/PUBLIC_DATASETS/SDM_generated_data/2_chamber_end_diastole/images/training/",
        masks_dir="/mnt/Enterprise/PUBLIC_DATASETS/SDM_generated_data/2_chamber_end_diastole/annotations/training/"
    )
    ds[0]