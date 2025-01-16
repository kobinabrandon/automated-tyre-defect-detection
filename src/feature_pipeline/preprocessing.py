from pathlib import Path
from loguru import logger 
from typing import Callable
from PIL.Image import Image

from torch.utils.data import DataLoader,  random_split

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Lambda, ToTensor, Resize, RandomHorizontalFlip, RandomRotation

from src.setup.paths import RAW_DATA_DIR
from src.setup.config import process_config, data_config, image_config 
from src.training_pipeline.models import get_image_processor


def prepare_images(augment_images: bool, model_name: str, path: Path = RAW_DATA_DIR/data_config.file_name) -> ImageFolder:
    """
    Initialise the transforms that will be used for data augmentation of our images. The exact transforms that will 
    be used depend on whether the model is being trained, validated during training, or tested after training.

    Torchvision's ImageFolder class expects images to be in directories, one for each class (which is awfully convenient).
    We can set up a Dataloader for the training, validation, and testing data.

    Args:
        path: the location of the folder containing the images. This will determine which transforms will be applied

    Returns:
        DataLoader: a Dataloader object which contains the training/validation/testing data.
    """
    new_size = (image_config.resized_image_width, image_config.resized_image_height)
    processor = get_image_processor(model_name=model_name) 

    if not augment_images:

        transforms = Compose([
            Lambda(
                lambda img: processor(img, return_tensors="pt")
            )
        ])

    else:
        transforms = Compose(
            [
                RandomHorizontalFlip(), RandomRotation(degrees=45), ToTensor(), Resize(size=new_image_size), 
                Lambda(
                    lambda img: processor(img, return_tensors="pt")
                )
            ]
        )

    return ImageFolder(root=path, transform=transforms)
    

def split_data(
    images: ImageFolder, 
    train_ratio: float = process_config.train_ratio, 
    val_ratio: float = process_config.val_ratio, 
    batch_size: int = process_config.batch_size 
    ) -> tuple[DataLoader[ImageFolder], DataLoader[ImageFolder], DataLoader[ImageFolder]]:
    """
    Perform train-validation-test splits and return the dataloaders for each split. 

    Args:
        images (ImageFolder): the full Dataset object to be divided up.
        train_ratio (float): the fraction of the data that is to be used for training
        val_ratio (float): the fraction of the data that is to be used for validation
        batch_size: the size of the batches that the dataset will be divided into.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: dataloaders for each data split
    """
    number_of_images: int = len(images)

    train_size = int(train_ratio * number_of_images)
    val_size = int(val_ratio * number_of_images)
    test_size = number_of_images - train_size - val_size

    logger.info("Performing train-validation-test split") 
    train_dataset, val_dataset, test_dataset = random_split(dataset=images, lengths=[train_size, val_size, test_size])    
    train_dataloader: DataLoader[ImageFolder] = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader: DataLoader[ImageFolder] = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader: DataLoader[ImageFolder] = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader


# def get_custom_transforms(path: Path, model_name: str, new_image_size: tuple[int, int]) -> Callable[[Image], Image]:
#     """
#     Perform selected transformations, and chain them, before applying the processor that corresponds  
#
#     Args:
#         path: 
#         new_image_size: 
#
#     Returns:
#
#     """
#     assert path in [TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR]; "Provide paths to either the training, validation, or test data" 
#
#     if path == TRAIN_DATA_DIR:
#
#         return Compose([
#             RandomHorizontalFlip(), RandomRotation(degrees=45), ToTensor(), Resize(size=new_image_size), Lambda(lambda img: process_image(img))
#         ])
#
#     else: 
#         return Compose([
#             ToTensor(), Resize(size=new_image_size), Lambda(lambda img: process_image(img))
#         ])
#

# def process_image(model_name: str, image: torch.Tensor) -> torch.Tensor:
#     processor = get_image_processor(model_name=model_name) 
#     processed_image: dict[str, torch.Tensor] = processor(image, return_tensors="pt")
#     breakpoint()
#     return processed_image["pixel_values"].squeeze(0)
#

