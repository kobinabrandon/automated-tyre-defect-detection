import torch 
from pathlib import Path
from typing import Callable
from PIL.Image import Image

from transformers import AutoImageProcessor
from torch.utils.data import DataLoader,  random_split

from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Lambda, ToTensor, Resize, RandomHorizontalFlip, RandomRotation

from src.training_pipeline.models import get_model_processor
from src.setup.config import model_config, data_config, image_config 
from src.setup.paths import RAW_DATA_DIR, TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR, DATA_DIR


def prepare_data() -> tuple[DataLoader[ImageFolder], DataLoader[ImageFolder], DataLoader[ImageFolder]]:
    
    images: ImageFolder = make_full_dataset(path=RAW_DATA_DIR/data_config.file_name, augment_images=False, model_name=model_config.vit_base)
    return split_data(images=images)


def process_image(model_name: str, image: Image) -> torch.Tensor:
    processor = get_model_processor(model_name=model_name) 
    processed_image: dict[str, torch.Tensor] = processor(image, return_tensors="pt")
    return processed_image["pixel_values"].squeeze(0)


def get_custom_transforms(path: Path, new_image_size: tuple[int, int]) -> Callable[[Image], Image]:
    
    assert path in [TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR]; "Provide paths to either the training, validation, or test data" 
    if path == TRAIN_DATA_DIR:

        return Compose([
            RandomHorizontalFlip(), RandomRotation(degrees=45), ToTensor(), Resize(size=new_image_size), Lambda(lambda img: process_image(img))
        ])

    else: 
        return Compose([
            ToTensor(), Resize(size=new_image_size), Lambda(lambda img: process_image(img))
        ])


def make_full_dataset(path: Path, augment_images: bool, model_name: str | None) -> ImageFolder:
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

    if not augment_images:
        processor = get_model_processor(model_name=model_name) 

        transforms = Compose([
            Lambda(lambda img: processor(img, return_tensors="pt"))
        ])

    else:
        transforms = get_custom_transforms(path=path, new_image_size=new_size) 
        
    return ImageFolder(root=path, transform=transforms)
    

def split_data(
    images: ImageFolder, 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15, 
    batch_size: int = 10 
    ) -> tuple[DataLoader[ImageFolder], DataLoader[ImageFolder], DataLoader[ImageFolder]]:
    """

    Args:
        train_ratio (float): the fraction of the data that is to be used for training
        val_ratio (float): the fraction of the data that is to be used for validation
        dataset (Dataset): the fraction of the data that is to be used for testing
        batch_size: the size of the batches that the dataset will be divided into.
        images (ImageFolder): the full Dataset object to be divided up.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: dataloaders for each data split
    """
    number_of_images: int = len(images)

    train_size = int(train_ratio * number_of_images)
    val_size = int(val_ratio * number_of_images)
    test_size = number_of_images - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset=images, lengths=[train_size, val_size, test_size])    
    train_dataloader: DataLoader[ImageFolder] = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader: DataLoader[ImageFolder] = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader: DataLoader[ImageFolder] = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader

