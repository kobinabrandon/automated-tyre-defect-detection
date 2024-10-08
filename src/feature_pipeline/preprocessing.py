import os 

from pathlib import Path
from PIL.Image import Image
from torchvision.datasets import ImageFolder
from transformers.models.auto import AutoFeatureExtractor
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose, ToTensor, Resize, RandomHorizontalFlip, RandomRotation, RandomAutocontrast

from src.setup.config import settings
from src.setup.paths import TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR, DATA_DIR

    
def get_num_classes(path: str = DATA_DIR) -> int: 
    """
    Each class of mushrooms is in a folder, and this function 
    will look through the subdirectories of the folder where 
    the training data is kept. It will then make a list of 
    these subdirectories, and return the length of said list.

    Returns:
        int: the length of the list of classes (the genera of mushrooms)
    """
    classes = [name for root, sub_dirs, files in os.walk(path, topdown=True) for name in sub_dirs] 
    return len(classes)


class DatasetForHuggingFace(Dataset):
    def __init__(self, image_folder: ImageFolder, transform_fn: callable) -> None:
        self.image_folder = image_folder
        self.transform_fn = transform_fn

    def __getitem__(self, image_index: int):
        image, label = self.image_folder[image_index]
        transformed_image = self.transform_fn(image)
        return transformed_image, label


def pretrained_transform_fn(image: Image, model_code: str = "google/vit-base-patch16-224"):
    feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path=model_code)
    return feature_extractor(image)
    

def make_full_dataset(path: Path, pretrained: bool) -> Dataset:
    """
    Initialise the transforms that will be used for data
    augmentation of our images. The exact transforms that will 
    be used depend on whether the model is being trained, validated 
    during training, or tested after training.

    Torchvision's ImageFolder class expects images to be in 
    directories, one for each class (which is awfully convenient).
    We can set up a Dataloader for the training, validation, and 
    testing data.

    Args:
        path: the location of the folder containing the images. This
              will determine which transforms will be applied

    Returns:
        DataLoader: a Dataloader object which contains the training/validation/testing data.
    """
    new_size = (settings.resized_image_width, settings.resized_image_height)
 
    if pretrained:
        dataset = DatasetForHuggingFace(image_folder=ImageFolder(root=path), transform_fn=pretrained_transform_fn)
    else:
        if path == TRAIN_DATA_DIR:
            transforms = [
                RandomHorizontalFlip(), RandomRotation(degrees=45), 
                RandomAutocontrast(), ToTensor(), Resize(size=new_size)
            ]
        elif path == VAL_DATA_DIR or path == TEST_DATA_DIR:
            transforms = [ToTensor(), Resize(size=new_size)]

        composed_transforms = Compose(transforms=transforms)
        dataset = ImageFolder(root=path, transform=composed_transforms)
    return dataset


def split_data(train_ratio: float, val_ratio: float, batch_size: int, dataset: Dataset) -> tuple[DataLoader, DataLoader, DataLoader]:
    """

    Args:
        train_ratio (float): the fraction of the data that is to be used for training
        val_ratio (float): the fraction of the data that is to be used for validation
        dataset (Dataset): the fraction of the data that is to be used for testing
        batch_size: the size of the batches that the dataset will be divided into.
        dataset (Dataset): the full Dataset object to be divided up.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: dataloaders for each data split
    """
    number_of_images = len(dataset)

    train_size = int(train_ratio(number_of_images))
    val_size = int(val_ratio(number_of_images))
    test_size = number_of_images - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_size, val_size, test_size])    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader
