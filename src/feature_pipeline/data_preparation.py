import os 
from typing import List, Tuple

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Resize, RandomHorizontalFlip, RandomRotation, RandomAutocontrast 

from src.setup.paths import TRAINING_DATA_DIR, TEST_DATA_DIR


def get_classes(path: str = TRAINING_DATA_DIR) -> List: 

    """
    Each class of mushrooms is in a folder, and this 
    function will look through the subdirectories of
    the folder where the training data is kept. It 
    will then make a list of these subdirectories, 
    and return said list.

    Returns:
        List: a list of the names of the classes
              (the genera of mushrooms)
    """

    classes = []

    for root, sub_dirs, files in os.walk(path, topdown=True):
        for name in sub_dirs:
            classes.append(name)

    return classes


def set_transforms(
    rotation_degrees: int = 45,
    size: Tuple[int, int] = (128,128)
) -> Compose:

    """
    Initialise the transforms that will be used for data
    augmentation of our images.

    Returns:
        _type_: _description_
    """

    transforms = Compose([
        RandomHorizontalFlip(),
        RandomRotation(degrees=rotation_degrees),
        RandomAutocontrast(),
        ToTensor(), 
        Resize(size=size)
    ])

    return transforms


def make_dataset(
    path: str, 
    batch_size: int
) -> DataLoader:

    """
    Torchvision's ImageFolder class expects images to be in 
    directories, one for each class (which is awfully convenient).
    We will set up a Dataloader for the training, validation, and 
    testing data.

    Returns:
        DataLoader: a Dataloader object which contains the 
                    training/validation/testing data.
    """

    transforms = set_transforms()

    data = ImageFolder(root=path, transform=transforms)
    data_loader = DataLoader(dataset=training_data, shuffle=True, batch_size=batch_size)

    return data_loader
