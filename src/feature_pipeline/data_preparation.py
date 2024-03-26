import os 

from pathlib import PosixPath
from typing import List, Tuple

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Resize, RandomHorizontalFlip, RandomRotation, RandomAutocontrast 

from src.setup.paths import TRAINING_DATA_DIR, VALIDATION_DATA_DIR, TEST_DATA_DIR


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


def make_dataset(
    path: PosixPath, 
    batch_size: int
) -> DataLoader:

    """
    Initialise the transforms that will be used for data
    augmentation of our images. The exact transforms
    that will be used depend on whether the model is 
    being trained, validated during training, or 
    tested after training.

    Torchvision's ImageFolder class expects images to be in 
    directories, one for each class (which is awfully convenient).
    We can set up a Dataloader for the training, validation, and 
    testing data.

    Args:
        path: the location of the folder containing the images
              This will determine which transforms will be applied
        
        batch_size: the size of the batches that the dataset will
                    be divided into.

    Returns:
        DataLoader: a Dataloader object which contains the 
                    training/validation/testing data.
    """

    # Initialise the image transformations
    if path == TRAINING_DATA_DIR:

        transforms = Compose([
            RandomHorizontalFlip(),
            RandomRotation(degrees=45),
            RandomAutocontrast(),
            ToTensor(), 
            Resize(size=(128,128))
        ])

    if path == VALIDATION_DATA_DIR or path == TEST_DATA_DIR:

        transforms = Compose([
            ToTensor(),
            Resize(size=(128,128))
        ])
    

    data = ImageFolder(root=path, transform=transforms)
    data_loader = DataLoader(dataset=data, shuffle=True, batch_size=batch_size)

    return data_loader
