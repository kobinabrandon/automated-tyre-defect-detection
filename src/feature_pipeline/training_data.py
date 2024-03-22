import os 
from typing import List 

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import Compose, ToTensor, Resize, RandomHorizontalFlip, RandomRotation, RandomAutocontrast 

from src.setup.paths import TRAINING_DATA_DIR, TEST_DATA_DIR


def get_classes(path: str = TRAINING_DATA_DIR) -> List: 

    classes = []

    for root, sub_dirs, files in os.walk(path, topdown=True):
        
        for name in sub_dirs:

            classes.append(name)

    return classes
    

def make_training_data(
    classes: List = get_classes(),
    root_path: str = TRAINING_DATA_DIR
    ) -> DataLoader:

    """
    The ImageFolder class expects images to be in directories, 
    one for each class

    Returns:
        DataLoader: a Dataloader object which contains the 
                    training data.
    """

    transforms = Compose([
        RandomHorizontalFlip(),
        RandomRotation(degrees=45),
        RandomAutocontrast(),
        ToTensor(), 
        Resize((128,128))
    ])
    
    image_folder = ImageFolder(root=root_path, transform=transforms)
    training_data = DataLoader(dataset=image_folder, shuffle=True, batch_size=3)

    return training_data
