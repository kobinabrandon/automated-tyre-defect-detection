import os 

from pathlib import Path

from PIL.Image import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from transformers.models.auto import AutoFeatureExtractor
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
    def __init__(self, image_folder: ImageFolder, transformer):
        self.image_folder = image_folder
        self.transformer = transformer

    def __getitem__(self, image_index: int):
        image, label = self.image_folder[image_index]
        transformed_image = self.transformer(image)
        return transformed_image, label


def transform_for_pretrained_model(model_code: str, image: Image):

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        pretrained_model_name_or_path="google/vit-base-patch16-224"
    )

    return feature_extractor(image)
    

def make_dataset(path: Path, batch_size: int, pretrained: bool) -> DataLoader:

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
        
        batch_size: the size of the batches that the dataset will be
                    divided into.

    Returns:
        DataLoader: a Dataloader object which contains the training/validation/testing data.
    """
    new_size = (settings.resized_image_width, settings.resized_image_height)
 
    if pretrained:
        dataset = ImageFolder(root=path)
        transformed_dataset = DatasetForHuggingFace(
            image_folder=dataset, transformer=transform_for_pretrained_model
        )

        data_loader = DataLoader(dataset=transformed_dataset, batch_size=batch_size, shuffle=True)
        
    else:
        if path == TRAIN_DATA_DIR:
            transforms = [
                RandomHorizontalFlip(), RandomRotation(degrees=45), 
                RandomAutocontrast(), ToTensor(), Resize(size=size)
            ]
        elif path == VAL_DATA_DIR or path == TEST_DATA_DIR:
            transforms = [ToTensor(), Resize(size=new_size)]

        composed_transforms = Compose(transforms=transforms)
        dataset = ImageFolder(root=path, transform=composed_transforms)
        data_loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)
    return data_loader
