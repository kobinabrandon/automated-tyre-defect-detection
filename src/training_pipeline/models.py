from transformers.models.vit import ViTImageProcessorFast
from transformers.models.beit import BeitImageProcessor 
from transformers import ViTForImageClassification, ViTHybridForImageClassification, BeitForImageClassification 
from src.setup.config import model_config


def get_pretrained_model(model_name: str) -> ViTForImageClassification | ViTHybridForImageClassification | BeitForImageClassification:
    """
    After specifying the pretrained model of choice, return the named model 

    Args:
        model_name: the name of the pretrained model 

    Returns:
        ViTForImageClassification | ViTHybridForImageClassification | BeitForImageClassification: the named model

    Raises:
        NotImplementedError: the named model is not being considered. 
    """
    model_nicknames_and_objects = {
        model_config.vit_hybrid: ViTHybridForImageClassification.from_pretrained(model_config.vit_hybrid),
        model_config.vit_base: ViTForImageClassification.from_pretrained(model_config.vit_base),
        model_config.beit_base: BeitForImageClassification.from_pretrained(model_config.beit_base)
    }

    if model_name.lower() in model_nicknames_and_objects.keys(): 
        return model_nicknames_and_objects[model_name.lower()]    
    else:
        raise NotImplementedError('The named model has not been implemented. Enter "vit", "hybrid_vit", and "beit".')


def get_image_processor(model_name: str) -> ViTImageProcessorFast | BeitImageProcessor:
    """
    Load the image processor associated with the named Hugging Face model 
    
    Args:
        model_name: the name of the model being requested. 

    Returns:
        ViTImageProcessorFast | BeitImageProcessor: the requested processor
    """
    model_nicknames_and_processors = {
        model_config.vit_hybrid: ViTImageProcessorFast.from_pretrained(model_config.vit_hybrid), 
        model_config.vit_base: ViTImageProcessorFast.from_pretrained(model_config.vit_base),
        model_config.beit_base: BeitImageProcessor.from_pretrained(model_config.beit_base)
    }
    
    if model_name.lower() in model_nicknames_and_processors.keys():
        return model_nicknames_and_processors[model_name.lower()]

