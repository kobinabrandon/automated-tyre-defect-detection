from transformers.models.vit import ViTImageProcessorFast
from transformers import ViTForImageClassification, ViTHybridForImageClassification, BeitForImageClassification 
from src.setup.config import model_config


def get_pretrained_model(model_name: str) -> ViTForImageClassification | ViTHybridForImageClassification | BeitForImageClassification:

    model_nicknames_and_objects = {
        model_config.vit_hybrid: ViTHybridForImageClassification.from_pretrained(model_config.vit_hybrid),
        model_config.vit_base: ViTForImageClassification.from_pretrained(model_config.vit_base),
        model_config.beit_base: BeitForImageClassification.from_pretrained(model_config.beit_base)
    }

    if model_name.lower() in model_nicknames_and_objects.keys(): 
        return model_nicknames_and_objects[model_name.lower()]    
    else:
        raise NotImplementedError('The named model has not been implemented. Enter "vit", "hybrid_vit", and "beit".')


def get_model_processor(model_name: str) -> ViTImageProcessorFast:
    
    assert model_name.lower() in [model_config.vit_base, model_config.vit_hybrid, "beit"]
    return ViTImageProcessorFast.from_pretrained(model_name) 

