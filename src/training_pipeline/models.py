from transformers import ViTForImageClassification, ViTHybridForImageClassification, BeitForImageClassification
from src.setup.config import model_config


def get_pretrained_model(model_name: str) -> ViTForImageClassification | ViTHybridForImageClassification | BeitForImageClassification:

    model_nicknames_and_objects = {
        "vit": ViTForImageClassification.from_pretrained(model_config.vit_base),
        "hybrid_vit": ViTHybridForImageClassification.from_pretrained(model_config.vit_hybrid)
        "beit": BeitForImageClassification.from_pretrained()
    }


    if model_name.lower() in model_nicknames_and_objects.keys(): 
        return model_nicknames_and_objects[model_name.lower()]    
    else:
        raise NotImplementedError('The named model has not been implemented. Enter "vit", "hybrid_vit", and "beit".')
