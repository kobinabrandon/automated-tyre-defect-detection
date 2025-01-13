from transformers import ViTForImageClassification, ViTHybridForImageClassification, BeitForImageClassification


def get_pretrained_model(model_name: str) -> ViTForImageClassification | ViTHybridForImageClassification | BeitForImageClassification:

    if model_name.lower() == "vit":
        return ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    elif model_name.lower() == "hybrid_vit":
        return ViTHybridForImageClassification.from_pretrained("googlevit-hybrid-base-bit-384")
    elif model_name.lower() == "beit":
        return BeitForImageClassification.from_pretrained()
    else: 
        raise NotImplementedError('The named model has not been implemented. Enter "vit", "hybrid_vit", and "beit".')
