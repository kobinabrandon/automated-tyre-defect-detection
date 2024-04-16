"""
This module contains some toy models that I built for purposes of practice, as this is my
first project with Pytorch.
"""


from collections import OrderedDict
from torch.nn import Module, Conv2d, Dropout2d, Sequential, ELU, MaxPool2d, Linear, Flatten

from optuna import trial
from src.setup.config import settings


class BaseCNN(Module): 

    """ Create a relatively basic convolutional neural network """
    def __init__(self, num_classes: int):
        
        """     
        Args 
            num_classes: the number of genera for our mushroom 
                       classification problem
        """

        super().__init__()
        self.layers = OrderedDict(
            [
                ("conv1", Conv2d(in_channels= 3, out_channels=8, kernel_size=3, padding=1)),
                ("elu1", ELU()),
                ("pool1", MaxPool2d(kernel_size=2)),
                ("conv2", Conv2d(in_channels= 8, out_channels=16, kernel_size=3, padding=1)),
                ("elu2", ELU()),
                ("pool2", MaxPool2d(kernel_size=2)),
                ("flatten", Flatten())
            ]
        )

        self.feature_extractor = Sequential(self.layers)

        feature_extractor_output_size = calculate_feature_extractor_output_size(model_fn=self)

        self.classifier = Linear(
            in_features=feature_extractor_output_size, 
            out_features=num_classes
        )

    def _forward(self, image):

        """
        Implement a forward pass, applying the extractor 
        and classifier to the input image.
        """ 

        return self.classifier(
            self.feature_extractor(image)
        )


class BiggerCNN(Module):

    """
    A slightly bigger network than the base model which
    exists only for purposes of comparison.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        tune_hyperparams: bool,
        trial: trial.Trial|None
    ):

        super().__init__()
        
        self.dropout_rate_1 = trial.suggest_float(name="first_dropout_rate", low=0, high=0.5, step=0.1) if tune_hyperparams else 0.2
        self.dropout_rate_2 = trial.suggest_float(name="second_dropout_rate", low=0, high=0.5, step=0.1) if tune_hyperparams else 0.2
        self.dropout_rate_3 = trial.suggest_float(name="third_dropout_rate", low=0, high=0.5, step=0.1) if tune_hyperparams else 0.2

        self.layers = OrderedDict(
            [
                ("conv1", Conv2d(in_channels= 3, out_channels=32, kernel_size=3, stride=1, padding=1)),
                ("pool1", MaxPool2d(kernel_size=2)),
                ("drop1", Dropout2d(p=self.dropout_rate_1)),
                ("elu1", ELU()),
                ("conv2", Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)),
                ("pool2", MaxPool2d(kernel_size=2)),
                ("drop2", Dropout2d(p=self.dropout_rate_2)),
                ("elu2", ELU()),
                ("conv3", Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)),
                ("pool3", MaxPool2d(kernel_size=2)),
                ("drop3", Dropout2d(p=self.dropout_rate_3)),
                ("elu3", ELU()),
                ("flatten", Flatten())
            ]
        )

        self.feature_extractor = Sequential(self.layers)

        self.feature_extractor_output_size = calculate_feature_extractor_output_size(model_fn=self)

        self.classifier = Linear(in_features=self.feature_extractor_output_size, out_features=num_classes)    

    def _forward(self, image):

        return self.classifier(
            self.feature_extractor(image)
        )


class DynamicCNN(Module):

    """
    This is an experimental class that allows us to create architecturally varied CNNs. 
    The hope is that this will allow us to play with different model setups to facilitate 
    testing. 
    
    The trouble is that predictably, hyperparameter tuning of objects in this class is
    proving to be difficult for dimensional reasons that I do not yet fully understand.
    If this problem can be solved, I will be able to test a wide range of architectures
    in a relatively convenient manner.
    """

    def __init__(
        self, 
        in_channels: int, 
        num_classes: int, 
        layer_configs: list[dict],
        dropout_prob: float
    ): 

        super().__init__()

        self.layers = OrderedDict(
            self._make_layers(
                in_channels=in_channels, num_classes=num_classes, layer_configs=layer_configs, dropout_prob=dropout_prob
            )
        )

        self.feature_extractor = Sequential(self.layers)
        self.feature_extractor_output_size = calculate_feature_extractor_output_size(model_fn=self)
        self.classifier = Linear(in_features=self.feature_extractor_output_size, out_features=num_classes)


    def _make_layers(self, in_channels, num_classes, layer_configs, dropout_prob) -> list[OrderedDict]:    

        """
        Args:
            in_channels: number of input channels for a given convolutional layer

            num_classes: the number of genera for our mushroom classification problem

            layer configs: a list of dictionaries that defines the whole network. Each
                          dictionary provides information about each layer. In particular,
                          each dictionary details the type of layer and its various 
                          features. For convolutional layers, we specify the number of 
                          input channels, output channels, the size of the filter/kernel, 
                          and finally, the stride and padding parameters.
        """

        layers = []

        prev_out_channels = in_channels
        conv_count = 0
        pool_count = 0

        for layer_config in layer_configs:

            if layer_config["type"] == "conv":

                conv_count+=1

                layers.append(

                    (
                        f"conv{conv_count}", Conv2d(
                                in_channels=prev_out_channels, 
                                out_channels=layer_config["out_channels"], 
                                kernel_size=layer_config["kernel_size"],
                                stride=layer_config["stride"],
                                padding=layer_config["padding"]
                            )
                    )
                    
                )

                layers.append(
                    (f"drop{conv_count}", Dropout2d(p=dropout_prob))
                )

                layers.append(
                    (f"elu{conv_count}", ELU())
                )
                
                prev_out_channels = layer_config["out_channels"]

            if layer_config.get("pooling", False):    

                pool_count+=1

                layers.append(
                    (f"pool{pool_count}", MaxPool2d(kernel_size=2))           
                )

                layers.append(
                    (f"elu{pool_count}", ELU())
                )
                
            # Ensures that the penultimate layer is for flattening 
            if layer_configs.index(layer_config) == len(layer_configs)-1:
            
                layers.append(
                    ("flatten", Flatten())
                )
            
        return layers

    
    def _forward(self, image):

        return  self.classifier(
            self.feature_extractor(image)
        )


def calculate_feature_extractor_output_size(
    model_fn: BaseCNN|BiggerCNN|DynamicCNN,
    image_resolution: tuple[int] = (settings.resized_image_width,settings.resized_image_height)
    ) -> int:

    """
    We compute the output size of the feature extractor, which is required by the 
    linear layers that we are using as classifiers at the end of each model.

    This output size is determined by the dimensions of the input images, the 
    number of feature maps of any convolutional layers, and the number of pooling
    layers used in the network.

    Returns:
        int: the size of the output produced by the given network's feature extractor
    """

    pooling_layer_count = 0
    output_feature_maps = 0
    
    layers = list(model_fn.layers.values())

    for layer in layers:
            
        if isinstance(layer, Conv2d):

            output_feature_maps = layer.out_channels

        if isinstance(layer, MaxPool2d):

            pooling_layer_count+=1

    reduced_width = image_resolution[0]/(2**pooling_layer_count)
    reduced_height = image_resolution[1]/(2**pooling_layer_count)

    feature_extractor_output_size = int(output_feature_maps*reduced_width*reduced_height)

    return feature_extractor_output_size
    