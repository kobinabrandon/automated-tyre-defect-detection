"""
This module contains the various models that I have in my attempts to solve 
this classification problem. They range from simple toy networks to slightly 
bigger ones, to famous architectures such as ResNet. 

I did much of this work to improve my skills with pytorch, as this is my
first project with Pytorch.
"""


from collections import OrderedDict
from torch import Tensor, reshape
from torch.nn import Module, Conv2d, BatchNorm2d, Dropout2d, Sequential, ReLU, MaxPool2d, Linear, Flatten, ModuleList, AdaptiveAvgPool2d

from optuna import trial
from src.setup.config import settings
from src.feature_pipeline.data_preparation import get_num_classes


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
                ("relu1", ReLU()),
                ("pool1", MaxPool2d(kernel_size=2)),
                ("conv2", Conv2d(in_channels= 8, out_channels=16, kernel_size=3, padding=1)),
                ("relu2", ReLU()),
                ("pool2", MaxPool2d(kernel_size=2)),
                ("flatten", Flatten())
            ]
        )

        self.feature_extractor = Sequential(self.layers)

        feature_extractor_output_size = calculation_output_feature_map_size(model_fn=self)

        self.classifier = Linear(
            in_features=feature_extractor_output_size, 
            out_features=num_classes
        )

    def forward(self, image):

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
                ("relu1", ReLU()),
                ("conv2", Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)),
                ("pool2", MaxPool2d(kernel_size=2)),
                ("drop2", Dropout2d(p=self.dropout_rate_2)),
                ("relu2", ReLU()),
                ("conv3", Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)),
                ("pool3", MaxPool2d(kernel_size=2)),
                ("drop3", Dropout2d(p=self.dropout_rate_3)),
                ("relu3", ReLU()),
                ("flatten", Flatten())
            ]
        )

        self.feature_extractor = Sequential(self.layers)

        self.feature_extractor_output_size = calculation_output_feature_map_size(model_fn=self)

        self.classifier = Linear(in_features=self.feature_extractor_output_size, out_features=num_classes)    

    def forward(self, image):

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

        super(DynamicCNN, self).__init__()

        self.layers = OrderedDict(
            self._make_layers(
                in_channels=in_channels, num_classes=num_classes, layer_configs=layer_configs, dropout_prob=dropout_prob
            )
        )

        self.feature_extractor = Sequential(self.layers)
        self.feature_extractor_output_size = calculation_output_feature_map_size(model_fn=self)
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
                    (f"ReLU{conv_count}", ReLU())
                )
                
                prev_out_channels = layer_config["out_channels"]

            if layer_config.get("pooling", False):    

                pool_count+=1

                layers.append(
                    (f"pool{pool_count}", MaxPool2d(kernel_size=2))           
                )

                layers.append(
                    (f"ReLU{pool_count}", ReLU())
                )
                
            # Ensures that the penultimate layer is for flattening 
            if layer_configs.index(layer_config) == len(layer_configs)-1:
            
                layers.append(
                    ("flatten", Flatten())
                )
            
        return layers

    
    def forward(self, image):

        return  self.classifier(
            self.feature_extractor(image)
        )


def calculation_output_feature_map_size(
    model_fn: BaseCNN|BiggerCNN|DynamicCNN,
    image_resolution: tuple[int] = (settings.resized_image_width,settings.resized_image_height)
    ) -> int:

    """
    We compute the size of the output feature map, which is required by the 
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
    

class ConvBlock(Module):

    """
    This class marks the beginning of my attempts to understand and build models that conform to the 
    ResNet (residual network) architecture.

    ResNet models make use of convolutional layers whose outputs are stabilised using batch normalisation.
    This occurs so frequently in these models that it seems justified to have a class dedicated to this 
    combination of layers to reduce code duplication, and improve readability.
    """

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int, 
        stride: int, 
        padding: int
    ):
        super().__init__()
        self.conv = Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            BatchNorm2d(num_features=out_channels)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        return self.conv.forward(x)


class ResidualBlock(Module):

    def __init__(
        self, 
        in_channels: int,
        out_channels: int, 
        stride=1, 
        shortcut_downsample=Sequential|None
    ):
        super().__init__()
        
        self.expansion = 4
        self.elements = Sequential(
            ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=0),
            ConvBlock(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            ConvBlock(in_channels=out_channels, out_channels=out_channels*self.expansion, kernel_size=1, stride=1, padding=0),
            ReLU()
        )

        self.shortcut_downsample = shortcut_downsample
    

    def forward(self, x:Tensor) -> Tensor:

        # Copy the input data
        residual = x

        # Apply the residual block to the input to get the output feature maps
        x = self.elements(x)
        
        if self.shortcut_downsample is not None:
            residual = self.shortcut_downsample(residual)

        # Add the (potentially downsampled) input data to the output feature maps, and return the result
        x += residual
        return self.relu(x)


class ResNet(Module):

    """
    This is a general class that should allow us to create different ResNet 
    model arhitectures.
    
    Its constructor initialises all the layers of the model, and a function follows
    which implements the forward pass. 

    Though ResNet models can be imported from Pytorch hub, I wanted to build
    them myself to improve my skills with pytorch, and learn more about this model
    architecture.
    """

    def __init__(
        self, 
        blocks_per_layer: list[int], 
        in_channels: int, 
        num_classes: int 
    ):

        """
        Initialise the various layers that will make up the residual network.

        Args: 
            layers: a list containing the number of residual blocks in each layer

            in_channels: the number of input channels in the first convolutional layer

            num_classes: the number of genera for our mushroom classification problem
        """

        super().__init__()
        self.in_channels = 3
        self.layers = []

        initial_layers = [
            ConvBlock(in_channels=self.in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
            ReLU()
        ]

        for layer in initial_layers:
            self.layers.append(layer)

        for index, num_blocks in enumerate(blocks_per_layer):

            stride = 1 if index == 0 else 2

            self.layers.extend(
                self._make_layer(
                    num_residual_blocks=num_blocks, 
                    out_channels=64*(2**index), 
                    stride=stride
                )
            )
        
        self.layers.append(
            AdaptiveAvgPool2d(output_size=1)
        )

        blocks_last_layer = len(blocks_per_layer)-1
        fc_input_features = 64*(2**blocks_last_layer)*4

        self.fully_connected = Linear(in_features=fc_input_features, out_features=num_classes)

    
    def forward(self, x: Tensor) -> Tensor:

        """
        Perform the forward pass for each layer in the layers object.
        Then run the output of that into a fully connected layer.

        Returns:
            x: a tensor containing the output of the ResNet model. 
        """

        for layer in self.layers:
            x = layer.forward(x)

        return self.fully_connected.forward(
            reshape(
                input=x, 
                shape=(x.shape[0], -1)
            )
        )

    
    def _make_layer(
        self, 
        num_residual_blocks: int, 
        out_channels:int, 
        stride:int  
        ) -> list:

        """

        Args:
            num_residual_blocks: the number of residual blocks that make up the layer.
            
            out_channels: the number of filters to use for the convolutions.

            stride: the stride to be used for the convolutions when downsampling is required.

        Returns:
            Sequential[ModuleList]: the initialised layers of the residual network
        """

        shortcut_downsample = None
        layers = []

        # Conditions under which the inputs and outputs of a residual block will mismatch.
        if stride !=1 or self.in_channels != out_channels*4:
            
            shortcut_downsample = Sequential(
                ConvBlock(
                    in_channels=3, 
                    out_channels=out_channels*4, 
                    kernel_size=1,
                    padding=1,
                    stride=stride
                )
            )

        layers.append(
            ResidualBlock(
                in_channels=self.in_channels,
                out_channels=out_channels,
                shortcut_downsample=shortcut_downsample,
                stride=stride
            )
        )
        
        self.in_channels = out_channels*4
 
        for _ in range(num_residual_blocks-1):
            
            # Establish the residual blocks that make up the layer
            layers.append(
                ResidualBlock(
                    in_channels=3, 
                    out_channels=out_channels, 
                    shortcut_downsample=shortcut_downsample
                )
            )

        return layers


def get_resnet(
    model_name: str, 
    num_classes: int = get_num_classes()
    ) -> ResNet:

    """
    Accept the name of a particular ResNet architecture, and return it.

    Returns:
        ResNet: the model with the requested ResNet architecture.
    """

    if model_name in ["resnet50", "Resnet50" "ResNet50"]:
        model_fn = ResNet(in_channels=3, blocks_per_layer=[3,4,6,3], num_classes=num_classes)

    elif model_name in ["resnet101", "Resnet101" "ResNet101"]:
        model_fn = ResNet(in_channels=3, blocks_per_layer=[3,4,24,3], num_classes=num_classes)

    elif model_name in ["resnet152", "Resnet152", "ResNet152"]:
        model_fn = ResNet(in_channels=3, blocks_per_layer=[3,8,36,3], num_classes=num_classes)

    return model_fn
    