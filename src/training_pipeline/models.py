from torch.nn import (
    Module, ModuleList, Conv2d, Dropout, Sequential, ELU, ReLU, MaxPool2d, Linear, Flatten
)


class BaseCNN(Module): 

    """ Create a convolutional neural network """
    def __init__(self, num_classes: int):
        
        """     
        Initialise the attributes of the CNN.

        Args 
        - num_classes: the number of genera of the mushrooms
        """

        super().__init__()
        self.feature_extractor = Sequential(    
            Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            ELU(),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            ELU(),
            MaxPool2d(kernel_size=2),
            Flatten()
        )

        self.classifier = Linear(
            in_features=16*32*32, 
            out_features=num_classes
        )

    def forward(self, image):

        """
        Implement a forward pass, applying the extractor and 
        classifier to the input image.
        """ 

        x = self.feature_extractor(image)
        x = self.classifier(x)

        return x


class DynamicCNN(Module):

    def __init__(
        self, 
        in_channels: int, 
        num_classes: int, 
        layer_config: list[dict],
        dropout_prob: float
    ):

        super().__init__()
        self.layers = self._make_layers(in_channels, num_classes, layer_config)
        self.dropout = Dropout(dropout_prob)

    def _make_layers(self, in_channels, num_classes, layer_config) -> ModuleList:    

        layers = ModuleList()
        prev_in_channels = in_channels

        for config in layer_config:

            if config["type"] == "conv":

                layers.append(
                    Conv2d(
                        in_channels=prev_in_channels, 
                        out_channels=config["out_channels"], 
                        kernel_size=config["kernel_size"],
                        stride=config["stride"],
                        padding=config["padding"]
                    )
                )

                layers.append(ELU())
                
                prev_in_channels = config["out_channels"]
            
            elif config["type"] == "fully_connected":
 
                layers.append(
                    Linear(in_features=prev_in_channels, out_features=num_classes),
                    ReLU()
                )

            if config.get(key="pooling", default=False):    

                layers.append(
                    MaxPool2d(kernel_size=config["kernel_size"])            
                )
        
            # Ensures that the penultimate layer is for flattening 
            if layer_config.index(config) == len(layer_config)-1:
            
                layers.append(Flatten())
            
        return layers

    
    def forward(self, x):

        for layer in self.layers:
           x = layer(x)

        x = self.dropout(x)

        return x
