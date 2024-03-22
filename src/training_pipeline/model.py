from loguru import logger 

from torch.nn import Module, Conv2d, Sequential, ELU, MaxPool2d, Linear, CrossEntropyLoss, Flatten
from torch.optim import Adam

from src.setup.paths import TRAINING_DATA_DIR
from src.feature_pipeline.training_data import make_training_data, get_classes


class CNN(Module):

    """ Create a convolutional neural network """
    def __init__(self, num_classes: int):
        
        """     
        Initialise the attributes of the CNN.

        Args 
        - num_classes: the number of genera of the mushrooms
        """

        super().__init__()
        self.feature_extractor = Sequential(    
            Conv2d(3, 8, kernel_size=3, padding=1),
            ELU(),
            MaxPool2d(kernel_size=2),
            Conv2d(8, 16, kernel_size=3, padding=1),
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


def train(
    model: CNN,
    num_classes: int,
    learning_rate: int,
    num_epochs: int
):
    
    logger.info("Setting up neural network")

    model = CNN(num_classes=12)
    classes = get_classes()
    criterion = CrossEntropyLoss()

    optimizer = Adam(
        params=model.parameters(), 
        lr=learning_rate
    )
    
    training_data = make_training_data()
    logger.info("Training the network")
    
    for epoch in range(num_epochs):

        for images, classes in training_data:

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, classes)

            loss.backward()
            optimizer.step()
    
    logger.info("Finished Training")


train(
    model=CNN(num_classes=12),
    num_classes=12,
    learning_rate=0.01,
    num_epochs=10
)
