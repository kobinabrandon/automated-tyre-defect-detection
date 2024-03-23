from loguru import logger 
from tqdm import tqdm 

from torch.nn import Module, Conv2d, Sequential, ELU, MaxPool2d, Linear, CrossEntropyLoss, Flatten
from torch.optim import Adam, SGD, RMSprop
from torch.optim.optimizer import Optimizer
from torch import cuda, device 

from src.setup.paths import TRAINING_DATA_DIR
from src.feature_pipeline.training_data import make_dataset, get_classes


def get_optimizer(
    model: CNN,
    optimizer: str,
    learning_rate: float
) -> Optimizer:

    """
    The function returns the required optimizer function, based on
    the entered specifications.

    Args: 
        Model: the model that is being trained

        optimizer: the function that will be used to search 
                   for the global minimum of the loss function.

        learning_rate: the learning rate that is optimizer is using for 
                        its search.
                        
    Raises:
        NotImplementedError: The requested optimizer has not been implemented

    Returns:
        Optimizer: the optimizer that will be returned.
    """

    opts_and_possible_names = {
        ("adam", "Adam"): Adam(params=model.parameters(), lr=learning_rate),
        ("sgd", "SGD"): SGD(params=model.parameters(), lr=learning_rate),
        ("rmsprop", "RMSprop"): RMSprop(params=model.parameters(), lr=learning_rate)
    }

    optimizers_and_names = {
        name: optimizer for names, optimizers in opts_and_possible_names.items() for name in names
    }

    if model in optimizers_and_names.keys():

        return optimizers_and_names[model]

    else:
        raise NotImplementedError("Consider using the Adam, SGD, or RMSprop optimizers")


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

        """igni
        Implement a forward pass, applying the extractor and 
        classifier to the input image.
        """ 

        x = self.feature_extractor(image)
        x = self.classifier(x)

        return x


def choose_training_device(model: CNN):

    """
    Check whether a GPU is available. If it is, use it.
    Otherwise, default to using the CPU.
    """

    if cuda.is_available():
        device = device("cuda")

    else:
        device = device("cpu")

    model.to(device)


def train(
    batch_size: int,
    learning_rate: int,
    num_epochs: int,
    optimizer: str,
    device: str
):
    
    classes = get_classes()

    logger.info("Setting up neural network")
    model = CNN(num_classes=len(classes))
    criterion = CrossEntropyLoss()
    
    optimizer = get_optimizer(
        model=model, 
        learning_rate=learning_rate,
        optimizer=optimizer
        )

    logger.info("Collecting training data")
    train_loader = make_dataset(path=TRAINING_DATA_DIR, batch_size=batch_size)

    logger.info(f"Training the network via {device}") 
    for epoch in tqdm(range(num_epochs)):

        training_loss = 0.0
        val_loss = 0.0

        model.train()

        for batch in train_loader:

            optimizer.zero_grad()
            images, classes = batch

            images = images.to(device)
            classes = classes.to(device)

            outputs = model(images)
            loss = criterion(outputs, classes)

            loss.backward()
            optimizer.step()
    
    logger.info("Finished Training")
