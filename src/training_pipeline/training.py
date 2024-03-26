import torch 

from loguru import logger 
from tqdm import tqdm 

from torch.nn import (
    Module, Conv2d, Sequential, ELU, MaxPool2d, Linear, CrossEntropyLoss, Flatten
)

from torch.optim import Adam, SGD, RMSprop
from torch.optim.optimizer import Optimizer 

from torchmetrics.classification import (
    MulticlassPrecision, MulticlassAccuracy, MulticlassRecall, MulticlassConfusionMatrix
)

from src.setup.paths import TRAINING_DATA_DIR, VALIDATION_DATA_DIR, MODELS_DIR
from src.feature_pipeline.data_preparation import make_dataset, get_classes


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

    optimizers_and_likely_spellings = {
        ("adam", "Adam"): Adam(params=model.parameters(), lr=learning_rate),
        ("sgd", "SGD"): SGD(params=model.parameters(), lr=learning_rate),
        ("rmsprop", "RMSprop"): RMSprop(params=model.parameters(), lr=learning_rate)
    }

    optimizer_for_each_spelling = {
        spelling: function for spellings, function in 
        optimizers_and_likely_spellings.items() for spelling in spellings
    }

    if optimizer in optimizer_for_each_spelling.keys():

        return optimizer_for_each_spelling[optimizer]

    else:
        raise NotImplementedError("Consider using the Adam, SGD, or RMSprop optimizers")


def set_training_device(model: CNN):

    """
    Check whether a GPU is available. If it is, use it.
    Otherwise, default to using the CPU.
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    model.to(device)


def train(
    batch_size: int,
    learning_rate: int,
    num_epochs: int,
    optimizer: str,
    device: str,
    save: bool
):
    
    classes = get_classes()

    # Prepare metrics
    precision = MulticlassPrecision(num_classes=len(classes), average="macro")
    recall = MulticlassRecall(num_classes=len(classes), average="macro")
    accuracy = MulticlassAccuracy(num_classes=len(classes), average="macro")

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
    train_iterator = iter(train_loader)

    logger.info(f"Setting training device to {device}")
    set_training_device(model=model)

    logger.info(f"Training...") 
    for epoch in tqdm(iterable=range(num_epochs)):

        training_loss = 0.0
        val_loss = 0.0
        model.train()

        for batch in train_loader:

            # Refresh gradients
            optimizer.zero_grad()
            images, label = batch

            images = images.to(device)
            label = label.to(device)
            
            output = model(images)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        training_loss /= len(train_iterator)

        # Set the model in evaluation mode
        model.eval()

        # Get validation data
        val_data_loader = make_dataset(path=VALIDATION_DATA_DIR, batch_size=batch_size)
        val_iterator = iter(val_data_loader)

        with torch.no_grad():
            
            for batch in val_data_loader:

                images, label = batch

                images = images.to(device)
                label = label.to(device)

                ouput = model(images)
                val_loss = criterion(output, label)
                
                _ , predictions = torch.max(input=outputs, dim=1)

                val_recall = recall(predictions, label)
                val_accuracy = accuracy(predictions, label)
                val_precision = precision(predictions, label)

                for metric in [val_recall, val_accuracy, val_precision]:

                    metric.compute()

                val_loss = loss.item()
            
            val_loss /= len(val_iterator)

            logger.success(
                "Epoch: {}, Training Loss: {:.2f}, Validation_loss: {:.2f}, \
                Accuracy: {:.2f}, Recall: {:.2f}, Precision: {:.2f} \
                ".format(epoch, training_loss, val_loss, val_accuracy, val_recall, val_precision)
            )

    if save:

        torch.save(model, MODELS_DIR)
    
    logger.info("Finished Training")



train(
    batch_size=20,
    learning_rate=0.01,
    num_epochs=10,
    optimizer="adam",
    device="cpu",
    save=True
)