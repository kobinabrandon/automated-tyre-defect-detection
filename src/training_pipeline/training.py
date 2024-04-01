import os
import torch

from loguru import logger 
from tqdm import tqdm 
from argparse import ArgumentParser
from comet_ml import Experiment

from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD, RMSprop
from torch.optim.optimizer import Optimizer 

from torchmetrics.classification import (
    MulticlassPrecision, MulticlassAccuracy, MulticlassRecall
)

from src.setup.paths import TRAINING_DATA_DIR, VALIDATION_DATA_DIR, MODELS_DIR
from src.feature_pipeline.data_preparation import make_dataset, get_classes
from src.training_pipeline.models import BaseCNN, DynamicCNN
from src.training_pipeline.hyperparameter_tuning import optimise_hyperparams


def get_model(model: str) -> BaseCNN|DynamicCNN:

    """
    Takes either the string "base" or "dynamic" and returns 
    the corresponding model object.

    Raises:
        Exception: a string other than the two accepted ones 
                   as provided

    Returns: 
        BaseCNN or DynamicCNN: the model object, which will later 
                               be initialised
    """

    models_and_names = {
        "base": BaseCNN,
        "dynamic": DynamicCNN
    }

    if model in models_and_names.keys():

        return models_and_names[model]

    else:

        raise Exception(
            "Please enter 'base' for the base model, or 'dynamic' for the dymnamic model"
        )


def get_optimizer(
    model: BaseCNN|DynamicCNN,
    optimizer: str,
    learning_rate: float,
    weight_decay: float|None,
    momentum: float|None
    ) -> Optimizer:

    """
    The function returns the required optimizer function, based on
    the entered specifications.

    Args: 
        Model: the model that is being trained

        optimizer: the function that will be used to search for the 
                   global minimum of the loss function.

        learning_rate: the learning rate that is optimizer is using for 
                        its search.
                        
    Raises:
        NotImplementedError: The requested optimizer has not been implemented

    Returns:
        Optimizer: the optimizer that will be returned.
    """

    optimizers_and_likely_spellings = {
        ("adam", "Adam"): Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay),
        ("sgd", "SGD"): SGD(params=model.parameters(), lr=learning_rate, momentum=momentum),
        ("rmsprop", "RMSprop"): RMSprop(params=model.parameters(), lr=learning_rate, momentum=momentum)
    }

    optimizer_for_each_spelling = {
        spelling: function for spellings, function in 
        optimizers_and_likely_spellings.items() for spelling in spellings
    }

    if optimizer in optimizer_for_each_spelling.keys():

        return optimizer_for_each_spelling[optimizer]

    else:
        raise NotImplementedError("Consider using the Adam, SGD, or RMSprop optimizers")


def set_training_device(model: BaseCNN|DynamicCNN):

    """ Use the GPU if available. Otherwise, default to using the CPU. """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


def run_training_loop(
    model: BaseCNN|DynamicCNN, 
    criterion: callable,
    save: bool,
    optimizer: callable,
    num_epochs: int,
    num_classes: int,
    batch_size: int
    ) -> tuple[float, float, float, float]:

    """

    Args:
        model: the model object that is to be trained

        criterion: the loss function to be used 
        
        save: whether or not the model is to be saved

        optimizer: the optimizer that we will use to 
                   seek the global minimum of the loss 
                   function

        num_epochs: the number of epochs that the model
                    should be trained for

        num_classes: the number of classes (genera) the 
                     mushrooms should be classified into

    """

    # Prepare metrics
    precision = MulticlassPrecision(num_classes=num_classes, average="macro")
    recall = MulticlassRecall(num_classes=num_classes, average="macro")
    accuracy = MulticlassAccuracy(num_classes=num_classes, average="macro")

    logger.info("Collecting training data")
    train_loader = make_dataset(path=TRAINING_DATA_DIR, batch_size=batch_size)
    train_iterator = iter(train_loader)

    logger.info("Setting training device")
    device = set_training_device(model=model)

    logger.info("Training an untuned model") 
    for epoch in range(num_epochs):

        logger.info(f"Starting Epoch {epoch}")

        # Put model in training mode
        model.train()

        # Initialise training loss
        training_loss_total = 0.0

        for batch in tqdm(train_loader):

            # Refresh gradients
            optimizer.zero_grad()
            images, label = batch

            images = images.to(device)
            label = label.to(device)
            
            output = model(images)

            # Calculate the training loss 
            training_loss = criterion(output, label)

            # Calculate the gradient of the loss function
            training_loss.backward()

            # Adjust weights and biases
            optimizer.step()

            training_loss_total += training_loss.item()

        training_loss_avg = training_loss_total / len(train_iterator)

        # Put the model in evaluation mode
        model.eval()

        # Initialise validation loss
        val_loss_total = 0.0
        val_recall_total = 0.0
        val_accuracy_total = 0.0
        val_precision_total = 0.0

        # Get validation data
        val_data_loader = make_dataset(path=VALIDATION_DATA_DIR, batch_size=batch_size)
        val_iterator = iter(val_data_loader)

        with torch.no_grad():
            
            for batch in val_data_loader:

                images, label = batch

                images = images.to(device)
                label = label.to(device)

                output = model.forward(images)
                val_loss = criterion(output, label).item()

                val_loss_total += val_loss

                _ , predictions = torch.max(input=output, dim=1)

                val_recall = recall(predictions, label)
                val_accuracy = accuracy(predictions, label)
                val_precision = precision(predictions, label)

                val_recall_total += val_recall
                val_accuracy_total += val_accuracy
                val_precision_total += val_precision

            val_loss_avg = val_loss_total / len(val_iterator)

            val_recall_avg = val_recall_total / len(val_iterator)

            val_accuracy_avg = val_accuracy_total / len(val_iterator)
            val_precision_avg = val_precision_total / len(val_iterator)

            logger.success(
                "Epoch: {}, Average Training Loss: {:.2f}, Average Validation_loss: {:.2f}, \
                Average Accuracy: {:.2f}, Average Recall: {:.2f}, Average Precision: {:.2f} \
                ".format(epoch, training_loss_avg, val_loss_avg, val_accuracy_avg, val_recall_avg, val_precision_avg)
            )

            return val_loss_avg, val_accuracy_avg, val_recall_avg, val_precision_avg

    # Save model parameters
    if save:
        torch.save(model.state_dict(), MODELS_DIR)
    
    logger.info("Finished Training")


def train(
    model: str,
    batch_size: int,
    learning_rate: int,
    num_epochs: int|None,
    optimizer: str,
    device: str,
    save: bool,
    tune_hyperparams: bool|None = True,
    tuning_trials: int|None = 10
    ):

    """
    Train the requested model in either an untuned 
    default state, or in the most optimal tuned form 
    that was obtained after the specified number of 
    tuning trials.


    """
    
    num_classes = len(get_classes())

    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=os.getenv("COMET_PROJECT_NAME"),
        workspace=os.getenv("COMET_WORKSPACE"),
        log_code=False
    )

    logger.info("Setting up neural network")
    model_fn = get_model(model=model)

    if not tune_hyperparams:

        if isinstance(model_fn, BaseCNN):

            model = model_fn(num_classes=num_classes)

        if isinstance(model_fn, DynamicCNN):

            default_layer_config = {
                [
                    {"type": "conv", "out_channels": 8, "kernel_size": 8, "padding": 1, "pooling": True},
                    {"type": "conv", "out_channels": 16, "kernel_size": 3, "padding": 1, "pooling": True},
                    {"type": "fully_connected"}
                ]
            }

            model = DynamicCNN(
                in_channels=3,
                num_classes=num_classes,
                layer_config=default_layer_config,
                dropout_prob=0.5
            )

        criterion = CrossEntropyLoss()

        chosen_optimizer = get_optimizer(
            model=model, 
            learning_rate=learning_rate,
            optimizer=optimizer
        )

        val_metrics = run_training_loop(
            model = model, 
            num_epochs=10,
            criterion=criterion,
            optimizer=chosen_optimizer,
            num_classes=num_classes,
            save=True
        )

    else:

        logger.info("Finding optimal values of hyperparameters")

        optimise_hyperparams(
            model_fn=model,
            tuning_trials=10,
            experiment=experiment
        )


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--model", type=str, default="dynamic")
    parser.add_argument("--tune_hyperparams", action="store_true", default=True)
    parser.add_argument("--tuning_trials", type=int, default=10)

    args = parser.parse_args()

    train(
        model=args.model,
        tune_hyperparams=args.tune_hyperparams,
        tuning_trials=args.tuning_trials
    )
