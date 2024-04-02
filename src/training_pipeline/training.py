from comet_ml import Experiment

import os
import torch

from loguru import logger 
from tqdm import tqdm 
from argparse import ArgumentParser

from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD, RMSprop
from torch.optim.optimizer import Optimizer 

from torchmetrics.classification import (
    MulticlassPrecision, MulticlassAccuracy, MulticlassRecall
)

from src.setup.paths import TRAINING_DATA_DIR, VALIDATION_DATA_DIR, MODELS_DIR
from src.feature_pipeline.data_preparation import make_dataset, get_classes
from src.training_pipeline.models import BaseCNN, DynamicCNN


def get_optimizer(
    model_fn: BaseCNN|DynamicCNN,
    optimizer: str,
    learning_rate: float,
    weight_decay: float|None,
    momentum: float|None
    ) -> Optimizer:

    """
    The function returns the required optimizer function, based on the entered
    specifications.

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
        ("adam", "Adam"): Adam(params=model_fn.parameters(), lr=learning_rate, weight_decay=weight_decay),
        ("sgd", "SGD"): SGD(params=model_fn.parameters(), lr=learning_rate, momentum=momentum),
        ("rmsprop", "RMSprop"): RMSprop(params=model_fn.parameters(), lr=learning_rate, momentum=momentum)
    }

    optimizer_for_each_spelling = {
        spelling: function for spellings, function in 
        optimizers_and_likely_spellings.items() for spelling in spellings
    }

    if optimizer in optimizer_for_each_spelling.keys():

        return optimizer_for_each_spelling[optimizer]

    else:
        raise NotImplementedError("Consider using the Adam, SGD, or RMSprop optimizers")


def set_training_device(model_fn: BaseCNN|DynamicCNN):

    """ Use the GPU if available. Otherwise, default to using the CPU. """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_fn.to(device)


def run_training_loop(
    model_fn: BaseCNN|DynamicCNN, 
    criterion: callable,
    save: bool,
    optimizer: callable,
    num_epochs: int,
    num_classes: int,
    batch_size: int
    ) -> tuple[float, float, float, float]:

    """
    Initialise the multi-class precision, recall, and accuracy metrics.
    Then load the training data and set the training device. Train the 
    network in question for the specified number of epochs, put the 
    model in evaluation mode and report the average values of the
    validation loss, recall, accuracy, and precision

    Args:
        model_fn: the model object that is to be trained

        criterion: the loss function to be used 
        
        save: whether or not the model is to be saved

        optimizer: the optimizer that we will use to seek the global
                minimum of the loss function
                   

        num_epochs: the number of epochs that the model should be
                    trained for.

        num_classes: the number of classes (genera) the mushrooms
                     should be classified into. 

    Returns:
        val_metrics: a list of floats which are the average values
                     of the loss, recall, accuracy, and precision
                     of the trained model on the validation set.             
    """ 

    # Prepare metrics
    precision = MulticlassPrecision(num_classes=num_classes, average="macro")
    recall = MulticlassRecall(num_classes=num_classes, average="macro")
    accuracy = MulticlassAccuracy(num_classes=num_classes, average="macro")

    logger.info("Collecting training data")
    train_loader = make_dataset(path=TRAINING_DATA_DIR, batch_size=batch_size)
    train_iterator = iter(train_loader)

    logger.info("Setting training device")
    device = set_training_device(model_fn=model_fn)

    logger.info("Training an untuned model") 
    for epoch in range(num_epochs):

        logger.info("Starting Epoch", epoch)

        # Put model in training mode
        model_fn.train()

        # Initialise training loss
        training_loss_total = 0.0

        for batch in tqdm(train_loader):

            # Refresh gradients
            optimizer.zero_grad()
            images, label = batch

            images = images.to(device)
            label = label.to(device)
            
            output = model_fn._forward(images)

            # Calculate the training loss 
            training_loss = criterion(output, label)

            # Calculate the gradient of the loss function
            training_loss.backward()

            # Adjust weights and biases
            optimizer.step()

            training_loss_total += training_loss.item()

        training_loss_avg = training_loss_total / len(train_iterator)

        # Put the model in evaluation mode
        model_fn.eval()

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

                output = model_fn.forward(images)
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
        torch.save(model_fn.state_dict(), MODELS_DIR)
    
    logger.info("Finished Training")


def train(
    model_name: str,
    batch_size: int,
    learning_rate: int,
    weight_decay: float|None,
    momentum: float|None,
    num_epochs: int,
    optimizer: str,
    device: str,
    save: bool,
    tune_hyperparams: bool|None = True,
    tuning_trials: int|None = 10
    ):

    """
    Train the requested model in either an untuned default state, or in the
    most optimal tuned form that was obtained after the specified number of 
    tuning trials.

    Args:

        batch_size: the batch size to be used during training.

        learning_rate: the learning rate of the optimizer.

        num_epochs: the number of epochs that the model should be trained 
                    for.

        optimizer: the name of the optimizer that is to be used.

        device: a string which determines whether the CPU or a GPU will be
                used for training.

        save: a boolean that determines whether the model is to be saved

        tune_hyperparams: a boolean that indicates whether hyperparameters
                          are to be tuned or not. If it is False, a default
                          version of the model will be trained.
    """
    
    num_classes = len(get_classes())

    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=os.getenv("COMET_PROJECT_NAME"),
        workspace=os.getenv("COMET_WORKSPACE"),
        log_code=False
    )

    logger.info("Setting up neural network")

    if not tune_hyperparams:

        if model_name == "base":

            model_fn = BaseCNN(num_classes=num_classes)

        if model_name == "dynamic":

            default_layer_config = {
                [
                    {"type": "conv", "out_channels": 8, "kernel_size": 8, "padding": 1, "pooling": True},
                    {"type": "conv", "out_channels": 16, "kernel_size": 3, "padding": 1, "pooling": True},
                    {"type": "fully_connected"}
                ]
            }

            model_fn = DynamicCNN(
                in_channels=3,
                num_classes=num_classes,
                layer_config=default_layer_config,
                dropout_prob=0.5
            )

        criterion = CrossEntropyLoss()
            
        chosen_optimizer = get_optimizer(
            model_fn=model_fn, 
            learning_rate=learning_rate,
            optimizer=optimizer,
            weight_decay=weight_decay,
            momentum=momentum
        )

        val_metrics = run_training_loop(
            model_fn=model_fn, 
            num_epochs=num_epochs,
            criterion=criterion,
            optimizer=chosen_optimizer,
            num_classes=num_classes,
            batch_size=batch_size,
            save=True
        )

    else:

        from src.training_pipeline.hyperparameter_tuning import optimize_hyperparams

        logger.info("Finding optimal values of hyperparameters")

        optimize_hyperparams(
            model_fn=model_fn,
            tuning_trials=10,
            experiment=experiment
        )


train(
    model_name="base",
    batch_size=20,
    learning_rate=1e-4,
    num_epochs=10,
    optimizer="adam",
    tune_hyperparams=False,
    device="cpu",
    weight_decay=0.01,
    momentum=0.02,
    save=False
)