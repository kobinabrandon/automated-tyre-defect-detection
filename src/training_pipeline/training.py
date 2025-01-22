from comet_ml import Experiment  # For some reason, to log to CometML automatically, we must import comet_ml before torch
import torch

from torch._C import device
from tqdm import tqdm
from loguru import logger
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from torch.optim.sgd import SGD
from torch.optim.adam import Adam
from torch.optim.rmsprop import RMSprop
from torch.nn import CrossEntropyLoss
from torch.optim.optimizer import Optimizer

from torchvision.datasets import ImageFolder
from torchmetrics.classification import Precision, Recall, Accuracy  
from transformers import ViTForImageClassification, ViTHybridForImageClassification, BeitForImageClassification

from src.setup.paths import MODELS_DIR
from src.setup.config import model_config, env, data_config 
from src.feature_pipeline.preprocessing import prepare_images, split_data
from src.training_pipeline.models import get_image_processor, get_pretrained_model


num_classes = data_config.num_classes

experiment = Experiment(
    api_key=env.comet_api_key,
    project_name=env.comet_project_name,
    workspace=env.comet_workspace,
    log_code=False
)


def remove_extra_batch_dimension(images: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Remove the extra batch dimension that is created when the VitImageProcessor is applied to 
    images in prior to the construction of the ImageFolder and DataLoader objects. 

    Args:
        images: a dictionary that consists of the "pixel_values" key and a corresponding tensor. 

    Returns:
        dict[str, torch.Tensor]: the same dictionary, except with the adjusted tensor. 
    """
    adjusted_tensor: torch.Tensor = images["pixel_values"].squeeze(1)
    return {"pixel_values": adjusted_tensor}


class CustomLoop:
    def __init__(
        self,
        model_name: str = model_config.vit_base,
        epochs: int = 10, 
        momentum: float = 0.1,
        learning_rate: float = 0.2, 
        optimizer_name: str = "Adam",
        weight_decay: float = 0.3,
        batch_size: int = model_config.batch_size,
        trials: int | None = 10
    ) -> None:
        """
        A custom training loop that includes hyperparameter-tuning options. 
        
        Args:
            trials: the number of optuna trials to run.
            model_name: the name of the model to be trained 
            batch_size: the batch size to be used during training.
            learning_rate: the learning rate of the optimizer.
            weight_decay: a regularization term that reduces the weights
            epochs: the number of epochs that the model should be trained for.
            optimizer_name: the name of the optimizer to be used during training.  
            momentum: the momentum coefficient used during stochastic gradient descent (SGD)
            tune_hyperparams: a boolean that indicates whether hyperparameters are to be tuned.
        """
        self.epochs: int = epochs 
        self.momentum: float = momentum
        self.trials: int | None = trials
        self.batch_size: int = batch_size
        self.model_name: str = model_name
        self.weight_decay: float = weight_decay
        self.learning_rate: float = learning_rate
        self.optimizer_name: str = optimizer_name   
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        self.images: ImageFolder = prepare_images(model_name=model_name, augment_images=False)  
        self.datasets: tuple[DataLoader[ImageFolder], DataLoader[ImageFolder], DataLoader[ImageFolder]] = split_data(images=self.images)
        self.train_dataloader: DataLoader[ImageFolder] = self.datasets[0]
        self.val_dataloader: DataLoader[ImageFolder] = self.datasets[1]

    def __prepare_metrics__(self):
        precision = Precision(task="binary", num_classes=num_classes, average="macro").to(device=self.device)
        recall = Recall(task="binary", num_classes=num_classes, average="macro").to(device=self.device)
        accuracy = Accuracy(task="binary", num_classes=num_classes, average="macro").to(device=self.device)
        return precision, recall, accuracy

    def train(self) -> tuple[float, float, float, float]:
        """
        Initialise the multi-class precision, recall, and accuracy metrics. Then load the training data and 
        set the training device. Train the network in question for the specified number of epochs, put the 
        model in evaluation mode and report the average values of the validation loss, recall, accuracy, 
        and precision

        Args:
            criterion: the loss function to be used 
            save: whether the model is to be saved
            optimizer: the optimizer that we will use to seek the global minimum of the loss function

        Returns:
            val_metrics: a list of floats which are the average values of the loss, recall, accuracy, and 
                         precision of the trained model on the validation set.             
        """
        model = get_pretrained_model(model_name=self.model_name)
        return self.run_training_loop(model=model)

    def get_optimizer(
        self,
        model: ViTForImageClassification | ViTHybridForImageClassification | BeitForImageClassification, 
    ) -> Optimizer:
        """
        The function returns the required optimizer function, based on the entered
        specifications.

        Args: 
            model: the model that is being trained
            batch_size: the number of data samples in each batch
            optimizer_name: the function that will be used to search for the global minimum of the loss function.
            learning_rate: the learning rate that is optimizer is using for its search.
            weight_decay: a regularization term that reduces the network's weights
            momentum: the momentum coefficient used during stochastic gradient descent (SGD)

        Raises:,
            NotImplementedError: The requested optimizer has not been implemented

        Returns:
            Optimizer: the optimizer that will be returned.
        """
        optimizers = {
            "adam": Adam(params=model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay),
            "sgd": SGD(params=model.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay),
            "rmsprop": RMSprop(params=model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        }

        if self.optimizer_name.lower() in optimizers.keys():
            return optimizers[self.optimizer_name.lower()]
        else:
            raise NotImplementedError("Please use the Adam, SGD, or RMSprop optimizers")

    def run_training_loop(
        self, 
        model: ViTForImageClassification | ViTHybridForImageClassification | BeitForImageClassification,
        save: bool = True
    ) -> tuple[float, float, float, float]: 
        """
        Train the requested model in either an untuned default state, or in the
        most optimal tuned form that was obtained after the specified number of tuning trials.

        Args:
            model: the model being trained 
            save: whether to save the model artifact as a pkl file.

        Returns:
          tuple[float, float, float, float]: tuple of metrics on the validation set.  
       """
        logger.info("Collecting training data")
        train_iterator = iter(self.train_dataloader)

        logger.info("Initialising models")
        model = get_pretrained_model(model_name=self.model_name)
        model.to(device=self.device)

        precision, recall, accuracy = self.__prepare_metrics__()
        optimizer = self.get_optimizer(model=model) 
        criterion = CrossEntropyLoss()

        for epoch in range(self.epochs):

            logger.info(f"Starting Epoch #{epoch}")
            model.train()  # Put model in training mode
            training_loss_total = 0.0  # Initialise training loss
            processor = get_image_processor(model_name=self.model_name)

            for images, labels in tqdm(self.train_dataloader):

                optimizer.zero_grad()  # Refresh gradients
                images, labels = images.to(self.device), labels.to(self.device)
                output = model(**remove_extra_batch_dimension(images))

                training_loss = criterion(output.logits, labels)  # Calculate the training loss 
                training_loss.backward()  # Calculate the gradient of the loss function
                optimizer.step()  # Adjust weights and biases
                training_loss_total += training_loss.item()

            training_loss_avg: float = training_loss_total / len(train_iterator)
            model.eval()  # Put the model in evaluation mode

            # Initialise validation loss
            val_loss_total = 0.0
            val_recall_total = 0.0
            val_accuracy_total = 0.0
            val_precision_total = 0.0

            # Get validation data
            val_iterator = iter(self.val_dataloader)

            with torch.no_grad():  # Initialise validation mode

                for (images, labels) in self.val_dataloader:

                    images, labels = images.to(self.device), labels.to(self.device)
                    output = model(**remove_extra_batch_dimension(images))
                    
                    val_loss = criterion(output.logits, labels).item()
                    val_loss_total += val_loss

                    _, predictions = torch.max(input=output.logits, dim=1)

                    experiment.log_confusion_matrix(
                        y_true=labels,
                        y_predicted=predictions,
                        title="Confusion Matrix: Evaluation",
                        file_name="confusion-matrix.json"
                    )

                    val_recall: float = recall(predictions, labels)
                    val_accuracy: float = accuracy(predictions, labels)
                    val_precision: float = precision(predictions, labels)

                    val_recall_total += val_recall
                    val_accuracy_total += val_accuracy
                    val_precision_total += val_precision

                val_loss_avg = val_loss_total / len(val_iterator)
                val_recall_avg = val_recall_total / len(val_iterator)
                val_accuracy_avg = val_accuracy_total / len(val_iterator)
                val_precision_avg = val_precision_total / len(val_iterator)

                logger.success(
                    "Epoch: [{}/{}], Average Training Loss: {:.2f}, Average Validation_loss: {:.2f}, \
                    Average Validation Accuracy: {:.2f}, Average Validation Recall: {:.2f},\
                    Average Validation Precision: {:.2f}".format(
                        epoch + 1, 
                        self.epochs, 
                        training_loss_avg, 
                        val_loss_avg, 
                        val_accuracy_avg, 
                        val_recall_avg, 
                        val_precision_avg
                    )
                )

                val_metrics = {
                    "Epoch": epoch,
                    "Average Training Loss": training_loss_avg,
                    "Average Validation_loss": val_loss_avg,
                    "Average Validation Accuracy": val_accuracy_avg,
                    "Average Validation Recall": val_recall_avg,
                    "Average Validation Precision": val_precision_avg
                }

                with experiment.test():
                    experiment.log_metrics(val_metrics)

        logger.info("Finished Training")
#            return val_loss_avg, val_accuracy_avg, val_loss_avg, val_precision_avg


if __name__ == "__main__":
    parser = ArgumentParser()
    _ = parser.add_argument("--tune_hyperparams", action="store_true")
    args = parser.parse_args()

    if not args.tune_hyperparams:

        trainer = CustomLoop(
            model_name=model_config.vit_base,
            epochs=10, 
            momentum=0.1,
            learning_rate=0.2, 
            optimizer_name="Adam",
            weight_decay=0.3,
            batch_size=model_config.batch_size,
            trials=10
        )

        trainer.train()
        
    else:
        from src.training_pipeline.hyperparameter_tuning import perform_tuning
        perform_tuning(model_name=model_config.vit_base, trials= 20, experiment=experiment)
 
