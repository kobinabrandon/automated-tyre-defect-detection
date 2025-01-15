from PIL.Image import Image
from comet_ml import Experiment  # For some reason, to log to CometML automatically, we must import comet_ml before torch

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger

from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD, RMSprop
from torch.optim.optimizer import Optimizer

from transformers import (
    Trainer,
    TrainingArguments,
    ViTForImageClassification, 
    ViTHybridForImageClassification, 
    BeitForImageClassification, 
    AutoImageProcessor 
)


from torchvision.datasets import ImageFolder
from torchmetrics.classification import MulticlassPrecision, MulticlassAccuracy, MulticlassRecall

from src.setup.config import model_config, env, data_config 
from src.setup.paths import TRAIN_DATA_DIR, VAL_DATA_DIR, MODELS_DIR, LOGS_DIR, DATA_DIR

from src.feature_pipeline.preprocessing import make_full_dataset, split_data, prepare_data
from src.training_pipeline.models import get_pretrained_model


num_classes = data_config.num_classes

experiment = Experiment(
    api_key=env.comet_api_key,
    project_name=env.comet_project_name,
    workspace=env.comet_workspace,
    log_code=False
)

class HFLoop:
    def __init__(
        self, 
        model_name: str, 
        epochs: int,
        learning_rate: float, 
        batch_size: int = model_config.batch_size
    ) -> None:
        self.epochs: int = epochs 
        self.model_name: str = model_name
        self.learning_rate: float = learning_rate
        self.batch_size: int = batch_size

        self.datasets: tuple[DataLoader[ImageFolder], DataLoader[ImageFolder], DataLoader[ImageFolder]] = prepare_data()
        self.training_set: DataLoader[ImageFolder] = self.datasets[0]
        self.validation_set: DataLoader[ImageFolder] = self.datasets[1]
    
    def train(self):
        """
        Train the requested model in either an untuned default state, or in the
        most optimal tuned form that was obtained after the specified number of 
        tuning trials.

        Args:
            batch_size: the batch size to be used during training.
            learning_rate: the learning rate of the optimizer.
            num_epochs: the number of epochs that the model should be trained for.
            optimizer_name: the name of the optimizer that is to be used.
            tune_hyperparams: a boolean that indicates whether hyperparameters are to be tuned.
            trials: the number of optuna trials to run.
        """
        model_fn: ViTForImageClassification | ViTHybridForImageClassification | BeitForImageClassification = get_pretrained_model(model_name=self.model_name)
        
        training_params = TrainingArguments(
            output_dir=str(MODELS_DIR),
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            logging_dir=str(LOGS_DIR),
            logging_strategy="epoch",
            eval_strategy="epoch",
            save_strategy="epoch"
        )

        trainer = Trainer(
            model=model_fn, 
            args=training_params, 
            train_dataset=self.training_set, 
            eval_dataset=self.validation_set
        )

        trainer.train()


# logger.info("Finding optimal values of hyperparameters")
# perform_tuning(model_name=model_name, trials=5, experiment=experiment)
#

if __name__ == "__main__":
    loop = HFLoop(model_name=model_config.vit_base, epochs=5, learning_rate=0.1)    
    loop.train()
    


# class ToyModelTrainer:
#     def __init__(self, model_name: str, batch_size: int, num_epochs: int) -> None:
#         self.model_name = model_name
#         self.batch_size = batch_size
#         self.num_epochs = num_epochs
#         self.dataset = make_full_dataset(path=DATA_DIR, pretrained=False)
#         self.train_dataloader, self.val_dataloader, self.test_dataloader = split_data(dataset=self.dataset)
#
#     def _prepare_metrics():
#         precision = MulticlassPrecision(num_classes=num_classes, average="macro")
#         recall = MulticlassRecall(num_classes=num_classes, average="macro")
#         accuracy = MulticlassAccuracy(num_classes=num_classes, average="macro")
#         return precision, recall, accuracy
#
#     def _get_optimizer(
#         self,
#         model_fn: BaseCNN | DynamicCNN | ResNet
#         learning_rate: float,
#         optimizer_name: str| None,
#         weight_decay: float | None,
#         momentum: float | None
#     ) -> Optimizer:
#         """
#         The function returns the required optimizer function, based on the entered
#         specifications.
#
#         Args: 
#             model_fn: the model that is being trained
#             optimizer_name: the function that will be used to search for the global minimum of the loss function.
#             learning_rate: the learning rate that is optimizer is using for its search.
#             weight_decay: a regularization term that reduces the network's weights
#             momentum: the momentum coefficient used during stochastic gradient descent (SGD)
#
#         Raises:,
#             NotImplementedError: The requested optimizer has not been implemented
#
#         Returns:
#             Optimizer: the optimizer that will be returned.
#         """
#         optimizers = {
#             "adam": Adam(params=model_fn.parameters(), lr=learning_rate, weight_decay=weight_decay),
#             "sgd": SGD(params=model_fn.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay),
#             "rmsprop": RMSprop(params=model_fn.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
#         }
#
#         if optimizer_name.lower() in optimizers.keys():
#             return optimizers[optimizer_name.lower()]
#         elif optimizer_name is None:
#             return optimizers["adam"]
#         else:
#             raise NotImplementedError("Please use the Adam, SGD, or RMSprop optimizers")
#
#     def _run_training_loop(self, criterion: callable, save: bool, optimizer: callable,) -> list[float]:
#         """
#         Initialise the multi-class precision, recall, and accuracy metrics.
#         Then load the training data and set the training device. Train the 
#         network in question for the specified number of epochs, put the 
#         model in evaluation mode and report the average values of the
#         validation loss, recall, accuracy, and precision
#
#         Args:
#             criterion: the loss function to be used 
#             save: whether the model is to be saved
#
#             optimizer: the optimizer that we will use to seek the global
#                     minimum of the loss function
#
#             num_epochs: the number of epochs that the model should be
#                         trained for.
#
#             batch_size: the number of data samples in each batch
#
#         Returns:
#             val_metrics: a list of floats which are the average values
#                         of the loss, recall, accuracy, and precision
#                         of the trained model on the validation set.             
#         """
#         precision, recall, accuracy = self._prepare_metrics()
#
#         logger.info("Collecting training data")
#         train_iterator = iter(self.train_dataloader)
#
#         logger.info("Setting training device")
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model_fn.to(device=device)
#
#         logger.info("Training the untuned model")
#         for epoch in range(self.num_epochs):
#             logger.info(f"Starting Epoch #{epoch}")
#             model_fn.train()  # Put model in training mode
#             training_loss_total = 0.0  # Initialise training loss
#
#             for (images, labels) in tqdm(self.train_dataloader):
#                 optimizer.zero_grad()  # Refresh gradients
#                 images, labels = images.to(device), labels.to(device)
#                 output = model_fn.forward(images)
#                 training_loss = criterion(output, labels)  # Calculate the training loss 
#                 training_loss.backward()  # Calculate the gradient of the loss function
#                 optimizer.step()  # Adjust weights and biases
#                 training_loss_total += training_loss.item()
#
#             training_loss_avg = training_loss_total / len(train_iterator)
#             model_fn.eval()  # Put the model in evaluation mode
#
#             # Initialise validation loss
#             val_loss_total = 0.0
#             val_recall_total = 0.0
#             val_accuracy_total = 0.0
#             val_precision_total = 0.0
#
#             # Get validation data
#             val_iterator = iter(self.val_dataloader)
#
#             with torch.no_grad():
#
#                 for (images, labels) in self.val_data_loader:
#                     images, labels = images.to(device), labels.to(device)
#
#                     output = model_fn.forward(images)
#                     val_loss = criterion(output, labels).item()
#                     val_loss_total += val_loss
#
#                     _, predictions = torch.max(input=output, dim=1)
#
#                     val_recall = recall(predictions, labels)
#                     val_accuracy = accuracy(predictions, labels)
#                     val_precision = precision(predictions, labels)
#
#                     val_recall_total += val_recall
#                     val_accuracy_total += val_accuracy
#                     val_precision_total += val_precision
#
#                 val_loss_avg = val_loss_total / len(val_iterator)
#                 val_recall_avg = val_recall_total / len(val_iterator)
#                 val_accuracy_avg = val_accuracy_total / len(val_iterator)
#                 val_precision_avg = val_precision_total / len(val_iterator)
#
#                 logger.success(
#                     "Epoch: [{}/{}], Average Training Loss: {:.2f}, Average Validation_loss: {:.2f}, \
#                     Average Validation Accuracy: {:.2f}, Average Validation Recall: {:.2f},\
#                     Average Validation Precision: {:.2f}".format(
#                         epoch + 1, 
#                         num_epochs, 
#                         training_loss_avg, 
#                         val_loss_avg, 
#                         val_accuracy_avg, 
#                         val_recall_avg, 
#                         val_precision_avg
#                     )
#                 )
#
#                 val_metrics = {
#                     "Epoch": epoch,
#                     "Average Training Loss": training_loss_avg,
#                     "Average Validation_loss": val_loss_avg,
#                     "Average Validation Accuracy": val_accuracy_avg,
#                     "Average Validation Recall": val_recall_avg,
#                     "Average Validation Precision": val_precision_avg
#                 }
#
#                 with experiment.test():
#                     experiment.log_metrics(val_metrics)
#
#                 experiment.log_confusion_matrix(
#                     y_true=labels,
#                     y_predicted=predictions,
#                     title="Confusion Matrix: Evaluation",
#                     file_name="confusion-matrix.json"
#                 )
#
#         # Save model parameters
#         if save:
#             torch.save(model_fn.state_dict(), MODELS_DIR)
#
#         logger.info("Finished Training")
#         return [val_loss_avg, val_accuracy_avg, val_loss_avg, val_precision_avg]
#
#     def train(
#         self,
#         learning_rate: float | None,
#         weight_decay: float | None,
#         momentum: float | None,
#         dropout: float | None,
#         optimizer_name: str | None,
#         tune_hyperparams: bool | None = True,
#         trials: int | None = 10
#         ) -> None:
#         """
#         Train the requested model in either an untuned default state, or in the
#         most optimal tuned form that was obtained after the specified number of 
#         tuning trials.
#
#         Args:
#             batch_size: the batch size to be used during training.
#             learning_rate: the learning rate of the optimizer.
#             weight_decay: a regularization term that reduces the weights
#             num_epochs: the number of epochs that the model should be trained for.
#             dropout: the proportion of nodes that will be omitted.
#             optimizer_name: the name of the optimizer that is to be used.
#             momentum: the momentum coefficient used during stochastic gradient descent (SGD)
#             tune_hyperparams: a boolean that indicates whether hyperparameters are to be tuned.
#             trials: the number of optuna trials to run.
#         """
#         assert self.model_name in ["base", "dynamic", "bigger", "resnet50", "resnet101", "resnet152"]
#
#         if not tune_hyperparams:            
#             model_fn = get_toy_model(model_name=self.model_name.lower())
#             criterion = CrossEntropyLoss()
#
#             chosen_optimizer = self._get_optimizer(
#                 model_fn=model_fn,
#                 learning_rate=learning_rate,
#                 optimizer_name=optimizer_name,
#                 weight_decay=weight_decay,
#                 momentum=momentum
#             )
#
#             val_metrics = self._run_training_loop(
#                 model_fn=model_fn,
#                 num_epochs=num_epochs,
#                 criterion=criterion,
#                 optimizer=chosen_optimizer,
#                 batch_size=batch_size,
#                 save=True
#             )
#
#         else:
#             logger.info("Finding optimal values of hyperparameters")
#             perform_tuning(model_name=self.model_name, trials=trials, batch_size=batch_size, experiment=experiment)
#

