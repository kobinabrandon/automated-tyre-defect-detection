import joblib

from comet_ml import Experiment
from loguru import logger 
from optuna import trial, create_study, Study
from torch.nn import CrossEntropyLoss

from src.feature_pipeline.data_preparation import get_classes
from src.training_pipeline.training import get_optimizer, run_training_loop, BaseCNN, DynamicCNN
from src.setup.paths import TRIALS_DIR


def save_trial_callback(study: Study, frozen_trial: trial.FrozenTrial):

    trial_name = TRIALS_DIR/f"Trial_{frozen_trial.number}.pkl"
    joblib.dump(value=frozen_trial, filename=trial_name)
 

class BestTrials():

    def __init__(self, study: Study):

        self.trial_lowest_avg_val_loss = min(study.best_trials, key=lambda t: t.values[0])
        self.trial_highest_avg_val_accuracy = max(study.best_trials, key=lambda t: t.values[1])
        self.trial_highest_val_recall = max(study.best_trials, key=lambda t: t.values[2])
        self.trial_highest_val_precision = max(study.best_trials, key=lambda t: t.values[3])

        self.metrics_and_trials = {
            "avg_val_loss": self.trial_lowest_avg_val_loss,
            "avg_val_accuracy": self.trial_highest_avg_val_accuracy,
            "avg_val_recall": self.trial_highest_val_recall,
            "avg_val_precision": self.trial_highest_val_precision
        }

    def _save_best_trials(self):

        for item in self.metrics_and_trials.items():

            if item[0] == "avg_val_loss":

                joblib.dump(
                    value=item[1], filename=TRIALS_DIR/f"Trial with the lowest {item[0]}.pkl"
                )

            else:

                joblib.dump(
                    value=item[1], filename=TRIALS_DIR/f"Trial with the highest {item[0]}.pkl"
                )


def optimise_hyperparams(
    model_fn: BaseCNN|DynamicCNN,
    tuning_trials: int,
    experiment: Experiment
    ):

    def objective(trial: trial.Trial) -> tuple[float, float, float, float]:

        num_classes = len(get_classes())

        if isinstance(model_fn, BaseCNN):

            model = model_fn(num_classes=num_classes)

        if isinstance(model_fn, DynamicCNN):

            # Choose the number of convolutional, and fully connected layers
            conv_layers = trial.suggest_int(name="conv_layers", low=1, high=4)
            fully_connected = trial.suggest_int(name="fully_connected_layers", low=1, high=4)

            # For each of these convolutional layers, choose values of the parameters of the model
            layer_config = []       
            for _ in range(conv_layers):

                config = {
                    "type": "conv",
                    "out_channels": trial.suggest_int(name="conv_out_channels", low=16, high=64),
                    "kernel_size": trial.suggest_int(name="conv_kernel_size", low=3, high=6),
                    "stride": trial.suggest_int(name="conv_stride", low=1, high=4),
                    "padding": trial.suggest_categorical(name="padding", choices=["same", "valid"])
                }

                layer_config.append(config)

            for _ in range(fully_connected):

                config = {
                    "type": "fully_connected", "out_features": num_classes
                }

                layer_config.append(config)
            
            # Create model
            model = model_fn(
                in_channels=3, 
                num_classes=num_classes, 
                layer_config=layer_config
            )

        num_epochs = trial.suggest_int(name="num_epochs", low=5, high=20)

        criterion = CrossEntropyLoss()
        
        optimizer_name = trial.suggest_categorical(
            name="optimizer", 
            choices=["Adam", "SGD", "RMSProp"]
        )

        if optimizer_name == "Adam":

            optimizer = get_optimizer(
                model=model, 
                optimizer=optimizer_name, 
                learning_rate=trial.suggest_float(name="lr", low=1e-5, high=1e-1, log=True),
                weight_decay=trial.suggest_float(name="weight_decay", low = 0.001, high = 0.08, log=True)
            )

        else: 

            optimizer = get_optimizer(
                model=model,
                optimizer=optimizer_name,
                learning_rate=trial.suggest_float(name="lr", low=1e-5, high=1e-1, log=True),
                momentum=trial.suggest_float(name="momentum", low=0.1, high=0.9)
            )

        
        val_metrics = run_training_loop(
            model=model, 
            criterion=criterion,
            optimizer=optimizer,
            num_classes=num_classes,
            num_epochs=num_epochs,
            save=True
        )
        
        return val_metrics
            
    logger.info("Searching for optimal values of the hyperparameters")

    study = create_study(
        directions=["minimize", "maximize", "maximize", "maximize"]
    )

    study.optimize(
        func=objective, 
        n_trials=tuning_trials, 
        callbacks=[save_trial_callback],
        timeout=300
    )
    
    logger.info("Number of finished trials:", len(study.trials))
    
    best_trials = BestTrials(study=study)

    # Save the best trials
    best_trials._save_best_trials()

    logger.info("The best hyperparameters are:")

    for key, value in study.best_params.items():

        logger.info(f"{key}:{value}")

    experiment.log_metrics(dic=best_trials.metrics_and_trials)

    