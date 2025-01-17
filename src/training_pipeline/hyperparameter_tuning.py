from comet_ml import Experiment
import joblib

from loguru import logger 
from torch.optim import Adam
from optuna import create_study, Study
from optuna.trial import Trial, FrozenTrial
from optuna.visualization import plot_param_importances

from src.setup.paths import TRIALS_DIR
from src.setup.config import model_config, env 
from src.training_pipeline.training import CustomLoop 


def save_trial_callback(study: Study, frozen_trial: FrozenTrial):
    """
    A callback allowing a given optuna trial to be saved as a pickle file.

    Args:
        study: the optuna optimisation task being run.
        frozen_trial: the trial to be saved.
    """
    trial_path = TRIALS_DIR/f"Trial_{frozen_trial.number}.pkl"
    joblib.dump(value=frozen_trial, filename=trial_path)


class BestTrials(Trial):
    """
    This class contains the trials of a given optuna study that optimize the
    average values of the validation loss, recall, accuracy, and precision
    """

    def __init__(self, study: Study):
        super().__init__(study=study)
        self.trial_lowest_avg_val_loss: int = min(study.best_trials, key=lambda t: t.values[0])
        self.trial_highest_avg_val_accuracy: float = max(study.best_trials, key=lambda t: t.values[1])
        self.trial_highest_val_recall: float = max(study.best_trials, key=lambda t: t.values[2])
        self.trial_highest_val_precision: float = max(study.best_trials, key=lambda t: t.values[3])

        self.metrics_and_trials: dict[str, int | float] = {
            "avg_val_loss": self.trial_lowest_avg_val_loss,
            "avg_val_accuracy": self.trial_highest_avg_val_accuracy,
            "avg_val_recall": self.trial_highest_val_recall,
            "avg_val_precision": self.trial_highest_val_precision
        }

    def save_best_trials(self):
        """ Save the best optuna trial for each metric as a pickle file. """

        for item in self.metrics_and_trials.items():
            if "loss" in item[0]:
                joblib.dump(value=item[1], filename=TRIALS_DIR/f"Trial with the lowest {item[0]}.pkl")
            else:
                joblib.dump(value=item[1], filename=TRIALS_DIR/f"Trial with the highest {item[0]}.pkl")

    def display_best_trials(self):
        """ Display the key details about the best trials """

        for item in self.metrics_and_trials.items():
            if "loss" in item[0]:
                logger.info(f"Trial with lowest {item[0]}:")
            else: 
                logger.info(f"Trial with highest {item[0]}:")

            logger.info(f"number: {item[1].number}")   
            logger.info(f"params: {item[1].params}")
            logger.info(f"values: {item[1].values}")   

    def _view_hyperparam_importances(self, study: Study):
        """ Plots hyperparameter importances """

        metric_names = list(self.metrics_and_trials.keys())
        for name in metric_names:
            plot_param_importances(study=study, target=lambda t: t.values[metric_names.index(name)], target_name=name)

    def log_to_comet(self):
        """ Log the optimization tasks with CometML """

        experiment = Experiment(
            api_key=env.comet_api_key, 
            project_name=env.comet_project_name, 
            workspace=env.comet_workspace, 
            log_code=False
        )   

        for key, value in list(self.metrics_and_trials.items()):
            objective = "minimize" if "loss" in key else "maximize"
            experiment.log_optimization(metric_name=key, metric_value=value, parameters=value.params, objective="minimize")

     
def perform_tuning(model_name: str, trials: int, experiment: Experiment):
    """
    Using the objective function below, optimise the specified hyperparameters by an optuna study for running the 
    specified number of tuning trials. Then save, display these trials before logging them with Comet

    Args:
        model_name (str): the chosen shorthand name of the model
        trials (int): the number of hyperparameter tuning trials to be run.
        batch_size (int): the batch size to be used during training.
        experiment (Experiment):a single instance of a CometML experiment.
    """
    assert model_name in [model_config.vit_base, model_config.vit_hybrid, model_config.beit_base] 

    def objective_function(trial: Trial) -> tuple[float, float, float, float]:
        """
        For each optuna trial, initialise values of the hyperparameters within the specified ranges, select one
        of three possible optimizers, and run the training loop to obtain a tuple of metrics on the validation set. 
        
        I didn't use get_optimizer() here because I would have to include momentum=None in the arguments because
        Adam does not have a momentum parameter. Upon doing this, optuna complains about momentum having no value. 
        It's simply easier to call Adam directly here.

        Args:
            trial: the particular optuna trial being run

        Returns:
            val_metrics: contains a list of floats which are the average values of  the loss,recall, accuracy, 
                        and precision of the trained model on the validation set.
        """
        num_epochs = trial.suggest_int(name="num_epochs", low=5, high=15)
        batch_size = trial.suggest_int(name="batch_size", low=8, high=64, step=8)
        learning_rate = trial.suggest_float(name="lr", low=1e-5, high=1e-3, log=True)
        weight_decay = trial.suggest_float(name="weight_decay", low = 0.001, high = 0.08, log=True)
        momentum = trial.suggest_float(name="weight_decay", low = 0.001, high = 0.1, log=True)

        optimizer_name = trial.suggest_categorical(name="optimizer", choices=["Adam", "SGD", "RMSProp"])
    
        if optimizer_name == "Adam":
           optimizer = Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        trainer = CustomLoop(
            model_name=model_name,
            epochs=num_epochs,
            momentum=momentum,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            batch_size=batch_size,
            trials=trials
        )

        return trainer.train()


    logger.info("Searching for optimal values of the hyperparameters")
    study = create_study(directions=["minimize", "maximize", "maximize", "maximize"])  # Study that corresponds to the metrics we want to optimize
    study.optimize(func=objective_function, n_trials=trials, callbacks=[save_trial_callback], timeout=300)

    logger.info(f"Number of finished trials: {len(study.trials)}")
    best_trials = BestTrials(study=study)
    best_trials.save_best_trials()
    best_trials.display_best_trials()
    best_trials.log_to_comet()

