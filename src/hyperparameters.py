"""
TODO
"""
import optuna
import logging
import torch.optim as optim
from concurrent.futures import ThreadPoolExecutor

import ConvNet
import data_handler
import yaml_handler
import logs_handler

def objective(trial):
    """
    Do one trial to try to find the best hyperparameters.

    For more information about how this function is working please refer to: https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
    """

    DEVICE = ConvNet.select_device()

    EPOCH = YAML['hyperparameters']['epoch']

    LOW_LR = YAML['hyperparameters']['low_lr']
    HGT_LR = YAML['hyperparameters']['hight_lr']
    OPTS = YAML['hyperparameters']['optimizers']

    logger = logging.getLogger("hyperparameters:objective")

    network = ConvNet.load_network(DEVICE)

    logger.debug("Suggesting optimizer")
    suggested_optimizer_name = trial.suggest_categorical("optimizer", OPTS)

    logger.debug("Suggesting learning rate")
    suggested_learning_rate = trial.suggest_float("lr", LOW_LR, HGT_LR, log=True)

    logger.debug("Selecting hyperparameters:\tlr:%d\topt:%s", suggested_learning_rate, suggested_optimizer_name)
    network.optimize = getattr(optim, suggested_optimizer_name)(network.parameters(), lr=suggested_learning_rate)

    for epoch in range(EPOCH):
        loss, _, _ = network.train_model(TRAIN_LOADER)

        logger.debug("Reporting trial at epoch %i", epoch)
        trial.report(loss, epoch)

        logger.debug("Checking if trial need to be pruned")
        if trial.should_prune():
            logger.debug("Prung trial")
            raise optuna.exceptions.TrialPruned()

    logger.info("Trial successfully completed")

    return loss

if __name__ == '__main__':
    YAML = yaml_handler.load_yaml("./config.yaml")

    DATA_DIR = YAML['global']['data_dir']
    LOG_PATH = YAML['global']['log_path']
    DB_PATH = YAML['hyperparameters']['db_path']
    TIMEOUT = YAML['hyperparameters']['timeout']
    N_TRIALS = YAML['hyperparameters']['n_trials']
    MODEL_HP_PATH = YAML['model']['model_hp_path']

    logging.basicConfig(filename=LOG_PATH, level=logs_handler.get_log_level())
    logger = logging.getLogger("hyperparameters:main")

    logger.debug("Creating study")
    study = optuna.create_study(
        direction='minimize',
        study_name='lr_n_optz',
        storage=DB_PATH,
        load_if_exists=True,
    )

    TRAIN_LOADER, _, _ = data_handler.load_dataset(DATA_DIR)

    logger.debug("Optimizing")
    study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT, show_progress_bar=True)

    logger.info("Hyperparameters successfully optimized")

    logger.debug("Changing yaml values")
    hyperparameters_yaml = {
        "lr": study.best_trial.params['lr'],
        "optimizer": study.best_trial.params['optimizer'],

    }
    # YAML['model']['lr'] = study.best_trial.params['lr']
    # YAML['model']['optimizer'] = study.best_trial.params['optimizer']

    logger.debug("Writing results")
    yaml_handler.write_yaml(hyperparameters_yaml, MODEL_HP_PATH)
    # yaml_handler.write_yaml(YAML, "./config.yaml")

    logger.debug("Getting statistics")
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    completed_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    logger.debug("""
    Study statistics
    \tNumber of finished trials: %i
    \tNumber of pruned trials: %i
    \tNumber of complete trials: %i""", len(study.trials), len(pruned_trials), len(completed_trials))


    logger.info("Results successfully writed")