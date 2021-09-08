import logging
import logs_handler

import os
from tqdm import tqdm
from time import time

from data_handler import load_dataset
from yaml_handler import load_yaml
from network import load_network, prune_model


if __name__ == '__main__':
    YAML = load_yaml("./config.yaml")

    EPOCH = YAML['main']['epoch']
    MODELS_DIR = YAML['main']['models_dir']
    DATA_DIR = YAML['global']['data_dir']
    LOG_PATH = YAML['global']['log_path']

    logging.basicConfig(filename=LOG_PATH, level=logs_handler.get_log_level())
    logger = logging.getLogger("main:main")

    train_loader, val_loader, test_loader = load_dataset(DATA_DIR)
    model = load_network()
    best_test_acc = None

    for epoch in tqdm(range(EPOCH)):
        train_loss, train_acc = model.train_model(train_loader)
        test_loss, test_acc = model.test_model(val_loader)

        if best_test_acc is None or test_acc < best_test_acc:
            logger.debug("Changing best model at epooch:", epoch)

            best_model = model
            best_test_acc = test_acc
            
        if (epoch - 1) % 10 == 0:
            print(f"Epoch {epoch}\t\tloss:{train_loss:.3f}\tacc:{train_acc}\t\ttest_loss:{test_loss:.3f}\ttest_acc:{test_acc}")

    print('\033[1m' + "UNPRUNED"+ '\033[0m')
    best_model.detailed_test(test_loader)
    test_loss, test_acc = best_model.test_model(test_loader)
    print(f"best model loss:{test_loss:.3f}\tmodel acc:{test_acc}")

    pruned_model = prune_model(best_model)   

    print('\033[1m' + "PRUNED"+ '\033[0m')
    pruned_model.detailed_test(test_loader)
    test_loss, test_acc = pruned_model.test_model(test_loader)
    print(f"model loss:{test_loss:.3f}\tmodel acc:{test_acc}")

    if not os.path.exists(MODELS_DIR):
        logger.debug("Creating %s folder", MODELS_DIR)
        os.mkdir(MODELS_DIR)
        logger.info("Directory %s created successfully", MODELS_DIR)

    ts = time()

    logger.debug("Saving models")
    best_model.save_dict(fr"{MODELS_DIR}/{ts}_best.pt")
    pruned_model.save_dict(fr"{MODELS_DIR}/{ts}_pruned.pt")
    logger.debug("Models saved")
