import logging
import logs_handler

import os
from tqdm import tqdm
from time import time
from tensorboardX import SummaryWriter

import ConvNet
from data_handler import load_dataset
from yaml_handler import load_yaml


def save_network(network: 'nn.Module', folder_path: str, model_id: str, device='cpu') -> None:
    logger.debug("Saving state dicts")
    ConvNet.save_state_dict(network, fr'{folder_path}/{model_id}_{device}_sdc.pt')
    logger.info("State dicts successfully saved")

    logger.debug("Saving jit models")
    ConvNet.save_jit_module(ConvNet.convert_to_jit(network), fr'{folder_path}/{model_id}_{device}_jit.pt')
    logger.info("jit models successfully saved")


if __name__ == '__main__':
    YAML = load_yaml("./config.yaml")

    EPOCH = YAML['main']['epoch']
    MODELS_DIR = YAML['main']['models_dir']

    DATA_DIR = YAML['global']['data_dir']
    LOG_PATH = YAML['global']['log_path']
    MODEL_HP_PATH = YAML['model']['model_hp_path']

    MODEL_YAML = load_yaml(MODEL_HP_PATH)
    LR = MODEL_YAML['lr']
    OPTIMIZER = MODEL_YAML['optimizer']

    DEVICE = ConvNet.select_device()

    logging.basicConfig(filename=LOG_PATH, level=logs_handler.get_log_level())
    logger = logging.getLogger("main:main")

    train_loader, val_loader, test_loader = load_dataset(DATA_DIR)
    model = ConvNet.load_network(device=DEVICE)
    best_test_acc = None

    model.lr = LR
    model.select_optimizer(OPTIMIZER)

    writer = SummaryWriter()

    for epoch in tqdm(range(EPOCH)):
        train_loss, train_acc, _ = model.train_model(train_loader)
        test_loss, test_acc = model.test_model(val_loader)

        if best_test_acc is None or test_acc < best_test_acc:
            logger.debug("Changing best model at epoch:", epoch)

            best_model = model
            best_test_acc = test_acc

            logger.info("Best model changed at epoch %i\tacc:%d", epoch, best_test_acc)

        writer.add_scalars('ConvNet/train', {'acc': train_acc, 'loss': train_loss}, epoch)
        writer.add_scalars('ConvNet/val', {'acc': test_acc, 'loss': test_loss}, epoch)

        if (epoch - 1) % 10 == 0:
            print(f"Epoch {epoch}\t\tloss:{train_loss:.3f}\tacc:{train_acc}\t\ttest_loss:{test_loss:.3f}\ttest_acc:{test_acc}")

    print('\033[1m' + "MODEL"+ '\033[0m')
    best_model.detailed_test(test_loader)
    bst_test_loss, bst_test_acc = best_model.test_model(test_loader)
    print(f"model loss:{bst_test_loss:.3f}\tmodel acc:{bst_test_acc}")

    pruned_model = ConvNet.prune_model(best_model)

    print('\033[1m' + "PRUNED"+ '\033[0m')
    pruned_model.detailed_test(test_loader)
    prn_test_loss, prn_test_acc = pruned_model.test_model(test_loader)
    print(f"model loss:{prn_test_loss:.3f}\tmodel acc:{prn_test_acc}")

    model_int8 = ConvNet.static_quantize_model(best_model, train_loader)

    print('\033[1m' + "QUANTIZED"+ '\033[0m')
    model_int8.detailed_test(test_loader)
    prn_test_loss, prn_test_acc = model_int8.test_model(test_loader)
    print(f"model loss:{prn_test_loss:.3f}\tmodel acc:{prn_test_acc}")

    if not os.path.exists(MODELS_DIR):
        logger.debug("Creating %s folder", MODELS_DIR)
        os.mkdir(MODELS_DIR)
        logger.info("Directory %s created successfully", MODELS_DIR)

    save_network(best_model, MODELS_DIR, fr"{time()}_{bst_test_acc}", DEVICE)
