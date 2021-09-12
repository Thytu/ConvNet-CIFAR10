import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

import logging
from typing import Tuple


class ConvNet(nn.Module):
    def __init__(self, device: str, optimizer=torch.optim.Adam, criterion=nn.CrossEntropyLoss(), lr=0.0001):
        super().__init__()

        self.CLASSES = (
            'plane', 'car',
            'bird', 'cat',
            'deer', 'dog',
            'frog', 'horse',
            'ship', 'truck'
        )
        logger = logging.getLogger("ConvNet:__init__")

        self.device = device
        self.criterion = nn.CrossEntropyLoss()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
                nn.BatchNorm2d(num_features=6),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ),

            nn.Flatten(),

            nn.Sequential(
                nn.Dropout(p=0.25),
                nn.Linear(1176, 500),
                nn.ReLU(),

                nn.Dropout(p=0.25),
                nn.Linear(500, 150),
                nn.ReLU(),

                nn.Linear(150, 10)
            ),
        ])

        logger.debug("Loading optimizer")
        self.optimizer = optimizer(self.parameters(), lr=lr)

        logger.info("Model successfully created")

    def forward(self, t):
        for idx, lay in enumerate(self.layers):
            t = lay(t)

        return t

    def train_model(self, train_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Train the model using the train_loader.
        """
        logger = logging.getLogger("ConvNet:train_model")

        self.train()

        total = 0
        correct = 0
        running_loss = 0.0

        for idx, data in enumerate(train_loader, 0):
            logger.debug("Training the model (%i)", idx)

            self.optimizer.zero_grad()

            logger.debug("Dispatching data")
            images, labels = data

            logger.debug("Forwarding data")
            outputs = self.forward(images)

            logger.debug("Calculating the predictions")
            _, predicted = torch.max(outputs.data, 1)

            logger.debug("Calculating the corrects predictions")
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            logger.debug("Calculating the loss")
            loss = self.criterion(outputs, labels)

            logger.debug("Backwarding grads")
            loss.backward()

            logger.debug("Making one step w/ the optimizer")
            self.optimizer.step()

            running_loss += loss.item()

        logger.info("Model successfully trained")

        return running_loss / total, 100 * correct / total

    def test_model(self, test_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Test the model using the train_loader.
        """
        logger = logging.getLogger("ConvNet:test_model")

        self.eval()

        total = 0
        correct = 0
        running_loss = 0.0

        with torch.no_grad():
            for data in test_loader:

                logger.debug("Dispatching data")
                images, labels = data

                logger.debug("Forwarding data")
                outputs = self.forward(images)

                logger.debug("Calculating the predictions")
                _, predicted = torch.max(outputs.data, 1)

                logger.debug("Calculating the corrects predictions")
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                logger.debug("Calculating the loss")
                running_loss += self.criterion(outputs, labels).item()

        logger.info("Model successfully tested")

        return running_loss / total,  100 * correct / total

    def detailed_test(self, data_loader: torch.utils.data.DataLoader) -> None:
        """
        Show model accuracy for each class
        """
        logger = logging.getLogger("ConvNet:detailed_test")

        self.eval()

        correct_pred = {class_name: 0 for class_name in self.CLASSES}
        total_pred = {class_name: 0 for class_name in self.CLASSES}

        with torch.no_grad():
            for data in data_loader:

                logger.debug("Dispatching data")
                images, labels = data

                logger.debug("Forwarding data")
                outputs = self.forward(images)

                logger.debug("Calculating the predictions")
                _, predictions = torch.max(outputs, 1)

                logger.debug("Calculating accuracy for each class")
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[self.CLASSES[label]] += 1

                    total_pred[self.CLASSES[label]] += 1

        for class_name, correct_count in correct_pred.items():
            logger.debug("Printing accuracy for %s class", class_name)

            print(f"Accuracy for class {class_name:5s} is: {100 * float(correct_count) / total_pred[class_name]:.1f} %")

        logger.info("Model successfully tested in details")


def _forward_pre_hook(module, inputs: Tuple[torch.Tensor]) -> torch.Tensor:
    """
    Cast inputs to float32 and send it to module.device

    Types depends if the model is in eager mode, traced or scripted.
    For more informations, please refer to : https://pytorch.org/docs/stable/jit_language_reference.html
    """

    if not isinstance(inputs[0], torch.Tensor):
        logger.debug("Returning casted tensor")

        return torch.tensor(inputs[0], dtype=torch.float32, device=module.device)

    return inputs[0].to(module.device)


def load_network(device='cpu') -> nn.Module:
    """
    Load the network, its hooks and return it
    """

    logger = logging.getLogger("data_handler:load_network")

    logger.debug("Loading network")
    net = ConvNet(device)

    logger.debug("Registering forward pre-hook")
    net.register_forward_pre_hook(_forward_pre_hook)

    logger.info("Network successfully loaded")

    return net


def prune_model(model: nn.Module) -> nn.Module:
    """
    Create a new pruned model from the provided model
    """

    logger = logging.getLogger("data_handler:create_pruned_model")

    logger.debug("Loading network")
    pruned_model = load_network()

    logger.debug("Loading state dict")
    pruned_model.load_state_dict(model.state_dict())

    for _, module in pruned_model.named_modules():

        if isinstance(module, torch.nn.Conv2d):
            logger.debug("Pruning Conv2D")
            prune.l1_unstructured(module, name='weight', amount=0.2)

        elif isinstance(module, torch.nn.Linear):
            logger.debug("Pruning Linear")
            prune.l1_unstructured(module, name='weight', amount=0.4)

    logger.info("Model successfully pruned")

    return pruned_model


def convert_to_jit(module: nn.Module) -> torch.jit.RecursiveScriptModule:
    """
    Script a torch module using torch.jit.script
    To trace your module, please write a custom function using torch.jit.trace
    """

    logger = logging.getLogger("data_handler:convert_to_jit")

    logger.debug("Scripting module")

    module = torch.jit.script(module)

    logger.info("Module successfully scripted")

    return module


def save_state_dict(module: nn.Module, path: str) -> None:
    """
    Save provided model's state dict to path file
    """
    logger = logging.getLogger("data_handler:save_state_dict")

    logger.debug("Saving module")
    torch.save(module.state_dict(), path)
    logger.debug("Module successfully saved")


def save_jit_module(module: torch.jit.RecursiveScriptModule, path: str) -> None:
    """
    Save scripted torch module to path file
    """

    logger = logging.getLogger("data_handler:save_jit_module")

    assert isinstance(module, torch.jit.RecursiveScriptModule), "Model need to be of instance ScriptModule, please be sure to trace the model or script it"

    logger.debug("Saving module")
    torch.jit.save(module, path)
    logger.debug("Module successfully saved")


def select_device() -> torch.device:
    """
    Return a torch.device selecting GPU if available, cpu otherwise.

    To use multiple GPUs, please write a custom function using nn.DataParallel
    For more informations please refer to https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
    """

    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
