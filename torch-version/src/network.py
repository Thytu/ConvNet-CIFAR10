import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from typing import Tuple

# TODO: add logger to train and test functions

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

        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        logger.debug("Loading optimizer")
        self.optimizer = optimizer(self.parameters(), lr=lr)

        logger.info("Model successfully created")

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)

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


def _forward_pre_hook(module: nn.Module, inputs: Tuple[torch.Tensor]) -> torch.float32:
    """
    Cast inputs to float32 and send it to module.device
    """

    logger = logging.getLogger("data_handler:_forward_pre_hook")

    if not torch.is_tensor(inputs[0]):
        logger.debug("Returning casted tensor")

        return torch.tensor(inputs[0], dtype=torch.float32, device=module.device)

    logger.debug("Returning tensor")

    return inputs[0].to(module.device)


def load_network(device='cpu'):
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
