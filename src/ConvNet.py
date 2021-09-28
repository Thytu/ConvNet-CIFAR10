import io

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import layers

import logging
from typing import Any, List, Tuple, Optional, Union, TypeVar


Callback = TypeVar('Callback', bound=type(lambda loss, acc : float))


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
            layers.ConvBlock(),
            nn.Flatten(),
            layers.LinearBlock(),
        ])

        self.lr = lr

        logger.debug("Loading optimizer")
        self.optimizer = optimizer(self.parameters(), lr=lr)

        logger.info("Model successfully created")

    def forward(self, t):
        for lay in self.layers:
            t = lay(t)

        return t

    def train_model(self, train_loader: torch.utils.data.DataLoader, Callbacks: Optional[Callback] = []) -> Tuple[float, float, List[Any]]:
        """
        Train the model using the train_loader.

            train_loader: data to train the model on

            Callbacks: execute a function at the end of each epoch.
                A callback take in parameters loss: float, acc: float
                A callback can return any value type and will be return at the end of this function
        """
        logger = logging.getLogger("ConvNet:train_model")

        self.train()

        total = 0
        correct = 0
        running_loss = 0.0
        callbacks_returns = []

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

        for callback in Callbacks:
            logger.debug("Calling callback")
            callbacks_returns.append(callback(loss, (predicted == labels).sum().item() / labels.size(0)))
        logger.debug("Callbacks successfully caled")

        logger.info("Model successfully trained")

        return running_loss / total, 100 * correct / total, callbacks_returns

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

    def select_optimizer(self, optimizer: Union[str, torch.optim.Optimizer]):
        """
        Select model optimizer
            optimizer : str | torch.optim.Optimizer
                exemple: self.select_optimizer("Adam")
        """

        logger = logging.getLogger("ConvNet:select_optimize")

        if type(optimizer) == torch.optim.Optimizer:
            logger.debug("Loading optimizer using optim.Optimizer")
            self.optimizer = optimizer(self.parameters, self.lr)

        logger.debug("Loading optimizer using str")
        self.optimize = getattr(torch.optim, optimizer)(self.parameters(), lr=self.lr)

        logger.info("Optimizer successfully loaded")


def _forward_pre_hook(module, inputs: Tuple[torch.Tensor]) -> torch.Tensor:
    """
    Cast inputs to float32 and send it to module.device

    Types depends if the model is in eager mode, traced or scripted.
    For more informations, please refer to : https://pytorch.org/docs/stable/jit_language_reference.html
    """

    # logger = logging.getLogger("data_handler:_forward_pre_hook")

    if not isinstance(inputs[0], torch.Tensor):
        # logger.debug("Returning casted tensor")

        return torch.tensor(inputs[0], dtype=torch.float32, device=module.device)

    # logger.debug("Returning tensor")

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


def static_quantize_model(model: nn.Module, data_loader: torch.utils.data.DataLoader) -> nn.Module:
    """
    Quantizes the weights and activations of the model and fuses activations into preceding layers where possible
    """

    logger = logging.getLogger("data_handler:static_quantize_model")

    SERVER_INFERENCE_CONFIG = 'fbgemm'

    logger.debug("Adding QuantStub & DeQuantStub layers to model")
    model.layers.insert(0, torch.quantization.QuantStub())
    model.layers.append(torch.quantization.DeQuantStub())

    logger.debug("Setting model to eval mode")
    model.eval()

    logger.debug("Setting model qconfig")
    model.qconfig = torch.quantization.get_default_qconfig(SERVER_INFERENCE_CONFIG)

    logger.debug("Fuzing model layers")
    model = torch.quantization.fuse_modules(model, [
        ['layers.1.block.0', 'layers.1.block.1'], # Conv, BN
        ['layers.3.block.1', 'layers.3.block.2'], # Line, RL
        ['layers.3.block.4', 'layers.3.block.5'], # Line, RL
    ])

    logger.debug("Preparing model to quantize")
    model = torch.quantization.prepare(model)

    logger.debug("Preparing model")
    for (images, _) in data_loader:
        model.forward(images)

    logger.debug("Quantizing model")
    model = torch.quantization.convert(model)

    logger.info("Model successfully quantized")

    return model


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


def load_jit_module(path: str) -> torch.jit.RecursiveScriptModule:
    """
    Load scripted module from path file and return it
    """

    logger = logging.getLogger("data_handler:load_jit_module")

    logger.debug("Opening prived file")
    with open(path, 'rb') as f:
        logger.debug("Creating buffer")
        buffer = io.BytesIO(f.read())

    logger.debug("Loading module")
    module = torch.jit.load(buffer)
    logger.info("Module successfully loaded")

    return module


def save_state_dict(module: nn.Module, path: str) -> None:
    """
    Save provided model's state dict to path file
    """
    logger = logging.getLogger("data_handler:save_state_dict")

    logger.debug("Saving module")
    torch.save(module.state_dict(), path)
    logger.info("Module successfully saved")


def load_state_dict(module: nn.Module, path: str) -> nn.Module:
    """
    Load provided model's state dict from path file
    """

    logger = logging.getLogger("data_handler:load_state_dict")

    logger.debug("Loading module")
    module.load_state_dict(torch.load(path))
    logger.info("Module successfully loaded")

    return module


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
