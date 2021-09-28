# Code Interview
> An old code interview reworked to test multiple torch concepts.\
> I was supposed to write a CIFAR10 classifier using a ConvNet, you can still find the model [here](src/ConvNet.py).

## Table of Contents
* [Setup](#setup)
* [Usage](#usage)
* [Architecture](#architecture)
* [Room for Improvement](#todo)
* [Contact](#contact)

## Setup
To use the projet please:
- Install [dvc](https://dvc.org)
- Do `dvc repro`


## Usage
To reproduce the all dvc pipeline: `dvc repro`\
To download the dataset: `python3 src/data_handler.py`\
To find the best learning rate and optimizer according to [Optuna](https://optuna.org): `python3 src/hyperparameters.py`\
To rerun only the model training: `python3 src/main.py`


## Architecture
This project can be read in mutliple parts:
* The [dvc pipeline](dvc.yaml) to order every step
* The [logs handler](src/logs_handler.py) to load the log level regarding var env
* The [yaml handler](src/yaml_handler.py) to load and write yaml
* The [data handler](src/data_handler.py) to download and load CIFAR10
* The [hyper-paramerters tuner](src/hyperparameters.py) using Optuna to find the best learning rate and optimizer
* The [model](src/ConvNet.py) and its [layers](src/layers.py) to classify CIFAR10\
  The model include multiple torch concepts: 
  - [hooks](https://pytorch.org/docs/stable/generated/torch.nn.modules.module.register_module_forward_pre_hook.html) to cast every inputs to torch.Tensor and send it to the selected device
  - [Torchscript and jit](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) to optimize the model for deployment by scripting the model.\
    Note: Here, we do not use [torch.trace](https://pytorch.org/docs/stable/generated/torch.trace.html)
  - [pruning](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) to optimize the model for deployment by pruning the model.
  - [quantization](https://pytorch.org/docs/stable/quantization.html) to optimize the model for deployment by casting tensors to `uint8`.
* The [main script](src/main.py) to run the training loop and save resuling models


## TODO
- [ ] Add torch profiler to the project
- [ ] Use CML w/ dvc
- [x] Use Optuna to find the best hyperparameters
  - [ ] Write module docstring
- [ ] Use MLFlow

## Contact
Created by [@Thytu](https://github.com/Thytu) - feel free to contact me!
