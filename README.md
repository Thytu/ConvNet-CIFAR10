# Blackfoot Code Interview : torch version
> Image Classification (CIFAR10) using a ConvNet

## Table of Contents
* [Setup](#setup)
* [Usage](#usage)
* [Room for Improvement](#todo)
* [Contact](#contact)

## Setup
To use the projet please:
- Install [dvc](https://dvc.org)
- Do `dvc repro`


## Usage
To reproduce the all dvc pipeline: `dvc repro`\
To rerun only the model training: `python3 src/main.py`\
To download the dataset: `python3 src/data_handler.py`

## TODO
- [ ] Display a graph showing the evolution of the accuracy and loss
- [ ] Use CML w/ dvc
- [ ] Add torch profiler to the project
- [x] Use Optuna to find the best hyperparameters
- [ ] Use MLFlow

## Contact
Created by [@Thytu](https://github.com/Thytu) - feel free to contact me!
