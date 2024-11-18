# Transformer_architecture_implementation

The goal of this repository is to create an implementation of the Transformer Architecture from scratch as it is described in the [Attention is all you need paper (Google 2017)](https://arxiv.org/pdf/1706.03762)

## Build Status

So far, the transformer architecture has been coded and is ready to be used ! 

For the next steps, latest updates about transformers such as [Differential Transformer (2024)](https://arxiv.org/abs/2410.05258) could be added (first elements of code can be found in the features/differential_transformer implementation branch). Also, other elements such as more loss functions and activation functions will be coded

Throughout the project, if you see any improvements that could be made in the code, do not hesitate to reach out at
Hippolyte.guigon@hec.edu. I will be delighted to get some insights !

## Code style

The all project was coded under PEP-8 (https://peps.python.org/pep-0008/) and flake8 (https://pypi.org/project/flake8/) compliancy. Such compliance is verified during commits with pre-commits file ```.pre-commit-config.yaml```

## Installation

* This project uses a specific conda environment, to get it, run the following command: ```conda env create -f transformer_architecture_environment.yml```

* To install all necessary libraries, run the following code: ```pip install -r requirements.txt```

* This project has its own package that is used. To get it, run the following command: ```python install setup.py```

## Screenshot

![alt text](https://raw.githubusercontent.com/HippolyteGuigon/Transformer_architecture_implementation/main/ressources/transformer_architecture.webp)

Transformer architecture as described in the original paper

## How to use ?

As a guideline for using this Transformer architecture, we'll use a French-English translation pipeline as an example (although, of course, this Transformer architecture can be used in any way you like).