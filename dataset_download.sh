#!/bin/bash

mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d dhruvildave/en-fr-translation-dataset
mkdir -p data
unzip en-fr-translation-dataset.zip -d data/
