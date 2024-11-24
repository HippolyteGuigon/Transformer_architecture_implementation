#!/bin/bash

pip install -r requirements.txt
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download -d dhruvildave/en-fr-translation-dataset
mkdir -p data
mkdir -p models
unzip en-fr-translation-dataset.zip -d data/
rm en-fr-translation-dataset.zip
python setup.py install
python transformer_architecture/traduction_test/traduction_pipeline.py
