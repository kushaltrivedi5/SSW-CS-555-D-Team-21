# EEG Classification using Deep Learning

## Overview
This repository contains code for a simple deep learning model to classify EEG (Electroencephalography) signals into different categories. The model architecture consists of an autoencoder followed by a classifier. The data used for training and testing are EEG signals with dimensions [4500, 64, 795] and [750, 64, 795] respectively.

## Requirements
- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- torchviz

## Data
The dataset used for training and testing can be downloaded from [this Google Drive link](https://drive.google.com/drive/folders/1ykR-mn4d4KfFeeNrfR6UdtebsNRY8PU2?usp=sharing). 
After downloading, place the data files (`train_data.npy`, `test_data.npy`, `train_label.npy`, `test_label.npy`) in the `./data` directory.


## Model Architecture
The model architecture consists of an autoencoder followed by a classifier:
- Encoder:
  - Input: 64 channels * 795 time points
  - Layers: Linear (64 * 795 -> 512) -> ReLU -> Linear (512 -> 256) -> ReLU -> Linear (256 -> 128) -> ReLU
- Classifier:
  - Input: 128 features
  - Output: LogSoftmax output for 5 categories

## Training
- Loss function: Negative Log-Likelihood Loss (NLLLoss)
- Optimizer: Adam optimizer with learning rate 0.0001
- Number of epochs: 30
- Batch size: 64

## Evaluation
After training, the model is evaluated on the test set, and the test accuracy is calculated. The accuracy is then displayed as a bar chart.

## Model Visualization
The computational graph for the model is generated using torchviz and displayed for visualization.
