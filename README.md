# SongNet: Music Genre Classification using Machine Learning and Deep Learning

This repository contains a **Machine Learning Mini Project** reproducing *SongNet: Real-time Music Classification*, which explores machine learning and deep learning approaches for automatic music genre classification.

The project implements and compares five baseline models and one deep learning model:

- **Random Guess:** A chance-level baseline that predicts genres uniformly at random.
- **K-Nearest Neighbors (KNN):** Classifies music using feature similarity.
- **Logistic Regression:** A linear classification baseline.
- **Multilayer Perceptron (MLP):** A neural network trained on audio features.
- **Support Vector Machine (SVM):** A linear SVM baseline.
- **Convolutional Recurrent Neural Network (C-RNN):** Uses mel-spectrograms and recurrent layers to learn spatial and temporal patterns in music.

## Dataset

The project uses the **Free Music Archive (FMA-small)** dataset, containing 8,000 audio clips of 30 seconds each across eight balanced music genres.

The dataset is split into 70% training, 20% validation, and 10% testing. The baseline models use FMA-provided features, while the C-RNN learns from mel-spectrograms extracted from audio.

Dataset: [Free Music Archive (FMA)](https://github.com/mdeff/fma)

## Reported Results

Test accuracy on the 800-track test split. "Paper" values are the results reported in the original paper; "This reproduction" values come from the notebooks in `notebooks/`.

| Model | Paper | This reproduction | Macro F1 |
|---|---:|---:|---:|
| Random Guess | 12.50% | — | — |
| K-Nearest Neighbors | 36.38% | — | — |
| Logistic Regression | 42.25% | — | — |
| Multilayer Perceptron | 44.88% | 56.75% | 0.560 |
| Support Vector Machine | 46.38% | 53.50% | 0.508 |
| C-RNN | 65.23% | 63.25% | 0.626 |

- The C-RNN is within about 2 points of the paper and remains the best-performing model.
- The MLP and SVM score higher than reported, likely because their hyperparameters were tuned on our validation split and our train/val/test split differs from the paper's.
- Pop is the hardest genre for every model.

Detailed metrics, confusion matrices and training curves are in `results/`.

## Environment

Experiments use Python and scikit-learn for the baseline models and PyTorch for the C-RNN. Google Colab is used for development and GPU-accelerated training.

## Objective

The objective is to reproduce the models, evaluate their performance, and compare the results with the reported benchmarks.
