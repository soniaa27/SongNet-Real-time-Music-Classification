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

| Archive | Size | Needed by |
|---|---|---|
| `fma_metadata.zip` | 342 MB | all notebooks |
| `fma_small.zip` | 7.2 GB | `CRNN.ipynb` only |

The dataset is not stored in this repository.

## Reported Results

Test accuracy on the 800-track test split. "Paper" values are the results reported in the original paper; "This reproduction" values come from the notebooks in `notebooks/`.

| Model | Paper | This reproduction | Macro F1 |
|---|---:|---:|---:|
| Random Guess           | 12.50% | 12.75% | 0.128 |
| K-Nearest Neighbors    | 36.38% | 51.75% | 0.500 |
| Logistic Regression    | 42.25% | 54.12% | 0.528 |
| Multilayer Perceptron | 44.88% | 56.75% | 0.560 |
| Support Vector Machine | 46.38% | 53.50% | 0.508 |
| C-RNN | 65.23% | 63.25% | 0.626 |

- The C-RNN is within about 2 points of the paper and remains the best-performing model.
- The MLP and SVM score higher than reported, likely because their hyperparameters were tuned on our validation split and our train/val/test split differs from the paper's.
- Pop is the hardest genre for every model.

Detailed metrics, confusion matrices and training curves are in `results/`.

## Repository Structure

```
├── notebooks/
│   ├── random_knn_logreg.ipynb   # Random Guess, KNN, Logistic Regression
│   ├── mlp_svm.ipynb             # Multilayer Perceptron, Linear SVM
│   └── CRNN.ipynb                # Mel-spectrograms + C-RNN
├── results/                      # Metrics, confusion matrices, training curves per notebook
├── splits.csv                    # Shared train/val/test split used by every model
└── README.md
```

## Setup and Usage

**Requirements:** Python 3.11 and about 15 GB of free disk space for the C-RNN (audio plus spectrogram cache). The baseline notebooks need about 1 GB.

1. Clone the repository and create an environment:
   ```bash
   git clone https://github.com/soniaa27/SongNet-Real-time-Music-Classification.git
   cd SongNet-Real-time-Music-Classification
   python3.11 -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install numpy pandas scikit-learn matplotlib seaborn librosa soundfile torch joblib jupyter
   ```
2. Download the FMA data into `data/` (local runs never download automatically):
   ```bash
   mkdir -p data && cd data
   curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
   unzip fma_metadata.zip fma_metadata/tracks.csv fma_metadata/features.csv
   # C-RNN only (7.2 GB):
   curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip
   unzip fma_small.zip
   cd ..
   ```
   You should now have `data/fma_metadata/` and `data/fma_small/`. If `unzip` fails on the large archive, use `7z x fma_small.zip`.
3. Use the committed split so your results match ours:
   ```bash
   cp splits.csv data/splits.csv
   ```
4. Start Jupyter and run the notebooks top to bottom:
   ```bash
   jupyter lab
   ```
   Outputs are written to `results/` and `models/` in the repository folder.

**Hardware:** the C-RNN uses an NVIDIA GPU if available, Apple Silicon (MPS) on a Mac, and otherwise the CPU, which is slow. The baselines run in minutes on any laptop.

### Shared split

All six models use `splits.csv`: 7,994 FMA-small tracks (6 corrupt files removed), split 70/20/10 with seed 42 into 5,595 / 1,599 / 800 tracks. Each notebook prints the file's SHA-256 checksum; it should start with `77644b59`.

### Alternatively, run on Google Colab

Open [Google Colab](https://colab.research.google.com/) → **File → Open notebook → GitHub**, paste this repository's URL and pick a notebook. For `CRNN.ipynb`, select **Runtime → Change runtime type → T4 GPU**. Then click **Runtime → Run all** and allow Google Drive access when prompted. The notebook downloads the dataset itself and saves outputs to `MyDrive/SongNet/`; Colab already has every required library.

## Environment

Experiments use Python and scikit-learn for the baseline models and PyTorch for the C-RNN. Google Colab is used for development and GPU-accelerated training.

## Objective

The objective is to reproduce the models, evaluate their performance, and compare the results with the reported benchmarks.
