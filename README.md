# Neurosymbolic Explainable Concept Drift

This repository contains code for generating neuro-symbolic explanations to analyze and interpret concept drift in machine learning models.

## Tasks

- ✅ **Data Split & Concept Drift Simulation** — *Script to generate data splits and simulate concept drift* (Salvatore) — **Completed**
- ✅ **CNN Training** — *Train a convolutional neural network on the generated splits* (Pietro) — **Completed**
- ⏳ **Attribute Extraction & Rule Generation** — *Extract interpretable attributes and generate neuro-symbolic rules* (Francesco) — **In Progress**


# Setup
1) Create and Start a new environment:
```sh
conda create -n nesy-drift-env python=3.11
conda activate nesy-drift-env
```
2) Install the required packages:
```sh
pip install -r requirements.txt
```

# Getting Started

## 1. Dataset Splits Generation
This step is done by the [CelebA_create_dataset.ipynb](./CelebA_create_dataset.ipynb) notebook.

## 2. CNN training
This step is done by the [train_cnn.ipynb](./train_cnn.ipynb) notebook. Pre-trained weights are available [here](https://politoit-my.sharepoint.com/:f:/g/personal/pietro_basci_polito_it/EkEAn9inC-dIghsxU7p4RBABVPRBPpG25SnJxwk2Ox2S6w?e=3A5QCF).
