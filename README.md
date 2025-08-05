# Neurosymbolic Explainable Concept Drift

This repository contains code for generating neuro-symbolic explanations to analyze and interpret concept drift in machine learning models.

## Tasks

- ✅ **Data Split & Concept Drift Simulation** — *Script to generate data splits and simulate concept drift* — **Completed**
- ✅ **CNN Training** — *Train a convolutional neural network on the generated splits* — **Completed**
- ✅ **GradCAM Analysis** — *GradCAM analysis on convolutional neural network* — **Completed**
- ⏳ **Attribute Extraction & Rule Generation** — *Extract interpretable attributes and generate neuro-symbolic rules* — **In Progress**


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
This step is done by the [1_CelebA_create_dataset.ipynb](./1_CelebA_create_dataset.ipynb) notebook.

## 2. CNN training
This step is done by the [2_CelebA_train_cnn.ipynb](./2_CelebA_train_cnn.ipynb) notebook. Pre-trained weights are available [here](https://politoit-my.sharepoint.com/:f:/g/personal/pietro_basci_polito_it/EkEAn9inC-dIghsxU7p4RBABVPRBPpG25SnJxwk2Ox2S6w?e=3A5QCF).

## 3. GradCAM Analysis
This step is done by the [3_CelebA_cnn_gradcam_analysis.ipynb](./3_CelebA_cnn_gradcam_analysis.ipynb) notebook.

## 4. Rule Extraction (optimal assumption)
This step is done by the [4_eric_optimal_naming.ipynb](./4_eric_optimal_naming.ipynb) notebook.

## 5. Concept Extractor (correlation-based)
This step is done by the [5_eric_corr_naming.ipynb](./5_eric_corr_naming.ipynb) notebook.

## 6. Concept Extractor (MLP-based)
This step is done by the [6_train_concepts_naming.ipynb](./6_train_concepts_naming.ipynb) notebook. Pre-trained weights are available [here](https://politoit-my.sharepoint.com/:u:/g/personal/pietro_basci_polito_it/EUkKexOIsJtIrNK1a3oEznoBTduwvP_BdTya5MEG7ESH6Q?e=xYopdC).
