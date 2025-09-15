# Explaining Concept Drift via Neuro-Symbolic Rules

This repository contains code of the paper *"Explaining Concept Drift via Neuro-Symbolic Rules"* accepted at the TRUST-AI workshop, organized as part of the European Conference on artificial intelligence (ECAI 2025).

This project investigates the use of neuro-symbolic explanations to analyze and interpret concept drift in machine learning models.

## Table of Contents
- [Setup](#setup)
- [Getting Started](#getting-started)
  - [1. Dataset Splits Generation](#1-dataset-splits-generation)
  - [2. CNN training](#2-cnn-training)
  - [3. GradCAM Analysis](#3-gradcam-analysis)
  - [4. Rule Extraction (optimal assumption)](#4-rule-extraction-(optimal-assumption))
  - [5. Concept Extractor (correlation-based)](#5-concept-extractor-(correlation-based))
  - [6. Concept Extractor (MLP-based)](#6-concept-Extractor-(mlp-based))
- [Future Developments](#future-developments)
- [References](#references)
- [Authors](#authors)

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

## Getting Started

### 1. Dataset Splits Generation
This step is done by the [1_CelebA_create_dataset.ipynb](./1_CelebA_create_dataset.ipynb) notebook.

### 2. CNN training
This step is done by the [2_CelebA_train_cnn.ipynb](./2_CelebA_train_cnn.ipynb) notebook. Pre-trained weights are available [here](https://politoit-my.sharepoint.com/:f:/g/personal/pietro_basci_polito_it/EkEAn9inC-dIghsxU7p4RBABVPRBPpG25SnJxwk2Ox2S6w?e=3A5QCF).

### 3. GradCAM Analysis
This step is done by the [3_CelebA_cnn_gradcam_analysis.ipynb](./3_CelebA_cnn_gradcam_analysis.ipynb) notebook.

### 4. Rule Extraction (optimal assumption)
This step is done by the [4_eric_optimal_naming.ipynb](./4_eric_optimal_naming.ipynb) notebook.

### 5. Concept Extractor (correlation-based)
This step is done by the [5_eric_corr_naming.ipynb](./5_eric_corr_naming.ipynb) notebook.

### 6. Concept Extractor (MLP-based)
This step is done by the [6_train_concepts_naming.ipynb](./6_train_concepts_naming.ipynb) notebook. Pre-trained weights are available [here](https://politoit-my.sharepoint.com/:u:/g/personal/pietro_basci_polito_it/EUkKexOIsJtIrNK1a3oEznoBTduwvP_BdTya5MEG7ESH6Q?e=xYopdC).

## Future Developments


## References
If you find this work useful or use this repo, please cite the following paper:

```bibtex

```

## Authors

- **Pietro Basci**, *Politecnico di Torino* 
- **Salvatore Greco**, *Politecnico di Torino* - [Homepage](https://grecosalvatore.github.io/) - [GitHub](https://github.com/grecosalvatore) - [Twitter](https://twitter.com/_salvatoregreco)
- **Francesco Manigrasso**, *Politecnico di Torino*
- **Tania Cerquitelli**, *Politecnico di Torino* - [Homepage](https://dbdmg.polito.it/dbdmg_web/people/tania-cerquitelli/)
- **Lia Morra**, *Politecnico di Torino*