# Breast cancer molecular subtype + PET/MRI

**Manuscript title:** Non-invasive prediction of breast cancer molecular
subtypes by AI-driven analysis of [18F]FDG PET/MRI data
<!-- **Journal:** Cell Reports Medicine -->
<!--**Authors:** [Author names]  -->
<!--**Corresponding author:** [Name and institutional email]  -->
<!-- **Repository version:** [Version number] -->
**Last updated:** [2026-10-27]

---

## 1. Overview

This repository contains the source code and supporting materials used in 
the study:

> **Non-invasive prediction of breast cancer molecular subtypes by AI-driven
analysis of [18F]FDG PET/MRI data**

The purpose of this repository is to document the implementation of the
machine-learning methods described in the manuscript and facilitate
reproducibility of the reported results.

The repository includes code for data preprocessing, model development, model
training, inference, and evaluation, as well as explainability analysis.

---

## 2. Repository Structure

```text
.
├── README.md
├── environment.yml
├── .gitignore
├── src/
│   ├── data_preprocessing
│   │    ├── ...
│   ├── ...
├── data/
│   ├── 006_LumA_LumB_BBox_liminal_tvt_prior.json
│   ├── ...
├── models/
│   ├── random_forest
│   │     ├── ...
│   └── resnet
│        ├── ...
└── notebooks/
   ├── demonstration_explainability.ipynb
   ├── ...
```

---

## 3. Software Requirements

The code was developed and tested using the following environment:

- **Operating system:** Windows 11
- **Programming language:** Python 3.12
- **Machine-learning
  framework:** PyTorch / scikit-learn
- **Hardware:** Intel i9 14900KF, NVIDIA GeForce RTX 5090, 192 GB RAM
- **CUDA version:** 13.1

### Dependencies

Install the required dependencies using:

```bash
conda env create -f environment.yml
conda activate bcsp
```

---

## 4. Data Availability and Preparation

The analyses described in the manuscript
use PET/MRI data acquired from a prospective trial. Due to privacy concerns, 
the data cannot be made publicly available, but can be accessed upon 
reasonable request. Meta-data of the dataset can be seen in file 

```text
/data/006_LumA_LumB_BBox_liminal_tvt_prior.json.
```

---

## 5. Model Architecture

The repository implements the following model (s):

### 5.1 Random Forest

### Model inputs

- **Input data:** Radiomics from PET/MRI tumor lesions
- **Input dimensions:** n_samples x 122
- **Preprocessing requirements:** Normalization

### Model outputs

- **Prediction target:** Breast cancer molecular subtype (Luminal A, Luminal B)
- **Output
  format:** Class probabilities

2. 3D ResNet Ensemble

### Model inputs

- **Input data:** Multi-channel bounding box around tumor lesion ROI
- **Input dimensions or feature representation:** h x w x d x 1 for T2W, ADC,
  PET, h x w x d x 7 for DCE-MRI
- **Preprocessing requirements:** Local z-score normalization for MRI, 
  global normalization for PET

### Model outputs

- **Prediction target:** Breast cancer molecular subtype (Luminal A, Luminal B)
- **Output
  format:** Class probabilities

---

## 6. Loading the Trained Model, Making Predictions, and Evaluation

If trained model weights are available, they can be loaded using the
corresponding model implementation. See files

```text
/notebooks/demonstration_random_forest.ipynb
```

and

```text
/notebooks/demonstration_resnet.ipynb
```

for examples of model loading, inference, and evaluation.

---

## 7. Explainability analysis

See files

```text
/notebooks/demonstration_shap.ipynb
```

and

```text
/notebooks/demonstration_grad_cam.ipynb
```

for examples of how to perform explainability analysis of the random forest 
and ResNet models.

---