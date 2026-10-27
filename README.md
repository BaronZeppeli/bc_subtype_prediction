# Breast cancer molecular subtype + PET/MRI

**Manuscript title:** Non-invasive prediction of breast cancer molecular subtypes by AI-driven analysis of [18F]FDG PET/MRI data
<!-- **Journal:** Cell Reports Medicine -->
<!--**Authors:** [Author names]  -->
<!--**Corresponding author:** [Name and institutional email]  -->
<!-- **Repository version:** [Version number] -->
**Last updated:** [2026-10-27]

---

## 1. Overview

This repository contains the source code, trained machine-learning models, and supporting materials used in the study:

> **Non-invasive prediction of breast cancer molecular subtypes by AI-driven analysis of [18F]FDG PET/MRI data**

The purpose of this repository is to document the implementation of the machine-learning methods described in the manuscript and facilitate reproducibility of the reported results.

The repository includes code for data preprocessing, model development, model training, inference, and evaluation, as applicable.

---

## 2. Repository Structure

```text
.
├── README.md
├── environment.yml
├── src/
│   ├── preprocessing.py
│   ├── ...
├── models/
│   └── [trained_model_weights]
├── notebooks/
│   └── demonstration.ipynb
└── evaluation/
    └── [evaluation_results]
```

---

## 3. Software Requirements

The code was developed and tested using the following environment:

- **Operating system:** [e.g., Ubuntu 22.04 / Windows 11 / macOS]
- **Programming language:** [e.g., Python 3.11]
- **Machine-learning framework:** [e.g., PyTorch 2.x / TensorFlow 2.x / scikit-learn]
- **Hardware:** [e.g., CPU / NVIDIA GPU model]
- **CUDA version, if applicable:** [Version]

### Dependencies

Install the required dependencies using:

```bash
conda env create -f environment.yml
conda activate bc_subtpype_prediction
```

For reproducibility, use the dependency versions specified in the supplied environment file.

---

## 4. Data Availability and Preparation

The analyses described in the manuscript use [brief description of the datasets].

**Data source:** [Dataset name, accession number, DOI, or repository URL]

**Data access:** [Publicly available / Available under controlled access / Available upon reasonable request, subject to applicable restrictions]

The raw data are [included in this repository / available from the external source identified above / not redistributed because of privacy, consent, licensing, or other restrictions].

### Data preprocessing

The preprocessing pipeline includes the following steps, as applicable:

1. Data loading and quality control.
2. Handling of missing values.
3. Feature extraction and selection.
4. Data normalization or transformation.
5. Training, validation, and test-set preparation.

Implementation details are provided in `[path/to/preprocessing_script]`.

No confidential, identifiable, or otherwise restricted data should be uploaded to this repository.

---

## 5. Model Architecture

The repository implements the following model(s):

| Model | Purpose | Implementation |
|---|---|---|
| [Model 1] | [Primary prediction task] | [File path] |
| [Model 2] | [Comparator or baseline] | [File path] |

### Model inputs

- **Input data:** [Description]
- **Input dimensions or feature representation:** [Description]
- **Preprocessing requirements:** [Description]

### Model outputs

- **Prediction target:** [Description]
- **Output format:** [e.g., class probabilities, predicted labels, continuous scores]
- **Interpretation:** [Description]

Additional architectural details, hyperparameters, and training procedures are described in the manuscript and the configuration files included in this repository.

---

## 6. Model Training

To train the model, first configure the relevant parameters in:

```text
configs/model_config.yaml
```

Then execute:

```bash
python src/train.py --config configs/model_config.yaml
```

The training procedure uses the following settings:

- **Training dataset:** [Description]
- **Validation strategy:** [Description]
- **Loss function:** [Description]
- **Optimizer:** [Description]
- **Learning rate:** [Value or configuration reference]
- **Batch size:** [Value]
- **Number of epochs or iterations:** [Value]
- **Random seed:** [Value, if applicable]
- **Checkpoint selection criterion:** [Description]

The trained model weights are saved to `[output_directory]`.

**Note:** The command above is a template. Adjust it to match the actual arguments accepted by the training script.

---

## 7. Loading the Trained Model and Making Predictions

If trained model weights are provided, they can be loaded using the corresponding model implementation.

Example usage:

```python
# Illustrative example: adapt to the model implementation.

from pathlib import Path

MODEL_PATH = Path("models/[trained_model_weights]")

# Initialize the model architecture.
# model = YourModel(...)

# Load the trained parameters using the framework-specific procedure.
# model.load_state_dict(...)
# model.eval()

# Prepare input data using the documented preprocessing pipeline.
# predictions = model.predict(input_data)
```

See `[path/to/prediction_script]` for the complete inference procedure.

Predictions should be generated using the same required preprocessing steps and input conventions described in the manuscript.

---

## 8. Model Evaluation

The evaluation code is provided in:

```text
src/evaluate.py
```

To run the evaluation:

```bash
python src/evaluate.py --config configs/model_config.yaml
```

The evaluation procedure uses the following metrics, as applicable:

- [Metric 1, e.g., AUROC]
- [Metric 2, e.g., AUPRC]
- [Metric 3, e.g., sensitivity]
- [Metric 4, e.g., specificity]
- [Additional metrics]

The reported results correspond to [brief description of the evaluation protocol, test cohort, validation scheme, or external validation dataset].

Evaluation outputs are saved to `[output_directory]`.

All commands should be adjusted to match the actual implementation.

---

## 9. Reproducing the Results

To reproduce the analyses reported in the manuscript:

1. Obtain the source code from this repository.
2. Install the specified software dependencies.
3. Obtain the required input data from the sources described in Section 4.
4. Apply the documented preprocessing pipeline.
5. Load the provided trained model or train the model using the supplied scripts.
6. Execute the evaluation procedure.
7. Compare the resulting metrics and outputs with the results reported in the manuscript.

The scripts associated with the principal results are listed below.

| Manuscript result | Corresponding script or notebook | Expected output |
|---|---|---|
| [Figure 1 / Table 1] | [File path] | [Description] |
| [Figure 2 / Table 2] | [File path] | [Description] |
| [Supplementary Figure S1] | [File path] | [Description] |

Exact numerical agreement may depend on software versions, hardware, random initialization, and other implementation details.

---

## 10. Model Weights and Checkpoints

The repository contains the following trained model artifacts, where applicable:

| File | Description | Format |
|---|---|---|
| [Model file 1] | [Description] | [e.g., .pth] |
| [Model file 2] | [Description] | [e.g., .keras] |

For each artifact, the associated architecture, configuration, preprocessing requirements, and intended use should be documented.

**Model availability:** [Included in this repository / Available from a linked repository / Available upon reasonable request, subject to applicable restrictions]

Where appropriate, provide file checksums or other integrity-verification information.

---

## 11. Ethical, Privacy, and Security Considerations

This repository is intended to support the transparent evaluation and reproducibility of the computational methods described in the manuscript.

Data and model artifacts are shared subject to applicable ethical approvals, participant consent, institutional policies, data-use agreements, and licensing restrictions.

No identifiable or unauthorized confidential information should be included in publicly accessible files.

Where model weights could reveal sensitive information about training participants, their release should be assessed under the applicable ethical and data-governance requirements.

---