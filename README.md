# SLF Classifier

A machine learning pipeline for classifying spotted lanternfly egg mass images using texture-based features with Support Vector Machine (SVM) modeling.

---

## Overview

This project provides a complete workflow to:

- Extract multiple texture descriptors from grayscale images, including GLCM, GLDS, LBP, Zernike moments, Hu moments, and TAS.
- Perform feature selection and dimensionality reduction.
- Train an optimized SVM classifier with hyperparameter tuning.
- Evaluate and visualize model performance with comprehensive plots.

---

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Required Python packages listed in `requirements.txt`

---

### Installation

```bash
git clone https://github.com/karnegre/slf-classifier.git
cd slf-classifier
pip install -r requirements.txt
```

---

## Usage

### 1. Feature Extraction

Place your image sets inside the `image_sets/` directory under subfolders (e.g., `severson/`, `dtd/`).

Run the feature extraction script:

```bash
python src/massfeatureextract.py
```

- The script will extract texture features from all images in the folders.
- Features are saved as an Excel file in the `data/` directory with the current date in the filename.

**Non-egg images are sourced from:**

- Severstal Steel Defect Detection Dataset:  
  https://www.kaggle.com/competitions/severstal-steel-defect-detection/overview
- Describable Textures Dataset (DTD):  
  https://www.robots.ox.ac.uk/~vgg/data/dtd/index.html#citation

---

### 2. Model Training and Evaluation

Ensure your `.xlsx` file (with both egg and non-egg features) is saved in the `data/` directory.

Run the classification pipeline:

```bash
python src/classifier.py
```

This script will:

- Load and preprocess the feature data
- Train an SVM classifier with hyperparameter tuning
- Save the best model to `models/final_model.pkl`
- Generate and save evaluation plots (ROC, PR curves, confusion matrix, PCA, etc.) to the `outputs/` directory

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


