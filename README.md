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

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/karnegre/slf-classifier.git
   cd slf-classifier

### Usage
1. Feature Extraction
Place your image sets inside the image_sets/ directory under subfolders (e.g., severson/, dtd/).

Run the feature extraction script:

bash
Copy
Edit
python src/massfeatureextract.py
The script will extract texture features from all images in the folders and save the output as an Excel file in the data/ directory, named with the current date.

Non-egg images are sourced from:

Severstal Steel Defect Detection Dataset

Describable Textures Dataset (DTD)

2. Model Training and Evaluation
Ensure your consolidated .xlsx file (containing both egg and non-egg features) is saved in the data/ directory.

Run the classification pipeline:

bash
Copy
Edit
python src/classifier.py
This script will:

Load and preprocess the data

Train an SVM classifier with hyperparameter tuning

Save the best model to models/final_model.pkl

Generate evaluation plots (e.g., ROC curves, precision-recall curves, confusion matrix, PCA visualization) and save them in the outputs/ directory


