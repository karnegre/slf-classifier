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
1. Feature extraction
   run feature extraction script
	python src/massfeatureextract.py
   it will pull images from the severson and dtd folders
   extracted features will be saved into the data folder with the current date. 

2. Model training and evaluation
   make sure data.xlsx has been copied and pasted to include all egg and non egg texture features
   run classifier script
	python src/classifier.py
   script will load and process data, train an SVM classifier with hypertuning, save the best model and evaluation plots to output


