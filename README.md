# Optimization of Accuracy in Political Fake News Detection

This project aims to improve accuracy in the detection of political fake news through the application and comparison of different optimization algorithms applied to logistic regression.

## Project Overview
Several gradient-based optimization algorithms were implemented and tuned, including:
- Gradient Descent (GD)
- Stochastic Gradient Descent (SGD)
- Mini-Batch Gradient Descent (MBGD)
- Adagrad
- Adam
- RMSProp

The dataset consists of **74,276 records**, divided into three subsets:
- **70% (Training):** Data used to train the model.
- **15% (Validation):** Data used for hyperparameter tuning.
- **15% (Test):** Data used to evaluate the final model performance.

## Data and Preprocessing
A preprocessed dataset was used, following these main steps:
- **Text vectorization:** `TfidfVectorizer` was used to transform text into numerical feature vectors.
- **Feature selection:** Implemented using `SelectKBest` with the chi-square test.
- **Standardization:** `StandardScaler` was applied to normalize the data.

## Libraries Used
The primary libraries used in this project were:
```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from scipy.sparse.linalg import norm
from scipy.stats import zscore
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
```

## Methodology
1. **Data Splitting:**
   - **70% Training:** 51,993 records
   - **15% Validation:** 11,141 records
   - **15% Test:** 11,142 records

2. **Model Training:**
   - Implementation of multiple optimization algorithms.
   - Monitoring of key metrics such as `accuracy`, `precision`, `f1_score`, along with confusion matrix analysis.

3. **Algorithm Optimization:**
   - Comparison of optimization algorithms to identify the approach that best balances accuracy and computational efficiency.
   - Evaluation metrics obtained on both validation and test subsets.

### Key Metrics
The primary evaluation metrics included:
- **Accuracy**
- **Precision**
- **F1 Score**
- **Confusion Matrix**

### Algorithm Comparison
A comparative evaluation was conducted to assess the performance of each optimization algorithm in terms of:
- Convergence
- Stability
- Final accuracy

## Results Visualization
The generated visualizations included:
- **Learning curves:** To monitor algorithm convergence.
- **Confusion matrix:** To analyze classification errors.

## Requirements
- Python 3.8+
- Required libraries:
```bash
pip install numpy pandas scikit-learn torch matplotlib
  ```

## Execution
1. Clone the repository:
   ```bash
   git clone <URL del repositorio>
   ```
2.	Install the required libraries.
3.	Run the script using the Python 3.11 kernel.

## Créditos
Project developed as part of an effort to improve fake news detection using machine learning and optimization techniques.
---
**Nota:** This README is a summary of the work conducted and may be updated with additional details regarding experimental results.
