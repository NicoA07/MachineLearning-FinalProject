# Introduction to Machine Learning Final Project, NYCU Fall 2022

## Overview
The Final Project is to compete the [The August 2022 edition of the Tabular Playground Series](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/overview), which we need to use the [provided dataset](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/data) to build a model that predicts product failures.
My implementation involves developing a logistic regression model for the provided dataset. The project makes use of various Python libraries for data preprocessing and feature selection and includes the following key features:

* Imputation of missing values using KNNImputer
* Feature scaling
* Feature selection using feature engineering
* k-fold cross-validation for model evaluation
* Calculation of area under the receiver operating characteristic (ROC) curve (AUC) and accuracy scores
* Implementation of additional features researched and implemented based on the analysis of feature correlations and discussion on Kaggle platforms.

## Environment Details
The following libraries are needed to run the code:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import math

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate, StratifiedKFold, RepeatedStratifiedKFold

from feature_engine.encoding import WoEEncoder

from colorama import Fore, Back, Style

from joblib import dump
from joblib import load
```
You can install these libraries by running by running `pip install -r requirements.txt` or by installing them individually using `pip install library-name`.
```python
# Requirements #
!pip install -r requirements.txt
# or
!pip install numpy pandas matplotlib scikit-learn joblib feature_engine colorama
```

## Implementation Details
The main notebook for the project is [`109550200_Final_train.ipynb`](), which contains the implementation of the logistic regression model. The notebook is organized into the following sections:

* Data Preprocessing
* Model Training
* Model Evaluation
* Feature Importance Analysis

The data preprocessing step includes imputation of missing values using KNNImputer and feature scaling. The model training step includes the implementation of k-fold cross-validation, training of the logistic regression model with selected features, and the calculation of AUC and accuracy scores. The model evaluation step includes the computation of out-of-fold predictions and the analysis of feature importances. The additional features that were researched and implemented based on the analysis of feature correlations and discussion on Kaggle platforms are also added in the train model function.

You can find the [full report]() in the repo as well which provides more detail on each step. 

## Inference
The repository includes the [pre-trained model](), so you can use the pre-trained model to make predictions directly on the dataset by running [`109550200_Final_inference.ipynb`]().

## Result
![]()
![]()
