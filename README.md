## Satellite-Imagery-Based-Property-Valuation

📌 Project Overview

This project focuses on predicting property prices by combining tabular real-estate attributes with satellite imagery. The core idea is to evaluate whether visual information extracted from satellite images adds predictive value beyond traditional tabular features. A multimodal machine learning pipeline was designed, trained, and evaluated to compare Tabular-only and Tabular + Image-based models.

#### 📂 Dataset Description

The project uses two primary data sources:

Tabular Data

Property-related attributes (e.g., area, location-based indicators, structural features)

Target variable: price

Unique identifier: id

Satellite Images

One satellite image per property

Images aligned with tabular data using the same id

#### 🧠 Methodology
1. Exploratory Data Analysis (EDA)

Analyzed price distribution, skewness, and outliers

Studied correlations between numerical features and price

Visual inspection of satellite images across different price ranges

2. Image Preprocessing & Feature Extraction

Images resized and normalized

A pretrained ResNet-50 CNN was used to extract deep visual embeddings

Grad-CAM was applied to visualize regions influencing model attention

Extracted embeddings were reduced using PCA to obtain compact image features

3. Feature Engineering

Cleaned tabular data by removing duplicates and handling missing values

Applied scaling and encoding to tabular features

Merged PCA-reduced image features with tabular data using property id

Ensured identical preprocessing for training, validation, and test sets

4. Modeling Strategy

Two modeling tracks were evaluated:

🔹 Tabular Models

Linear Regression

Ridge Regression

Lasso Regression

Random Forest Regressor

Gradient Boosting Regressor

XGBoost Regressor

🔹 Multimodal Models (Tabular + Image)

Random Forest Regressor

Gradient Boosting Regressor

XGBoost Regressor

All models were trained using scikit-learn Pipelines, ensuring that preprocessing steps were consistently applied.

#### 📊 Results Summary
Data Type	Best Model	RMSE ↓	MAE ↓	R² ↑

Tabular Only	XGBoost	~105K	~63K	0.91

Multimodal	XGBoost	~109K	~66K	0.90

🔹 Tabular models performed slightly better overall

🔹 Satellite imagery provided complementary spatial context but did not significantly outperform tabular-only models

🔹 Grad-CAM visualizations revealed meaningful spatial patterns in high-value properties

#### 🔍 Interpretability (Grad-CAM)

Grad-CAM was used to interpret CNN attention on satellite images

High-value properties showed focus on structured layouts and open spaces

Low-value properties highlighted dense or irregular regions

These visual cues align with real-world valuation factors

#### 📁 Project Structure
Satellite Imagery Based Property Valuation/
│

├── data/
│   ├── train_tabular.csv

│   ├── test_tabular.csv

│   ├── image_embeddings_train.csv

│   └── image_embeddings_test.csv
│

├── images/
│   ├── train/

│   └── test/
│

├── notebooks/
│   ├── EDA.ipynb

│   ├── Image_Preprocessing.ipynb

│   ├── Feature_Engineering.ipynb

│   └── Model_Training.ipynb
│

├── models/
│   └── best_model.pkl
│

├── final_submission.csv

└── README.md

#### 🚀 How to Run

The project is organized into three sequential stages, each implemented as a separate notebook/script. These stages must be executed in the order listed below, as each step depends on the outputs of the previous one.

All intermediate and final datasets are saved in structured folders on Google Drive, and the notebooks are connected via file paths to access these saved files.

1️⃣ Data Fetcher

Runs the data fetching pipeline

Downloads satellite images for each property using their unique id

Stores images in structured train/test directories on Google Drive

Run first to ensure all images are available for downstream steps.

2️⃣ Preprocessing

Performs tabular data cleaning and feature engineering

Preprocesses satellite images

Extracts CNN-based image embeddings

Applies PCA to reduce image feature dimensionality

Merges image features with tabular data using property id

Saves processed datasets for modeling

Run after the data fetcher.

3️⃣ Model Training

Trains multiple machine learning models on:

Tabular data only

Tabular + image (multimodal) data

Evaluates performance using RMSE, MAE, and R²

Selects the best-performing model

Generates final price predictions and submission file

Run last, once preprocessing is complete.

#### 📌 Key Takeaways

Strong tabular features can outperform multimodal approaches in structured datasets

Satellite imagery adds interpretability and spatial insights

Multimodal learning is valuable for explainability even when performance gains are marginal

🛠️ Tools & Libraries

Python, Pandas, NumPy

Scikit-learn

PyTorch, Torchvision

XGBoost

OpenCV, Matplotlib, Seaborn

#### ✨ Author

Rishabh Singhal
Satellite Imagery | Machine Learning | Multimodal Modeling
